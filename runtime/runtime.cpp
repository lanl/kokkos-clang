#include <iostream>
#include <cmath>
#include <thread>
#include <mutex>
#include <atomic>
#include <vector>
#include <functional>
#include <cassert>
#include <deque>
#include <queue>
#include <map>
#include <unordered_map>
#include <cstring>

#include <pthread.h>
#include <sys/time.h>

#include <cuda.h>

//#define COPY 1

using namespace std;
//using namespace ideas;

#define ndump(X) cout << __FILE__ << ":" << __LINE__ << ": " << \
__PRETTY_FUNCTION__ << ": " << #X << " = " << (X) << std::endl

#define nlog(X) std::cout << __FILE__ << ":" << __LINE__ << ": " << \
__PRETTY_FUNCTION__ << ": " << (X) << std::endl

#define check(err) if(err != CUDA_SUCCESS){ \
      const char* s; \
      cuGetErrorString(err, &s); \
      cerr << "CUDARuntime error: " << s << endl; \
      assert(false); \
    }

namespace{

  const size_t NUM_THREADS = 16;

  const size_t DEFAULT_THREADS = 128;

  const uint8_t FIELD_READ = 0x01;
  const uint8_t FIELD_WRITE = 0x02;

  const size_t BLOCK_SIZE = 256;

  const uint32_t REDUCE_KERNEL = 0b1;

  class VSem{
  public:
    
    VSem(int count=0)
    : count_(count),
    maxCount_(0){

      pthread_mutex_init(&mutex_, 0);
      pthread_cond_init(&condition_, 0);
    }
    
    VSem(int count, int maxCount)
    : count_(count),
    maxCount_(maxCount){

      pthread_mutex_init(&mutex_, 0);
      pthread_cond_init(&condition_, 0);
    }
    
    ~VSem(){
      pthread_cond_destroy(&condition_);
      pthread_mutex_destroy(&mutex_);
    }

    bool acquire(double dt){
      timeval tv;
      gettimeofday(&tv, 0);
      
      double t = tv.tv_sec + tv.tv_usec/1e6;
      t += dt;
      
      pthread_mutex_lock(&mutex_);
            
      double sec = floor(t);
      double fsec = t - sec;

      timespec ts;
      ts.tv_sec = sec;
      ts.tv_nsec = fsec*1e9;

      while(count_ <= 0){
        if(pthread_cond_timedwait(&condition_, 
                                  &mutex_,
                                  &ts) != 0){
          pthread_mutex_unlock(&mutex_);
          return false;
        }
      }
      
      --count_;
      pthread_mutex_unlock(&mutex_);
      
      return true;
    }
    
    bool acquire(){
      pthread_mutex_lock(&mutex_);
      
      while(count_ <= 0){
        pthread_cond_wait(&condition_, &mutex_);
      }
      
      --count_;
      pthread_mutex_unlock(&mutex_);
      
      return true;
    }
    
    bool tryAcquire(){
      pthread_mutex_lock(&mutex_);
      
      if(count_ > 0){
        --count_;
        pthread_mutex_unlock(&mutex_);
        return true;
      }

      pthread_mutex_unlock(&mutex_);
      return false;
    }
    
    void release(){
      pthread_mutex_lock(&mutex_);

      if(maxCount_ == 0 || count_ < maxCount_){
        ++count_;
      }

      pthread_cond_signal(&condition_);
      pthread_mutex_unlock(&mutex_);
    }
    
    VSem& operator=(const VSem&) = delete;
    
    VSem(const VSem&) = delete;

  private:
    pthread_mutex_t mutex_;
    pthread_cond_t condition_;

    int count_;
    int maxCount_;
  };

  class Synch{
  public:
    Synch(int count)
    : sem_(-count){}

    void release(){
      sem_.release();
    }

    void await(){
      sem_.acquire();
    }

  private:
    VSem sem_;
  };

  struct FuncArg{
    FuncArg(Synch* synch, int n)
      : n(n),
      synch(synch){}

    Synch* synch;
    int n;
  };

  using Func = function<void(FuncArg*)>;
  using FuncPtr = void (*)(FuncArg*);

  class ThreadPool{
  public:
    class Queue{
    public:

      class Item{
      public:
        Item(Func func, FuncArg* arg, uint32_t priority)
        : func(func),
        arg(arg),
        priority(priority){}

        Func func;
        FuncArg* arg;
        uint32_t priority;
      };

      void push(Func func, FuncArg* arg, uint32_t priority){
        mutex_.lock();
        queue_.push(new Item(func, arg, priority));
        mutex_.unlock();
        
        sem_.release();
      }
      
      Item* get(){
        if(!sem_.acquire()){
          return nullptr;
        }

        mutex_.lock();
        Item* item = queue_.top();
        queue_.pop();
        mutex_.unlock();
        return item;
      }

    private:
      struct Compare_{
        bool operator()(const Item* i1, const Item* i2) const{
          return i1->priority < i2->priority;
        }
      };
      
      typedef priority_queue<Item*, vector<Item*>, Compare_> Queue_;

      Queue_ queue_;
      VSem sem_;
      mutex mutex_;
    };

    ThreadPool(){
      start();
    }

    void push(Func func, FuncArg* arg, uint32_t priority){
      queue_.push(func, arg, priority);
    }

    void start(){
      for(size_t i = 0; i < NUM_THREADS; ++i){
        threadVec_.push_back(new thread(&ThreadPool::run_, this));  
      }
    }

    void run_(){
      for(;;){
        Queue::Item* item = queue_.get();
        assert(item);
        item->func(item->arg);
        delete item;
      }
    }

  private:
    using ThreadVec = std::vector<thread*>;

    Queue queue_;

    mutex mutex_;

    ThreadVec threadVec_;
  };

  ThreadPool* _threadPool = new ThreadPool;

  template<class T>
  void sum_(T* hostPtr, CUdeviceptr& ptr, size_t size, void* resultPtr){
    size_t bytes = size * sizeof(T);

    CUresult err = cuMemcpyDtoH(hostPtr, ptr, bytes);
    check(err);

    T result = 0;
    for(size_t i = 0; i < size; ++i){
      result += hostPtr[i];
    }

    memcpy(resultPtr, &result, sizeof(T));
  }

  template<class T>
  void product_(T* hostPtr, CUdeviceptr& ptr, size_t size, void* resultPtr){
    size_t bytes = size * sizeof(T);

    CUresult err = cuMemcpyDtoH(hostPtr, ptr, bytes);
    check(err);

    T result = 1;
    for(size_t i = 0; i < size; ++i){
      result *= hostPtr[i];
    }

    memcpy(resultPtr, &result, sizeof(T));
  }

  void reduce(void* hostPtr,
              CUdeviceptr& ptr,
              size_t size,
              size_t scalarBytes,
              bool isFloat,
              bool isSigned,
              bool isSum,
              void* resultPtr){
    switch(scalarBytes){
      case 8:
        if(isFloat){
          if(isSum){
            sum_<double>((double*)hostPtr, ptr, size, resultPtr);
          }
          else{
            product_<double>((double*)hostPtr, ptr, size, resultPtr);
          }
        }
        else{
          if(isSigned){
            if(isSum){
              sum_<int64_t>((int64_t*)hostPtr, ptr, size, resultPtr);
            }
            else{
              product_<int64_t>((int64_t*)hostPtr, ptr, size, resultPtr);
            }
          }
          else{
            if(isSum){
              sum_<uint64_t>((uint64_t*)hostPtr, ptr, size, resultPtr);
            }
            else{
              product_<uint64_t>((uint64_t*)hostPtr, ptr, size, resultPtr);
            }
          }
        }
        break;
      case 4:
        if(isFloat){
          if(isSum){
            sum_<float>((float*)hostPtr, ptr, size, resultPtr);
          }
          else{
            product_<float>((float*)hostPtr, ptr, size, resultPtr);
          }
        }
        else{
          if(isSigned){
            if(isSum){
              sum_<int32_t>((int32_t*)hostPtr, ptr, size, resultPtr);
            }
            else{
              product_<int32_t>((int32_t*)hostPtr, ptr, size, resultPtr);
            }
          }
          else{
            if(isSum){
              sum_<uint32_t>((uint32_t*)hostPtr, ptr, size, resultPtr);
            }
            else{
              product_<uint32_t>((uint32_t*)hostPtr, ptr, size, resultPtr);
            }
          }
        }
        break;
      case 2:
        if(isFloat){
          assert(false);
        }
        else{
          if(isSigned){
            if(isSum){
              sum_<int16_t>((int16_t*)hostPtr, ptr, size, resultPtr);
            }
            else{
              product_<int16_t>((int16_t*)hostPtr, ptr, size, resultPtr);
            }
          }
          else{
            if(isSum){
              sum_<uint16_t>((uint16_t*)hostPtr, ptr, size, resultPtr);
            }
            else{
              product_<uint16_t>((uint16_t*)hostPtr, ptr, size, resultPtr);
            }
          }
        }
        break;
      case 1:
        if(isFloat){
          assert(false);
        }
        else{
          if(isSigned){
            if(isSum){
              sum_<int8_t>((int8_t*)hostPtr, ptr, size, resultPtr);
            }
            else{
              product_<int8_t>((int8_t*)hostPtr, ptr, size, resultPtr);
            }
          }
          else{
            if(isSum){
              sum_<uint8_t>((uint8_t*)hostPtr, ptr, size, resultPtr);
            }
            else{
              product_<uint8_t>((uint8_t*)hostPtr, ptr, size, resultPtr);
            }
          }
        }
        break;
      default:
        assert(false);
    }
  }

  class CUDARuntime{
  public:
    class View{
    public:
      View(void* hostPtr)
      : hostPtr_(hostPtr){}

      ~View(){
        CUresult err = cuMemFree(devPtr_);
        check(err);
      }

      void setDevPtr(CUdeviceptr ptr){
        devPtr_ = ptr;
      }

      void setSize(size_t size){
        size_ = size;
      }

      CUdeviceptr& devPtr(){
        return devPtr_;
      }

      CUdeviceptr& dimsPtr(){
        return dimsPtr_;
      }

      void* hostPtr(){
        return hostPtr_;
      }

      size_t size(){
        return size_;
      }

      vector<uint32_t>& dims(){
        return dims_;
      }

      void pushDim(uint32_t dim){
        dims_.push_back(dim);
      }

    private:
      void* hostPtr_;
      CUdeviceptr devPtr_;
      CUdeviceptr dimsPtr_;
      size_t size_;
      vector<uint32_t> dims_;
    };

    class KernelView{
    public:
      KernelView(View* view, uint32_t flags)
      : view_(view),
      flags_(flags){}

      bool isRead(){
        return flags_ & FIELD_READ;
      }

      bool isWrite(){
        return flags_ & FIELD_WRITE;
      }

      View* view(){
        return view_;
      }

     private:
       View* view_; 
       uint32_t flags_;
    };

    class Kernel{
    public:
      Kernel(char* ptx,
             uint32_t reduceSize,
             bool reduceFloat,
             bool reduceSigned,
             bool reduceSum)
        : ready_(false),
          reduceSize_(reduceSize),
          reduceFloat_(reduceFloat),
          reduceSigned_(reduceSigned),
          reduceSum_(reduceSum),
          lastSize_(0),
          numThreads_(DEFAULT_THREADS){
        
        CUresult err = cuModuleLoadData(&module_, (void*)ptx);
        check(err);

        if(reduceSize_ > 0){
          err = cuModuleGetFunction(&function_, module_, "reduce");
        }
        else{
          err = cuModuleGetFunction(&function_, module_, "run");
        }

        check(err);
      }

      ~Kernel(){

      }
      
      void setNumThreads(size_t numThreads){
        numThreads_ = numThreads;
      }

      void addView(View* view, uint32_t flags){
        auto itr = viewMap_.find(view);
        if(itr != viewMap_.end()){
          return;
        }

        auto kernelView = new KernelView(view, flags);
        viewMap_[view] = kernelView;
        views_.push_back(kernelView);
      }

      void addVar(void* varPtr, size_t size){
        (void)size;
        vars_.push_back(varPtr);
      }

      void run(uint32_t n, void* reduceRetPtr){
        n_ = n;
        size_t gridDimX;
        size_t sharedMemBytes;

        if(reduceRetPtr){
          gridDimX = (n_ + (BLOCK_SIZE * 2 - 1))/(BLOCK_SIZE * 2);
          
          if(n_ != lastSize_){
            if(lastSize_ > 0){
              free(hostPtr_);
              CUresult err = cuMemFree(reducePtr_);
              check(err);
            }

            hostPtr_ = malloc(gridDimX * reduceSize_);
            CUresult err = cuMemAlloc(&reducePtr_, gridDimX * reduceSize_);
            check(err);
          }

          sharedMemBytes = reduceSize_ * gridDimX;
        }
        else{
          gridDimX = (n_ + BLOCK_SIZE - 1)/BLOCK_SIZE;
          sharedMemBytes = 0;
        }

        if(!ready_){
          for(KernelView* kernelView : views_){
            View* view = kernelView->view();

            kernelParams_.push_back(&view->devPtr());
            
            auto& dims = view->dims();

            /*
            for(auto di : dims){
              ndump(di);
            }
            */

            CUresult err = cuMemAlloc(&view->dimsPtr(), dims.size() * 4);
            check(err);

            err = cuMemcpyHtoD(view->dimsPtr(), dims.data(),
                               dims.size() * 4);
            check(err);

            kernelParams_.push_back(&view->dimsPtr());
          }

          for(void* varPtr : vars_){
            kernelParams_.push_back(varPtr);
          }

          kernelParams_.push_back(&n_);

          if(reduceSize_ > 0){
            kernelParams_.push_back(&reducePtr_);
          }

          ready_ = true;
        }

        CUresult err;

#ifdef COPY
        for(KernelView* kernelView : views_){
          View* view = kernelView->view();

          /*
          double* dp = (double*)view->hostPtr();
          for(size_t i = 0; i < view->size()/8; ++i){
            cout << "dp[" << i << "] = " << dp[i] << endl;
          }
          */

          if(kernelView->isRead()){
            err = cuMemcpyHtoD(view->devPtr(), view->hostPtr(),
                               view->size());
            check(err);
          }
        }
#endif


        //CUstream stream;
        //cuStreamCreate(&stream, 0);

        err = cuLaunchKernel(function_, gridDimX, 1, 1,
                             BLOCK_SIZE, 1, 1, 
                             sharedMemBytes, nullptr/*stream*/,
                             kernelParams_.data(), nullptr);

        //ndump(kernelParams_.size());

        //cuStreamSynchronize(stream);

        //cuStreamDestroy(stream);

        check(err);

#ifdef COPY
        for(KernelView* kernelView : views_){
          View* view = kernelView->view();

          //ndump(view->size());

          if(kernelView->isWrite()){
            err = cuMemcpyDtoH(view->hostPtr(), view->devPtr(),
                               view->size());
            check(err);
          }
        }
#endif

        if(reduceRetPtr){
          lastSize_ = n;

          reduce(hostPtr_, reducePtr_, gridDimX, reduceSize_, reduceFloat_,
                 reduceSigned_, reduceSum_, reduceRetPtr);
        }
      }
      
      bool ready(){
        return ready_;
      }
      
    private:
      using KernelParams_ = vector<void*>;
      using ViewVec_ = vector<KernelView*>;
      using VarVec_ = vector<void*>;      
      using ViewMap_ = unordered_map<View*, KernelView*>;

      CUmodule module_;    
      CUfunction function_;
      uint32_t reduceSize_;
      bool reduceFloat_;
      bool reduceSigned_;
      bool reduceSum_;
      void* hostPtr_;
      CUdeviceptr reducePtr_;
      CUdeviceptr fakeSharedPtr_;
      bool ready_;
      size_t numThreads_;
      KernelParams_ kernelParams_;
      ViewVec_ views_;
      VarVec_ vars_;
      ViewMap_ viewMap_;
      uint32_t n_;
      uint32_t lastSize_;  
    };

    CUDARuntime(){

    }

    ~CUDARuntime(){
      //CUresult err = cuCtxDestroy(context_);
      //check(err);
    }

    void init(){
      if(initalized_){
        return;
      }

      CUresult err = cuInit(0);
      check(err);
      
      err = cuDeviceGet(&device_, 0);
      check(err);

      err = cuCtxCreate(&context_, 0, device_);
      check(err);

      int threadsPerBlock;
      err = 
        cuDeviceGetAttribute(&threadsPerBlock,
                             CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device_);
      check(err);
      numThreads_ = threadsPerBlock;

      initalized_ = true;
    }

    bool initKernel(uint32_t kernelId,
                    char* ptx,
                    uint32_t reduceSize,
                    bool reduceFloat,
                    bool reduceSigned,
                    bool reduceSum){ 
      auto itr = kernelMap_.find(kernelId);
      if(itr != kernelMap_.end()){
        return true;
      }

      auto kernel = 
      new Kernel(ptx, reduceSize, reduceFloat, reduceSigned, reduceSum);
      
      kernelMap_[kernelId] = kernel;
      return false;
    }

    void addView(uint32_t kernelId,
                 void** viewPtr,
                 uint32_t elementSize,
                 uint32_t staticDims,
                 uint32_t* staticSizes,
                 uint32_t runTimeDims,
                 uint32_t flags){
      
      void* data = viewPtr[0];

      auto itr = viewMap_.find(data);
      
      View* view;

      if(itr != viewMap_.end()){
        view = itr->second;
      }
      else{
        view = new View(data);

        ++viewPtr;
        uint32_t* sizes = (uint32_t*)viewPtr;

        size_t size = elementSize;

        for(size_t i = 0; i < runTimeDims; ++i){
          uint32_t si = *sizes;
          size *= si;

          view->pushDim(si);

          ++sizes;
        }

        for(size_t i = 0; i < staticDims; ++i){
          uint32_t si = staticSizes[i];
          size *= si;

          view->pushDim(si);
        }

        CUdeviceptr devPtr;
        CUresult err = cuMemAlloc(&devPtr, size);
        check(err);

        view->setDevPtr(devPtr);
        view->setSize(size);

        viewMap_[data] = view;
      }

      auto kitr = kernelMap_.find(kernelId);
      assert(kitr != kernelMap_.end());

      Kernel* kernel = kitr->second;

      kernel->addView(view, flags);
    }

    void addVar(uint32_t kernelId, void* varPtr, size_t size){
      auto kitr = kernelMap_.find(kernelId);
      assert(kitr != kernelMap_.end());

      Kernel* kernel = kitr->second;

      kernel->addVar(varPtr, size);     
    }

    void addKernelView(uint32_t kernelId,
                       void** viewPtr,
                       uint32_t flags){

      auto itr = kernelMap_.find(kernelId);
      assert(itr != kernelMap_.end());

      Kernel* kernel = itr->second;
      
      auto vitr = viewMap_.find(viewPtr);
      assert(vitr != viewMap_.end());

      View* view = vitr->second;

      kernel->addView(view, flags);
    }

    void runKernel(uint32_t kernelId, uint32_t n, void* reducePtr){ 
      auto itr = kernelMap_.find(kernelId);
      assert(itr != kernelMap_.end());

      Kernel* kernel = itr->second;

      kernel->run(n, reducePtr);
    }

  private:
    using ViewMap_ = map<void*, View*>;
    using KernelMap_ = unordered_map<uint32_t, Kernel*>;

    bool initalized_ = false;

    CUdevice device_;
    CUcontext context_;
    size_t numThreads_;

    ViewMap_ viewMap_;
    KernelMap_ kernelMap_;
  };

  CUDARuntime _cudaRuntime;

} // namespace

extern "C"{

  void* __ideas_create_synch(uint32_t count){
    return new Synch(count - 1);
  }

  void __ideas_queue_func(void* synch, void* fp, int index, uint32_t priority){
    _threadPool->push(reinterpret_cast<FuncPtr>(fp),
      new FuncArg(reinterpret_cast<Synch*>(synch), index), priority);
  }

  void __ideas_finish_func(void* arg){
    auto a = reinterpret_cast<FuncArg*>(arg);
    a->synch->release();
    delete a;
  }

  void __ideas_await_synch(void* synch){
    auto s = reinterpret_cast<Synch*>(synch);
    s->await();
    delete s;
  }

  bool __ideas_cuda_init_kernel(uint32_t kernelId,
                                char* ptx,
                                uint32_t reduceSize,
                                bool reduceFloat,
                                bool reduceSigned,
                                bool reduceSum){
    _cudaRuntime.init();
    return _cudaRuntime.initKernel(kernelId, ptx, reduceSize,
                                   reduceFloat, reduceSigned, reduceSum);
  }

  void __ideas_cuda_add_view(uint32_t kernelId,
                             void** viewPtr,
                             uint32_t elementSize,
                             uint32_t staticDims,
                             uint32_t* staticSizes,
                             uint32_t runtimeDims,
                             uint32_t flags){
    _cudaRuntime.addView(kernelId, viewPtr, elementSize,
      staticDims, staticSizes, runtimeDims, flags);
  }

  void __ideas_cuda_add_var(uint32_t kernelId, void* varPtr){
    _cudaRuntime.addVar(kernelId, varPtr, 0);
  }

  void __ideas_cuda_run_kernel(uint32_t kernelId, uint32_t n, void* reducePtr){
    return _cudaRuntime.runKernel(kernelId, n, reducePtr);
  }

  void __ideas_debug1(void* ptr){

  }

} // extern "C"
