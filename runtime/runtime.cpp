/*
 * ###########################################################################
 * Copyright (c) 2016, Los Alamos National Security, LLC All rights
 * reserved. Copyright 2016. Los Alamos National Security, LLC. This
 * software was produced under U.S. Government contract DE-AC52-06NA25396
 * for Los Alamos National Laboratory (LANL), which is operated by Los
 * Alamos National Security, LLC for the U.S. Department of Energy. The
 * U.S. Government has rights to use, reproduce, and distribute this
 * software.  NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY,
 * LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY
 * FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
 * derivative works, such modified software should be clearly marked, so
 * as not to confuse it with the version available from LANL.
 *  
 * Additionally, redistribution and use in source and binary forms, with
 * or without modification, are permitted provided that the following
 * conditions are met: 1.       Redistributions of source code must
 * retain the above copyright notice, this list of conditions and the
 * following disclaimer. 2.      Redistributions in binary form must
 * reproduce the above copyright notice, this list of conditions and the
 * following disclaimer in the documentation and/or other materials
 * provided with the distribution. 3.      Neither the name of Los Alamos
 * National Security, LLC, Los Alamos National Laboratory, LANL, the U.S.
 * Government, nor the names of its contributors may be used to endorse
 * or promote products derived from this software without specific prior
 * written permission.
  
 * THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING,
 * BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL LOS
 * ALAMOS NATIONAL SECURITY, LLC OR CONTRIBUTORS BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
 * GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
 * IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE US
 * ########################################################################### 
 * 
 * Notes
 *
 * ##### 
 */

#include <stdio.h>
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

using namespace std;

#define np(X) cout << __FILE__ << ":" << __LINE__ << ": " << \
__PRETTY_FUNCTION__ << ": " << #X << " = " << (X) << std::endl

#define check(err) if(err != CUDA_SUCCESS){ \
      const char* s; \
      cuGetErrorString(err, &s); \
      cerr << "CUDARuntime error: " << s << endl; \
      assert(false); \
    }

namespace ideas{
  
  extern void reduce(size_t kernel,
                     size_t size,
                     size_t scalarBytes,
                     bool isFloat,
                     bool isSigned,
                     bool isSum,
                     void* args,
                     void* resultPtr);

} // namespace ideas

namespace{

  const size_t NUM_THREADS = 16;

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

      void copyToDevice(){
        //fprintf(stderr,"copy %p->%p %d\n",hostPtr_, devPtr_, size_);
        CUresult err = cuMemcpyHtoD(devPtr_, hostPtr_, size_);
        check(err);
      }

      void copyFromDevice(){
        //fprintf(stderr,"copy %p<-%p %d\n",hostPtr_, devPtr_, size_);
        CUresult err = cuMemcpyDtoH(hostPtr_, devPtr_, size_);
        check(err); 
      }

    private:
      void* hostPtr_;
      CUdeviceptr devPtr_;
      CUdeviceptr dimsPtr_;
      size_t size_;
      vector<uint32_t> dims_;
    };

    class Array{
    public:
      Array(void* hostPtr)
      : hostPtr_(hostPtr){}

      ~Array(){
        CUresult err = cuMemFree(devPtr_);
        check(err);
      }

      void setDevPtr(CUdeviceptr ptr){
        devPtr_ = ptr;
      }

      void setSize(uint32_t size){
        size_ = size;
      }

      CUdeviceptr& devPtr(){
        return devPtr_;
      }

      void* hostPtr(){
        return hostPtr_;
      }

      uint32_t size(){
        return size_;
      }

      void copyToDevice(){
        CUresult err = cuMemcpyHtoD(devPtr_, hostPtr_, size_);
        check(err);
      }

      void copyFromDevice(){
        CUresult err = cuMemcpyDtoH(hostPtr_, devPtr_, size_);
        check(err); 
      }

    private:
      void* hostPtr_;
      CUdeviceptr devPtr_;
      uint32_t size_;
    };

    struct Var{
      Var(void* hostPtr, size_t size)
      : hostPtr(hostPtr), size(size){}

      void* hostPtr;
      size_t size; 
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
          lastSize_(0){
        
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

      Kernel(bool kernel2,
             char* ptx,
             uint32_t reduceSize,
             bool reduceFloat,
             bool reduceSigned,
             bool reduceSum)
        : ready_(false),
          reduceSize_(reduceSize),
          reduceFloat_(reduceFloat),
          reduceSigned_(reduceSigned),
          reduceSum_(reduceSum),
          lastSize_(0){
        
        // ndm - finish implementing

        CUresult err = cuModuleLoadData(&module_, (void*)ptx);
        check(err);

        err = cuModuleGetFunction(&function_, module_, "run");
        check(err);

        err = cuStreamCreate(&stream_, 0);
        check(err);
      }

      ~Kernel(){
        cuStreamDestroy(stream_);
      }
      
      void setNumThreads(size_t numThreads){
        numThreads_ = numThreads;
      }

      void addView(View* view){
        views_.push_back(view);
      }

      void addArray(Array* array){
        arrays_.push_back(array);
      }

      void addVar(void* varPtr, size_t size){
        auto v = new Var(varPtr, size);
        vars_.push_back(v);
      }

      void run(uint32_t n, void* reduceRetPtr){
        reduceRetPtr_ = reduceRetPtr;
        n_ = n;
        size_t sharedMemBytes;

        if(reduceRetPtr_){
          gridDimX_ = (n_ + (BLOCK_SIZE * 2 - 1))/(BLOCK_SIZE * 2);
          
          if(n_ != lastSize_){
            if(lastSize_ > 0){
              free(hostPtr_);
              CUresult err = cuMemFree(reducePtr_);
              check(err);
            }

            hostPtr_ = malloc(gridDimX_ * reduceSize_);
            CUresult err = cuMemAlloc(&reducePtr_, gridDimX_ * reduceSize_);
            check(err);
          }

          sharedMemBytes = reduceSize_ * gridDimX_;
        }
        else{
          gridDimX_ = (n_ + BLOCK_SIZE - 1)/BLOCK_SIZE;
          sharedMemBytes = 0;
        }

        if(!ready_){
          for(View* view : views_){
            kernelParams_.push_back(&view->devPtr());
            
            auto& dims = view->dims();

            CUresult err = cuMemAlloc(&view->dimsPtr(), dims.size() * 4);
            check(err);

            err = cuMemcpyHtoD(view->dimsPtr(), dims.data(),
                               dims.size() * 4);
            check(err);

            kernelParams_.push_back(&view->dimsPtr());
          }

          for(Array* array : arrays_){
            kernelParams_.push_back(&array->devPtr());
          }

          for(Var* var : vars_){
            kernelParams_.push_back(var->hostPtr);
          }

          kernelParams_.push_back(&n_);

          if(reduceSize_ > 0){
            kernelParams_.push_back(&reducePtr_);
          }

          ready_ = true;
        }

        CUresult err;

        err = cuLaunchKernel(function_, gridDimX_, 1, 1,
                             BLOCK_SIZE, 1, 1, 
                             sharedMemBytes, stream_,
                             kernelParams_.data(), nullptr);

        check(err);

        lastSize_ = n;
      }

      void await(){
        cuStreamSynchronize(stream_);

        if(reduceRetPtr_){
          reduce(hostPtr_, reducePtr_, gridDimX_, reduceSize_,
                 reduceFloat_, reduceSigned_, reduceSum_, reduceRetPtr_);
        }
      }

      void runReduce(uint32_t n, void* reduceRetPtr){
        reduceRetPtr_ = reduceRetPtr;
        n_ = n;
        CUdeviceptr ptr;
        
        vector<CUdeviceptr> fields;

        if(!ready_){
          for(View* view : views_){
            fields.push_back(view->devPtr());
            
            auto& dims = view->dims();

            CUresult err = cuMemAlloc(&view->dimsPtr(), dims.size() * 4);
            check(err);

            err = cuMemcpyHtoD(view->dimsPtr(), dims.data(),
                               dims.size() * 4);
            check(err);

            fields.push_back(view->dimsPtr());
          }

          for(Array* array : arrays_){
            fields.push_back(array->devPtr());
          }

          for(Var* var : vars_){
            CUdeviceptr devPtr;
            CUresult err = cuMemAlloc(&devPtr, var->size);
            check(err);

            fields.push_back(devPtr);
          }

          size_t bytes = fields.size() * sizeof(CUdeviceptr);

          CUresult err = cuMemAlloc(&reduceArgsDevPtr_, bytes);
          check(err);

          err = cuMemcpyHtoD(reduceArgsDevPtr_, fields.data(), bytes);
          check(err);

          ready_ = true;
        }

        ideas::reduce(0, n_, reduceSize_, reduceFloat_, reduceSigned_,
                      reduceSum_, (void*)reduceArgsDevPtr_, reduceRetPtr_);
      }    
      
      bool ready(){
        return ready_;
      }
      
    private:
      using KernelParams_ = vector<void*>;
      using ViewVec_ = vector<View*>;
      using ArrayVec_ = vector<Array*>;
      using VarVec_ = vector<Var*>;      

      CUmodule module_;    
      CUfunction function_;
      CUstream stream_;
      uint32_t reduceSize_;
      void* reduceRetPtr_;
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
      ArrayVec_ arrays_;
      VarVec_ vars_;
      uint32_t n_;
      uint32_t lastSize_;
      size_t gridDimX_;
      CUdeviceptr reduceArgsDevPtr_;
    };

    CUDARuntime(){

    }

    ~CUDARuntime(){

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

    bool initKernel2(uint32_t kernelId,
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
      new Kernel(true, ptx, reduceSize, reduceFloat, reduceSigned, reduceSum);
      
      kernelMap_[kernelId] = kernel;
      return false;
    }

    void addView(void** viewPtr,
                 uint32_t elementSize,
                 uint32_t staticDims,
                 uint32_t* staticSizes,
                 uint32_t runTimeDims){
   
      // offset based on Kokkos view layout which
      // may change in future versions (has changed in the past...) -dpx  
      void* data = viewPtr[1];

      auto itr = viewMap_.find(data);
      
      if(itr != viewMap_.end()){
        return;
      }

      View* view = new View(data);

      // offset based on Kokkos view layout -dpx
      viewPtr+=2;
      uint64_t* sizes = (uint64_t*)viewPtr;

      size_t size = elementSize;

      for(size_t i = 0; i < runTimeDims; ++i){
        uint64_t si = *sizes;
        //fprintf(stderr,"sizes %ld %ld\n",i,si);   
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

    void addArray(void* arrayPtr,
                  uint32_t elementSize,
                  uint32_t size){
      
      auto itr = arrayMap_.find(arrayPtr);
      
      if(itr != arrayMap_.end()){
        return;
      }

      Array* array = new Array(arrayPtr);

      size_t bytes = elementSize * size;

      CUdeviceptr devPtr;
      CUresult err = cuMemAlloc(&devPtr, bytes);
      check(err);

      array->setDevPtr(devPtr);
      array->setSize(bytes);

      arrayMap_[arrayPtr] = array;
    }

    void addKernelVar(uint32_t kernelId, void* varPtr, size_t size){
      auto kitr = kernelMap_.find(kernelId);
      assert(kitr != kernelMap_.end());

      Kernel* kernel = kitr->second;

      kernel->addVar(varPtr, size);     
    }

    void addKernelView(uint32_t kernelId, void** viewPtr, uint32_t flags){
       // offset based on Kokkos view layout -dpx
      void* data = viewPtr[1];

      auto itr = kernelMap_.find(kernelId);
      assert(itr != kernelMap_.end());

      Kernel* kernel = itr->second;
      
      auto vitr = viewMap_.find(data);
      assert(vitr != viewMap_.end());

      View* view = vitr->second;

      kernel->addView(view);
    }

    void addKernelArray(uint32_t kernelId, void* arrayPtr, uint32_t flags){
      auto itr = kernelMap_.find(kernelId);
      assert(itr != kernelMap_.end());

      Kernel* kernel = itr->second;
      
      auto aitr = arrayMap_.find(arrayPtr);
      assert(aitr != arrayMap_.end());

      Array* array = aitr->second;

      kernel->addArray(array);
    }

    void runKernel(uint32_t kernelId, uint32_t n, void* reducePtr){ 
      auto itr = kernelMap_.find(kernelId);
      assert(itr != kernelMap_.end());

      Kernel* kernel = itr->second;

      if(reducePtr){
        //kernel->runReduce(n, reducePtr);
        kernel->run(n, reducePtr);
      }
      else{
        kernel->run(n, nullptr);
      }
    }

    void awaitKernel(uint32_t kernelId){ 
      auto itr = kernelMap_.find(kernelId);
      if (itr != kernelMap_.end()) {
        Kernel* kernel = itr->second;
        kernel->await();
      } else {
        fprintf(stderr, "missing kernelId = %d\n", kernelId);
      }
    }

    void copyViewToDevice(void** viewPtr){
       // offset based on Kokkos view layout -dpx
      void* data = viewPtr[1];

      auto itr = viewMap_.find(data);
      assert(itr != viewMap_.end());

      View* view = itr->second;
      view->copyToDevice();
    }

    void copyViewFromDevice(void** viewPtr){
       // offset based on Kokkos view layout -dpx
      void* data = viewPtr[1];

      auto itr = viewMap_.find(data);
      assert(itr != viewMap_.end());

      View* view = itr->second;
      view->copyFromDevice();
    }

    void copyArrayToDevice(void* arrayPtr){
      auto itr = arrayMap_.find(arrayPtr);
      assert(itr != arrayMap_.end());

      Array* array = itr->second;
      array->copyToDevice();
    }

    void copyArrayFromDevice(void* arrayPtr){
      auto itr = arrayMap_.find(arrayPtr);
      assert(itr != arrayMap_.end());

      Array* array = itr->second;
      array->copyFromDevice();
    }

  private:
    using ViewMap_ = map<void*, View*>;
    using ArrayMap_ = map<void*, Array*>;
    using KernelMap_ = unordered_map<uint32_t, Kernel*>;

    bool initalized_ = false;

    CUdevice device_;
    CUcontext context_;
    size_t numThreads_;

    ViewMap_ viewMap_;
    ArrayMap_ arrayMap_;
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

  void __ideas_debug1(void* ptr){

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

  bool __ideas_cuda_init_kernel2(uint32_t kernelId,
                                 char* ptx,
                                 uint32_t reduceSize,
                                 bool reduceFloat,
                                 bool reduceSigned,
                                 bool reduceSum){
    _cudaRuntime.init();
    return _cudaRuntime.initKernel2(kernelId, ptx, reduceSize,
                                    reduceFloat, reduceSigned, reduceSum);
  }

  void __ideas_cuda_add_view(void** viewPtr,
                             uint32_t elementSize,
                             uint32_t staticDims,
                             uint32_t* staticSizes,
                             uint32_t runtimeDims){
    _cudaRuntime.init();
    _cudaRuntime.addView(viewPtr, elementSize,
      staticDims, staticSizes, runtimeDims);
  }

  void __ideas_cuda_add_array(void* arrayPtr,
                              uint32_t elementSize,
                              uint32_t size){
    _cudaRuntime.init();
    _cudaRuntime.addArray(arrayPtr, elementSize, size);
  }

  void __ideas_cuda_add_kernel_view(uint32_t kernelId,
                                    void** viewPtr,
                                    uint32_t flags){
    _cudaRuntime.addKernelView(kernelId, viewPtr, flags);
  }

  void __ideas_cuda_add_kernel_array(uint32_t kernelId,
                                     void* arrayPtr,
                                     uint32_t flags){
    _cudaRuntime.addKernelArray(kernelId, arrayPtr, flags);
  }

  void __ideas_cuda_add_kernel_var(uint32_t kernelId, void* varPtr){
    _cudaRuntime.addKernelVar(kernelId, varPtr, 0);
  }

  void __ideas_cuda_copy_view_to_device(void** viewPtr){
    _cudaRuntime.copyViewToDevice(viewPtr);
  }

  void __ideas_cuda_copy_view_from_device(void** viewPtr){
    _cudaRuntime.copyViewFromDevice(viewPtr);
  }

  void __ideas_cuda_copy_array_to_device(void* arrayPtr){
    _cudaRuntime.copyArrayToDevice(arrayPtr);
  }

  void __ideas_cuda_copy_array_from_device(void* arrayPtr){
    _cudaRuntime.copyArrayFromDevice(arrayPtr);
  }

  void __ideas_cuda_run_kernel(uint32_t kernelId, uint32_t n, void* reducePtr){
    return _cudaRuntime.runKernel(kernelId, n, reducePtr);
  }

  void __ideas_cuda_await_kernel(uint32_t kernelId){
    return _cudaRuntime.awaitKernel(kernelId);
  }

} // extern "C"
