#include <iostream>

#include <cstdint>
#include <cassert>

#include <cuda.h>

#include "mcub/device/device_reduce.cuh"

#define check(err) if(err != CUDA_SUCCESS){ \
      const char* s; \
      cuGetErrorString(err, &s); \
      cerr << "CUDARuntime error: " << s << endl; \
      assert(false); \
    }

#define np(X)                                                           \
std::cout << __FILE__ << ":" << __LINE__ << ": " << __PRETTY_FUNCTION__ \
           << ": " << #X << " = " << (X) << std::endl

using namespace std;
using namespace cub;

#ifdef IDEAS_TEST
  extern "C" __device__ void run(int index, void* args, void* result){
    double* ptr = (double*)result;
    *ptr = 1.0;
  }
#else
  extern "C" __device__ void run(int index, void* args, void* result);
#endif

typedef void(*KernelFunc)(int, void*, void*);

namespace{

  KernelFunc getKernel(size_t kernel){
    switch(kernel){
      case 0:
        return run;
      default:
        assert(false);
    }

    return NULL;
  }

  template<class T>
  class Reduce{
  public:
    Reduce(size_t kernel, size_t size, void* args)
    : size_(size),
    tempStorage_(NULL),
    tempStorageBytes_(0),
    args_(args){
      kernelNum_ = kernel;

      kernelFunc_ = getKernel(kernel);

      CUresult err = cuMemAlloc(&result_, sizeof(T));
      check(err);

      T* in = NULL;
      T* temp = NULL;
      T* result = NULL;

      DeviceReduce::Sum(temp, tempStorageBytes_,
                        in, result, size_, kernelNum_, args_);
      
      err = cuMemAlloc(&temp_, sizeof(T) * tempStorageBytes_);
      check(err);
    }

    void run(CUdeviceptr ptr, void* resultPtr){
      T* in = (T*)ptr;
      T* temp = (T*)temp_;
      T* result = (T*)result_;

      DeviceReduce::Sum(temp, tempStorageBytes_,
                        in, result, size_, kernelNum_, args_);

      CUresult err = cuMemcpyDtoH(resultPtr, result_, sizeof(T));
      check(err);
    }
    
    CUdeviceptr result_;
    CUdeviceptr temp_;

    void* tempStorage_;
    size_t tempStorageBytes_;
    size_t size_;
    KernelFunc kernelFunc_;
    void* args_;
    int kernelNum_;
  };

  template<class T>
  void sum_(size_t kernel,
            CUdeviceptr ptr,
            size_t size,
            void* args,
            void* resultPtr){
    Reduce<T>* r = new Reduce<T>(kernel, size, args);
    r->run(ptr, resultPtr);
  }

  template<class T>
  void product_(size_t kernel,
                CUdeviceptr ptr,
                size_t size,
                void* args,
                void* resultPtr){
    assert(false);
  }

} // namespace

namespace ideas{

void reduce(size_t kernel,
            CUdeviceptr ptr,
            size_t size,
            size_t scalarBytes,
            bool isFloat,
            bool isSigned,
            bool isSum,
            void* args,
            void* resultPtr){
  switch(scalarBytes){
    case 8:
      if(isFloat){
        if(isSum){
          sum_<double>(kernel, ptr, size, args, resultPtr);
        }
        else{
          product_<double>(kernel, ptr, size, args, resultPtr);
        }
      }
      else{
        if(isSigned){
          if(isSum){
            sum_<int64_t>(kernel, ptr, size, args, resultPtr);
          }
          else{
            product_<int64_t>(kernel, ptr, size, args, resultPtr);
          }
        }
        else{
          if(isSum){
            sum_<uint64_t>(kernel, ptr, size, args, resultPtr);
          }
          else{
            product_<uint64_t>(kernel, ptr, size, args, resultPtr);
          }
        }
      }
      break;
    case 4:
      if(isFloat){
        if(isSum){
          sum_<float>(kernel, ptr, size, args, resultPtr);
        }
        else{
          product_<float>(kernel, ptr, size, args, resultPtr);
        }
      }
      else{
        if(isSigned){
          if(isSum){
            sum_<int32_t>(kernel, ptr, size, args, resultPtr);
          }
          else{
            product_<int32_t>(kernel, ptr, size, args, resultPtr);
          }
        }
        else{
          if(isSum){
            sum_<uint32_t>(kernel, ptr, size, args, resultPtr);
          }
          else{
            product_<uint32_t>(kernel, ptr, size, args, resultPtr);
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
            sum_<int16_t>(kernel, ptr, size, args, resultPtr);
          }
          else{
            product_<int16_t>(kernel, ptr, size, args, resultPtr);
          }
        }
        else{
          if(isSum){
            sum_<uint16_t>(kernel, ptr, size, args, resultPtr);
          }
          else{
            product_<uint16_t>(kernel, ptr, size, args, resultPtr);
          }
        }
      }
      break;
    default:
      assert(false);
  }
}

} // namespace ideas
