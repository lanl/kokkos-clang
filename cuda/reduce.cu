#include <iostream>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>

//#include <cstdint>

//#include "cuda.h"

#include "mcub/cub/device/device_reduce.cuh"

#define check(err) if(err != CUDA_SUCCESS){ \
      const char* s; \
      cuGetErrorString(err, &s); \
      cerr << "CUDARuntime error: " << s << endl; \
      assert(false); \
    }

using namespace std;
using namespace cub;

namespace{

  template<class T>
  class Data{
  public:
    Data(size_t size)
    : size(size){
      d_temp_storage = NULL;
      temp_storage_bytes = 0;

      CUresult err = cuMemAlloc(&out, sizeof(T));
      check(err);

      d_out = (T*)out;

      DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, size);
      
      err = cuMemAlloc(&temp, sizeof(T) * temp_storage_bytes);
      check(err);

      d_temp = (T*)temp;

    }

    void run(CUdeviceptr ptr, size_t size){
      d_in = (T*)ptr;
      DeviceReduce::Sum(d_temp, temp_storage_bytes, d_in, d_out, size);
    }

    void copyOut(void* resultPtr){
      CUresult err = cuMemcpyDtoH(resultPtr, (CUdeviceptr)d_out, sizeof(T));
      check(err);
    }

    T* d_in;
    T* d_out;
    T* d_temp;
    
    CUdeviceptr out;
    CUdeviceptr temp;

    void            *d_temp_storage;
    size_t          temp_storage_bytes;
    size_t          size;

  };

  Data<double>* _data = NULL;

  template<class T>
  void sum_(CUdeviceptr ptr, size_t size, void* resultPtr){
    if(!_data){
      _data = new Data<double>(size);
    }
    _data->run(ptr, size);
    _data->copyOut(resultPtr);
  }

  template<class T>
  void product_(CUdeviceptr ptr, size_t size, void* resultPtr){
    /*
    thrust::device_ptr<T> begin((T*)ptr);
    thrust::device_ptr<T> end((T*)ptr + size);

    T result = thrust::reduce(begin, end, T(1), thrust::multiplies<T>());
    memcpy(resultPtr, &result, sizeof(T));
    */
  }

} // namespace

namespace ideas{

void reduce(CUdeviceptr ptr,
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
          sum_<double>(ptr, size, resultPtr);
        }
        else{
          product_<double>(ptr, size, resultPtr);
        }
      }
      else{
        if(isSigned){
          if(isSum){
            sum_<int64_t>(ptr, size, resultPtr);
          }
          else{
            product_<int64_t>(ptr, size, resultPtr);
          }
        }
        else{
          if(isSum){
            sum_<uint64_t>(ptr, size, resultPtr);
          }
          else{
            product_<uint64_t>(ptr, size, resultPtr);
          }
        }
      }
      break;
    case 4:
      if(isFloat){
        if(isSum){
          sum_<float>(ptr, size, resultPtr);
        }
        else{
          product_<float>(ptr, size, resultPtr);
        }
      }
      else{
        if(isSigned){
          if(isSum){
            sum_<int32_t>(ptr, size, resultPtr);
          }
          else{
            product_<int32_t>(ptr, size, resultPtr);
          }
        }
        else{
          if(isSum){
            sum_<uint32_t>(ptr, size, resultPtr);
          }
          else{
            product_<uint32_t>(ptr, size, resultPtr);
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
            sum_<int16_t>(ptr, size, resultPtr);
          }
          else{
            product_<int16_t>(ptr, size, resultPtr);
          }
        }
        else{
          if(isSum){
            sum_<uint16_t>(ptr, size, resultPtr);
          }
          else{
            product_<uint16_t>(ptr, size, resultPtr);
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
            sum_<int8_t>(ptr, size, resultPtr);
          }
          else{
            product_<int8_t>(ptr, size, resultPtr);
          }
        }
        else{
          if(isSum){
            sum_<uint8_t>(ptr, size, resultPtr);
          }
          else{
            product_<uint8_t>(ptr, size, resultPtr);
          }
        }
      }
      break;
    default:
      assert(false);
  }
}

} // namespace ideas
