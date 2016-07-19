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

//  extern "C" __device__ void run(int index, void* args, void* result);
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

    void run(void* resultPtr){
      T* in = NULL;
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
            size_t size,
            void* args,
            void* resultPtr){
    Reduce<T>* r = new Reduce<T>(kernel, size, args);
    r->run(resultPtr);
  }

  template<class T>
  void product_(size_t kernel,
                size_t size,
                void* args,
                void* resultPtr){
    assert(false);
  }

} // namespace

namespace ideas{

void reduce(size_t kernel,
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
          sum_<double>(kernel, size, args, resultPtr);
        }
        else{
          product_<double>(kernel, size, args, resultPtr);
        }
      }
      else{
        if(isSigned){
          if(isSum){
            sum_<int64_t>(kernel, size, args, resultPtr);
          }
          else{
            product_<int64_t>(kernel, size, args, resultPtr);
          }
        }
        else{
          if(isSum){
            sum_<uint64_t>(kernel, size, args, resultPtr);
          }
          else{
            product_<uint64_t>(kernel, size, args, resultPtr);
          }
        }
      }
      break;
    case 4:
      if(isFloat){
        if(isSum){
          sum_<float>(kernel, size, args, resultPtr);
        }
        else{
          product_<float>(kernel, size, args, resultPtr);
        }
      }
      else{
        if(isSigned){
          if(isSum){
            sum_<int32_t>(kernel, size, args, resultPtr);
          }
          else{
            product_<int32_t>(kernel, size, args, resultPtr);
          }
        }
        else{
          if(isSum){
            sum_<uint32_t>(kernel, size, args, resultPtr);
          }
          else{
            product_<uint32_t>(kernel, size, args, resultPtr);
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
            sum_<int16_t>(kernel, size, args, resultPtr);
          }
          else{
            product_<int16_t>(kernel, size, args, resultPtr);
          }
        }
        else{
          if(isSum){
            sum_<uint16_t>(kernel, size, args, resultPtr);
          }
          else{
            product_<uint16_t>(kernel, size, args, resultPtr);
          }
        }
      }
      break;
    default:
      assert(false);
  }
}

} // namespace ideas
