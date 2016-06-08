#include <cuda.h>

#include <iostream>
#include <cassert>

#define check(err) if(err != CUDA_SUCCESS){ \
      const char* s; \
      cuGetErrorString(err, &s); \
      cerr << "CUDARuntime error: " << s << endl; \
      assert(false); \
    }

using namespace std;

namespace ideas{

  extern void reduce(size_t kernel,
              CUdeviceptr ptr,
              size_t size,
              size_t scalarBytes,
              bool isFloat,
              bool isSigned,
              bool isSum,
              void* args,
              void* resultPtr);
  
} // namespace ideas

int main(int argc, char** argv){  
  CUresult err = cuInit(0);
  check(err);
  
  CUdevice device;
  CUcontext context;

  err = cuDeviceGet(&device, 0);
  check(err);

  err = cuCtxCreate(&context, 0, device);
  check(err);

  CUdeviceptr ptr;

  double r = 0;

  ideas::reduce(0, ptr, 1024, 8, true, true, true, NULL, &r);

  cout << "r = " << r << endl;

  return 0;
}
