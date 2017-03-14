#include <iostream>

using namespace std;

extern "C" __device__ __host__ void myfunc(int, void*, int* xi);

int main(int argc, char const *argv[]){
  void* bodyFunc = (void*)myfunc;

  return 0;
}

