#include <Kokkos_Core.hpp>
#include <cstdio>
#include <iostream>

#include <sys/time.h>

using namespace std;

typedef Kokkos::View<double*[3]> view_type;

const size_t SIZE = 1048576;

double now(){
  timeval tv;
  gettimeofday(&tv, 0);
  
  return tv.tv_sec + tv.tv_usec/1e6;
}

int main (int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);

  double t1;

  for(size_t i = 0; i < 10000; ++i){
    if(i == 1){
      t1 = now();
    }

    double sum = 1.0;
    Kokkos::parallel_reduce(SIZE, KOKKOS_LAMBDA (const int i, double& lsum){
      lsum += 1.00001;
    }, sum);
  }

  double dt = now() - t1;

  cout << "dt = " << dt << endl;

  //cout << "sum = " << sum << endl;

  Kokkos::finalize ();
}
