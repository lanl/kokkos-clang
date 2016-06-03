#include <Kokkos_Core.hpp>
#include <cstdio>
#include <iostream>

using namespace std;

typedef Kokkos::View<double*[3]> view_type;

const size_t SIZE = 1048576;

int main (int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);

  double sum = 1.0;
  Kokkos::parallel_reduce(SIZE, KOKKOS_LAMBDA (const int i, double& lsum){
    lsum += 1.0;
  }, sum);

  cout << "sum = " << sum << endl;

  Kokkos::finalize ();
}
