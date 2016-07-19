#include <Kokkos_Core.hpp>
#include <cstdio>
#include <iostream>

using namespace std;

const size_t SIZE = 1024;

int main (int argc, char* argv[]) {
  Kokkos::initialize (argc, argv);

  typedef Kokkos::View<double*[2]> view_type;

  view_type a ("A", SIZE);

  for(size_t i = 0; i < SIZE; ++i){
    a(i, 0) = 0.0;
    a(i, 1) = 0.0;
  }

  Kokkos::parallel_for (SIZE, KOKKOS_LAMBDA (const int i) {
    a(i, 0) += i * 100.0;
    a(i, 1) += i * 10000.0;
  });

  for(size_t i = 0; i < SIZE; ++i){
    double ai0 = a(i, 0);
    double ai1 = a(i, 1);

    cout << "a(" << i << ", 0) = " << ai0 << endl;
    cout << "a(" << i << ", 1) = " << ai1 << endl;
  }

  Kokkos::finalize ();
}
