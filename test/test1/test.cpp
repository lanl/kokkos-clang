#include <Kokkos_Core.hpp>
#include <cstdio>
#include <iostream>

using namespace std;

typedef Kokkos::View<double*[3]> view_type;

const size_t SIZE = 1024;

int main (int argc, char* argv[]) {
  Kokkos::initialize (argc, argv);

  view_type a ("A", SIZE);

  Kokkos::parallel_for (SIZE, KOKKOS_LAMBDA (const int i) {
    a(i,0) = 1.0*i;
    a(i,1) = 1.0*i*i;
    a(i,2) = 1.0*i*i*i;
  });

  double sum = 0;
  for(size_t i = 0; i < SIZE; ++i){
    sum += a(i,0);
    sum += a(i,1);
    sum += a(i,2);
  }

  cout << "sum = " << sum << endl;

  Kokkos::finalize ();
}
