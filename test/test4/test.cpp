#include <Kokkos_Core.hpp>
#include <cstdio>
#include <iostream>
#include <map>

using namespace std;

typedef Kokkos::View<double*[3]> view_type;

const size_t SIZE = 1024;

int main (int argc, char* argv[]) {
  Kokkos::initialize (argc, argv);

  view_type a ("A", SIZE);
  view_type b ("A", SIZE);

  Kokkos::parallel_for (SIZE, KOKKOS_LAMBDA (const int i) {
    a(i,0) = 0.0;
    a(i,1) = 0.0;
    a(i,2) = 0.0;
  });

  Kokkos::parallel_for (SIZE, KOKKOS_LAMBDA (const int i) {
    b(i,0) = 0.0;
    b(i,1) = 0.0;
    b(i,2) = 0.0;
  });

  for(size_t i = 0; i < 10000; ++i){
    Kokkos::parallel_for (SIZE, KOKKOS_LAMBDA (const int i) {
      a(i,0) += 1.0*i;
      a(i,1) += 1.0*i*i;
      a(i,2) += 1.0*i*i*i;
    });

    Kokkos::parallel_for (SIZE, KOKKOS_LAMBDA (const int i) {
      a(i,0) += 3.0*i;
      a(i,1) += 3.0*i*i;
      a(i,2) += 3.0*i*i*i;
    });

    Kokkos::parallel_for (SIZE, KOKKOS_LAMBDA (const int i) {
      a(i,0) += b(i,0);
      a(i,1) += b(i,1);
      a(i,2) += b(i,2);
    });
  }

  /*
  double sum = 0.0;

  Kokkos::parallel_for (SIZE, KOKKOS_LAMBDA (const int i) {
    sum += a(i,0);
    sum += a(i,1);
    sum += a(i,2);
  });

  cout << "sum = " << sum << endl;
*/

  Kokkos::finalize ();
}
