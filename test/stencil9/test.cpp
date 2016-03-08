#include <Kokkos_Core.hpp>
#include <cstdio>
#include <iostream>

using namespace std;

typedef Kokkos::View<double*[2]> view_type;

const size_t SIZE = 256;

int main (int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);

  view_type a ("A", SIZE*SIZE);

  Kokkos::parallel_for(SIZE, KOKKOS_LAMBDA(const int i){
    int x = i % SIZE;
    int y = i / SIZE;

    int n = 0;
    float s = 0;

    if(x > 0){
      if(y > 0){
        // nw
        int j = (y - 1) * SIZE + x - 1;
        s += a(j, 0);
        ++n;
      }
      
      // w
      int j = y * SIZE + x - 1;
      s += a(j, 0);
      ++n;

      // sw
      if(y < SIZE - 1){
       int j = (y + 1) * SIZE + x - 1;
       s += a(j, 0);
       ++n;       
      }
    }

    if(x < SIZE - 1){
      if(y > 0){
        // ne
        int j = (y - 1) * SIZE + x + 1;
        s += a(j, 0);
        ++n;  
      }

      // e
      int j = y * SIZE + x + 1;
      s += a(j, 0);
      ++n;

      // se
      if(y < SIZE - 1){
       int j = (y + 1) * SIZE + x + 1;
       s += a(j, 0);
       ++n;       
      }
    }

    if(y < SIZE - 1){
      // s
      int j = (y + 1) * SIZE + x;
      s += a(j, 0);
      ++n; 
    }

    if(y > 0){
      // n
      int j = (y - 1) * SIZE + x;
      s += a(j, 0);
      ++n;
    }

    a(i, 1) = s/n;
  });

  Kokkos::parallel_for(SIZE, KOKKOS_LAMBDA(const int i){
    a(i, 0) += a(i, 1);
  });

  Kokkos::finalize();
}
