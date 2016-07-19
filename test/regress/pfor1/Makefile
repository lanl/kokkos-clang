KOKKOS_PATH = ../../../kokkos
SRC = $(wildcard *.cpp)

default: build
	echo "Start Build"

CXX = ../../../build/llvm/bin/clang++ -std=c++14 -isystem /projects/opt/centos7/gcc/5.2.0/include/c++/5.2.0 -I/projects/opt/centos7/gcc/5.2.0/include/c++/5.2.0/x86_64-unknown-linux-gnu
LINK = ${CXX}
LINKFLAGS = -L/home/nickm/ideas_cuda/build -L/home/nickm/kokkos-clang/build/runtime -L/usr/local/cuda/lib64 -L/projects/opt/centos7/gcc/5.2.0/packages/gcc-5.2.0/lib64 
EXE = $(SRC:.cpp=.host)
KOKKOS_DEVICES = "Pthread"
KOKKOS_ARCH = "SNB"
DEPFLAGS = -M

OBJ = $(SRC:.cpp=.o)
LIB = -lideas_runtime -lideas_runtime_cuda -lcuda -lcudart

include $(KOKKOS_PATH)/Makefile.kokkos

build: $(EXE)

$(EXE): $(OBJ) $(KOKKOS_LINK_DEPENDS)
	$(LINK) $(KOKKOS_LDFLAGS) $(LINKFLAGS) $(EXTRA_PATH) $(OBJ) $(KOKKOS_LIBS) $(LIB) -o $(EXE)

clean: kokkos-clean 
	rm -f *.o *.cuda *.host

# Compilation rules

%.o:%.cpp $(KOKKOS_CPP_DEPENDS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -c $<
