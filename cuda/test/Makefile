all: ../reduce.cu test.cu
	nvcc -std=c++11 -c -DIDEAS_TEST=1 ../reduce.cu -o reduce.o
	nvcc -std=c++11 -c test.cu -o test.o
	nvcc -std=c++11 -lcuda test.o reduce.o -o test
