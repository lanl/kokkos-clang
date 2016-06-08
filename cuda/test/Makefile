all: ../reduce.cu test.cu
	nvcc -std=c++11 -c -DIDEAS_TEST=1 ../reduce.cu -o reduce.o
	nvcc -std=c++11 -c test.cu -o test.o
	#g++ -std=c++11 -L/projects/opt/centos7/cuda/8.0/lib64 -lcuda -lcudart reduce.o test.o -o test
	nvcc -std=c++11 -lcuda reduce.o test.o -o test
