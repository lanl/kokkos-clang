all: ../reduce.cu test.cu
	nvcc -dc -arch sm_35 -std=c++11 -c -DIDEAS_TEST=1 ../reduce.cu -o reduce-device.o
	nvcc -arch sm_35 -std=c++11 -c -DIDEAS_TEST=1 ../reduce.cu -o reduce.o
	nvcc -arch sm_35 -std=c++11 -c test.cu -o test.o
	nvcc -dc -arch sm_35 -std=c++11 ptx.ptx -o ptx.o
	nvcc -dlink -arch sm_35 -std=c++11 reduce-device.o ptx.o -o link.o

	#g++ -std=c++11 -L/projects/opt/centos7/cuda/8.0/lib64 -lcuda -lcudart reduce.o test.o -o test
	nvcc -arch sm_35 -std=c++11 -lcuda test.o link.o -o test
