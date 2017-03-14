bad:
	nvcc -arch compute_35 -ptx func.cu
	nvcc -arch compute_35 -dc func.ptx -o func.o
	nvcc -arch compute_35 -dc main.cu -o main.o
	nvcc -arch compute_35 -lcuda main.o func.o -o test

good:
	nvcc -arch compute_35 -dc func.cu -o func.o
	nvcc -arch compute_35 -dc main.cu -o main.o
	nvcc -arch compute_35 -lcuda main.o func.o -o test

try:
	nvcc -arch compute_35 -ptx --keep-device-functions func.cu
	nvcc -arch compute_35 -dc --keep-device-functions func.ptx -o func.o
	nvcc -arch compute_35 -dc main.cu -o main.o
	nvcc -arch compute_35 -lcuda main.o func.o -o test
clean:
	rm -f *.o
	rm -f *.ptx
