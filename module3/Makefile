all: assignment.exe

%.exe: %.cu
	nvcc $^ -o $@ -std=c++11

check: assignment.exe
	./$< 512 256
	./$< 1024 256
	./$< 1024 32
	./$< 65536 256
	./$<

.PHONY: check