all: assignment.exe

check: assignment.exe
	./$< 512 256 5
	./$< 1024 256 27
	./$< 1024 32 92
	./$< 65536 256 2
	./$<

%.exe: %.cu
	nvcc $^ -o $@ -std=c++11

check-%: %.exe
	$<
