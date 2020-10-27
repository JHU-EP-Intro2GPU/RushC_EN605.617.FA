all: assignment.exe
check: check-assignment
clean: clean-assignment
memcheck: memcheck-assignment

check-assignment: assignment.exe
	./$< 4 4
	./$< 512 1024
	./$< 1024 512
	./$< 2048 2048

%.exe: %.cu utilities.cuh
	nvcc $< -o $@ --std=c++11 -lcublas

check-%: %.exe
	./$<

clean-%:
	rm -f $*.exe $*.exp $*.lib

memcheck-%: %.exe
	cuda-memcheck $<
