all: assignment.exe
check: check-assignment
clean: clean-assignment
memcheck: memcheck-assignment

check-assignment: assignment.exe
	./$< 512 256
	./$< 1024 256
	./$< 1024 32
	./$< 65536 256
	./$<

%.exe: %.cu utilities.cuh
	nvcc $< -o $@ --std=c++11

check-%: %.exe
	./$<

clean-%:
	rm -f $*.exe $*.exp $*.lib

memcheck-%: %.exe
	cuda-memcheck $<
