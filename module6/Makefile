all: assignment.exe
check: check-assignment
clean: clean-assignment
memcheck: memcheck-assignment

check-all:
	make clean
	make check NUM_BLOCKS=2 BLOCK_SIZE=256
	make clean
	make check NUM_BLOCKS=4 BLOCK_SIZE=256
	make clean
	make check NUM_BLOCKS=32 BLOCK_SIZE=32
	make clean
	make check NUM_BLOCKS=256 BLOCK_SIZE=256

%.exe: %.cu utilities.cuh
	nvcc $< -o $@ --std=c++11

check-%: %.exe
	./$<

clean-%:
	rm -f $*.exe $*.exp $*.lib

memcheck-%: %.exe
	cuda-memcheck $<

	check: assignment.exe
	./$< 512 256 5
	./$< 1024 256 27
	./$< 1024 32 92
	./$< 65536 256 2
	./$<