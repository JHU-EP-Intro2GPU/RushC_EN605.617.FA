BLOCK_SIZE=64
NUM_BLOCKS=128

all: assignment.exe
check: check-assignment
clean: clean-assignment
memcheck: memcheck-assignment

%.exe: %.cu utilities.cuh
	nvcc $< -o $@ -std=c++11 -DBLOCK_SIZE=$(BLOCK_SIZE) -DNUM_BLOCKS=$(NUM_BLOCKS)

check-%: %.exe
	$<

clean-%:
	rm -f $*.exe $*.exp $*.lib

memcheck-%: %.exe
	cuda-memcheck $<