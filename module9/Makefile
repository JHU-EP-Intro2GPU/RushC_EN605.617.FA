all: assignment.exe
check: check-assignment
clean: clean-assignment
memcheck: memcheck-assignment

check-thrust: thrust.exe
	./$<
	./$< 1024
	./$< 1048576
	./$< 5335040

%.exe: FLAGS="--std=c++11"
%.exe: %.cu
	nvcc $< -o $@ $(FLAGS)

check-%: %.exe ARGS=""
	./$< $(ARGS)

clean-%:
	rm -f $*.exe $*.exp $*.lib

memcheck-%: %.exe
	cuda-memcheck $<
