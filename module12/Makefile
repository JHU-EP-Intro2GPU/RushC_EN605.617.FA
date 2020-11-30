all: assignment.exe
check: check-assignment
clean: clean-assignment

%.exe: %.cpp
	nvcc $^ -o $@ -lOpenCL --std=c++11

check-%: ARGS=""
check-%: %.exe
	./$< $(ARGS)

clean-%:
	rm -rf $*.exe
