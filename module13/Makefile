all: assignment.exe
clean: clean-assignment

check:
	@make check-assignment ARGS="square"
	@make check-assignment ARGS="square cube"
	@make check-assignment ARGS="square cube square cube cube"
	@make check-assignment ARGS="negate_arithmetic negate_bitwise"
	@make check-assignment ARGS="square cube negate_arithmetic negate_bitwise"

%.exe: %.cpp
	nvcc $^ -o $@ -lOpenCL --std=c++11

check-%: ARGS=""
check-%: %.exe
	./$< $(ARGS)

clean-%:
	rm -rf $*.exe
