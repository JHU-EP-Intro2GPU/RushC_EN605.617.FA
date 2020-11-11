UTILNPP_INCLUDES = common/UtilNPP
FREEIMAGE_INCLUDES = common/FreeImage/include

ifeq ($(shell uname), Linux)
FREEIMAGE_LIB = common/FreeImage/lib/linux/x86_64/libfreeimage.a
else
FREEIMAGE_LIB = common/FreeImage/lib/darwin/libfreeimage.a
endif

all: assignment.exe
check: check-assignment
clean: clean-assignment
memcheck: memcheck-assignment

check-thrust: thrust.exe
	./$<
	./$< 1024
	./$< 1048576
	./$< 5335040

%.exe: %.cu
	nvcc -I$(UTILNPP_INCLUDES) -I$(FREEIMAGE_INCLUDES) $< -o $@ $(FREEIMAGE_LIB) --std=c++11

check-%: %.exe ARGS=""
	./$< $(ARGS)

clean-%:
	rm -f $*.exe $*.exp $*.lib

memcheck-%: %.exe
	cuda-memcheck $<
