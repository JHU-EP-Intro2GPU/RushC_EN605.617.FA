# Vocareum doesn't seem to have OpenCL installed, so the paths to the OpenCL
# headers and library need to be specified manually. On my Windows machine it is
# already installed, so nothing needs to be done manually.
ifeq ($(shell uname), Linux)
CUDA_PATH ?= /usr/local/cuda
INCLUDES = -I$(CUDA_PATH)/include
LIB_PATHS = -L$(CUDA_PATH)/lib64
endif

all: assignment.exe
check: check-assignment
clean: clean-assignment

%.exe: %.cpp
	g++ $(INCLUDES) $< $(LIB_PATHS) -lOpenCL -o $@ --std=c++11

check-%: %.exe
	./$<

clean-%:
	rm -rf $*.exe
