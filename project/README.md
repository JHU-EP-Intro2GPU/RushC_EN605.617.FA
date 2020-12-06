# Intent

This project aims to implement a GPU-accelerated implementation of the
[Advanced Encryption Standard (AES)](https://en.wikipedia.org/wiki/Advanced_Encryption_Standard) cipher in [counter (CTR)](https://en.wikipedia.org/wiki/Block_cipher_mode_of_operation#Counter_(CTR)) mode. The primary focus of
this project is on optimizing CTR mode specifically. Optimizing the AES cipher
itself is a stretch goal.

<br>

# Background

AES is the standard encryption algorithm established by the U.S. National 
Institute of Standards and Technology (NIST). On its own, AES is designed to
either encrypt or decrypt a single block (16 bytes) of data. In order to apply
AES to larger amounts of data, there exist various modes of operations, such as
CTR. 

The reason for focusing on CTR mode specifically is because it supports better
parallelization than other common modes such as [cipher block chaining (CBC)](https://en.wikipedia.org/wiki/Block_cipher_mode_of_operation#Cipher_block_chaining_(CBC)).
CBC mode uses the previous output of one block as the input for the next, 
meaning that every block of data must be encrypted/decrypted sequentially. By
comparison, CTR uses a simple counter that is incremented with each block. Each
block can therefore be encrypted in parallel, since the counter value can be
calculated independently for each block.

<br>

# Implementation

This project consists of 3 separate implementation of AES-CTR:

* **pyaes** - a Python module
* **libaes** - a C++ library
* **cuAES** - a CUDA library

**pyaes** was written first for reference and to get practice with implementing
AES in a language that was a little easier and more familiar. It was also 
helpful in debugging issues with the other implementations.

**libaes** was derived from the Python implementation and is used as the 
baseline CPU implementation that the GPU implementation will be compared 
against. To make sure the comparison is fair, and that both the CPU and GPU are
utilized as much as possible, this implementation parallelizes each of the 
blocks in CTR mode by utilizing multiple threads.

**cuAES** is the final GPU implementation. It started off as a copy of libaes
which was modified so that the AES cipher could be called from GPU code. CTR
mode is optimized by splitting each of the blocks among various GPU threads.

Each implementation's AES cipher is tested against various NIST KATs (Known 
Answer Tests). There are no full tests for CTR mode, since NIST does not provide
specific CTR KATs, however some test vectors may be added if time permits.

Additionally, there is a **utils** library that contains various C++/CUDA 
helpers that are used by each project.

Finally, there is a **benchmark** application which compares the performance of
libaes and cuAES for encrypting various sizes of plaintext.

<br>

# Dependencies

The following dependencies are required to build and run the projects in this
repo:

| Dependency    | Version | Notes                                                              |
|:--------------|:-------:|:-------------------------------------------------------------------|
| Python        | >= 3.6  | Needed to run pyaes or its unit tests                              |
| C++ Compiler  | -       | Must support C++14 or greater; tested with MSVC and Clang          |
| CUDA          | >= 3.0? | Tested on Windows with version 11.0                                |
| CMake         | >= 3.10 | Used to configure and build the projects; tested with version 3.19 |

<br>

# Build Instructions

After cloning the repo, navigate to the repo's directory and configure the 
project using CMake:

```sh
cmake3 -S . -B build
```

Then you can build any of the targets using whichever generator is used by 
CMake. For example, on Linux, CMake will generate Makefiles by default, so you 
can build each target by switching into the `build` directory and invoking 
`make <target>`.

Build all of the libraries and executables using the `all` target. Run all of 
the unit tests using the `test` target. Run the benchmark application using the
`run-benchmark` target.

<br>

# Benchmark Results

These are the results of running the benchmark on a Windows 10 machine with an
Intel Core i7-7700K CPU, an NVIDEA GeForce GTX 1060 GPU, and 32 GB of RAM:

```log
Plaintext Size:           16 B          512 B           1 kB          64 kB           1 MB          32 MB           1 GB
------------------------------------------------------------------------------------------------------------------------
  libaes (CPU):            0ms            0ms            0ms           31ms          335ms         9787ms       309141ms
   cuaes (GPU):            3ms            3ms            3ms            6ms           57ms         1727ms        55319ms
```
