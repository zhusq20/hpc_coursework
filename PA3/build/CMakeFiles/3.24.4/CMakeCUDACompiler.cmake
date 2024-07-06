set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.1/bin/nvcc")
set(CMAKE_CUDA_HOST_COMPILER "")
set(CMAKE_CUDA_HOST_LINK_LAUNCHER "/home/spack/spack/opt/spack/linux-debian12-broadwell/gcc-12.2.0/gcc-10.2.0-rngqwhteky3csksvpgujyc2dpdyqyqej/bin/g++")
set(CMAKE_CUDA_COMPILER_ID "NVIDIA")
set(CMAKE_CUDA_COMPILER_VERSION "11.1.74")
set(CMAKE_CUDA_DEVICE_LINKER "/usr/local/cuda-11.1/bin/nvlink")
set(CMAKE_CUDA_FATBINARY "/usr/local/cuda-11.1/bin/fatbinary")
set(CMAKE_CUDA_STANDARD_COMPUTED_DEFAULT "14")
set(CMAKE_CUDA_EXTENSIONS_COMPUTED_DEFAULT "ON")
set(CMAKE_CUDA_COMPILE_FEATURES "cuda_std_03;cuda_std_11;cuda_std_14;cuda_std_17")
set(CMAKE_CUDA03_COMPILE_FEATURES "cuda_std_03")
set(CMAKE_CUDA11_COMPILE_FEATURES "cuda_std_11")
set(CMAKE_CUDA14_COMPILE_FEATURES "cuda_std_14")
set(CMAKE_CUDA17_COMPILE_FEATURES "cuda_std_17")
set(CMAKE_CUDA20_COMPILE_FEATURES "")
set(CMAKE_CUDA23_COMPILE_FEATURES "")

set(CMAKE_CUDA_PLATFORM_ID "Linux")
set(CMAKE_CUDA_SIMULATE_ID "GNU")
set(CMAKE_CUDA_COMPILER_FRONTEND_VARIANT "")
set(CMAKE_CUDA_SIMULATE_VERSION "10.2")



set(CMAKE_CUDA_COMPILER_ENV_VAR "CUDACXX")
set(CMAKE_CUDA_HOST_COMPILER_ENV_VAR "CUDAHOSTCXX")

set(CMAKE_CUDA_COMPILER_LOADED 1)
set(CMAKE_CUDA_COMPILER_ID_RUN 1)
set(CMAKE_CUDA_SOURCE_FILE_EXTENSIONS cu)
set(CMAKE_CUDA_LINKER_PREFERENCE 15)
set(CMAKE_CUDA_LINKER_PREFERENCE_PROPAGATES 1)

set(CMAKE_CUDA_SIZEOF_DATA_PTR "8")
set(CMAKE_CUDA_COMPILER_ABI "ELF")
set(CMAKE_CUDA_BYTE_ORDER "LITTLE_ENDIAN")
set(CMAKE_CUDA_LIBRARY_ARCHITECTURE "x86_64-linux-gnu")

if(CMAKE_CUDA_SIZEOF_DATA_PTR)
  set(CMAKE_SIZEOF_VOID_P "${CMAKE_CUDA_SIZEOF_DATA_PTR}")
endif()

if(CMAKE_CUDA_COMPILER_ABI)
  set(CMAKE_INTERNAL_PLATFORM_ABI "${CMAKE_CUDA_COMPILER_ABI}")
endif()

if(CMAKE_CUDA_LIBRARY_ARCHITECTURE)
  set(CMAKE_LIBRARY_ARCHITECTURE "x86_64-linux-gnu")
endif()

set(CMAKE_CUDA_COMPILER_TOOLKIT_ROOT "/usr/local/cuda-11.1")
set(CMAKE_CUDA_COMPILER_TOOLKIT_LIBRARY_ROOT "/usr/local/cuda-11.1")
set(CMAKE_CUDA_COMPILER_TOOLKIT_VERSION "11.1.74")
set(CMAKE_CUDA_COMPILER_LIBRARY_ROOT "/usr/local/cuda-11.1")

set(CMAKE_CUDA_ARCHITECTURES_ALL "35-real;37-real;50-real;52-real;53-real;60-real;61-real;62-real;70-real;72-real;75-real;80-real;86")
set(CMAKE_CUDA_ARCHITECTURES_ALL_MAJOR "35-real;50-real;60-real;70-real;80")
set(CMAKE_CUDA_ARCHITECTURES_NATIVE "")

set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES "/usr/local/cuda-11.1/targets/x86_64-linux/include")

set(CMAKE_CUDA_HOST_IMPLICIT_LINK_LIBRARIES "")
set(CMAKE_CUDA_HOST_IMPLICIT_LINK_DIRECTORIES "/usr/local/cuda-11.1/targets/x86_64-linux/lib/stubs;/usr/local/cuda-11.1/targets/x86_64-linux/lib")
set(CMAKE_CUDA_HOST_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")

set(CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES "/home/spack/spack/opt/spack/linux-debian12-broadwell/gcc-12.2.0/ncurses-6.4-7lr4y2bfbsx4gbztlu7rkf27r7qbg7tu/include;/usr/local/cuda-11.1/include;/home/spack/spack/opt/spack/linux-debian12-broadwell/gcc-12.2.0/gcc-10.2.0-rngqwhteky3csksvpgujyc2dpdyqyqej/include;/home/spack/spack/opt/spack/linux-debian12-broadwell/gcc-12.2.0/zstd-1.5.2-b7naonmcsrankrpr3l2djbhe7h3fmka5/include;/home/spack/spack/opt/spack/linux-debian12-broadwell/gcc-12.2.0/zlib-1.2.13-35xuqkethhgeiwg5icruve6x4zmlybph/include;/home/spack/spack/opt/spack/linux-debian12-broadwell/gcc-12.2.0/mpc-1.2.1-oc4526ngoxkwu7baaoz4xi2wj5tss5fl/include;/home/spack/spack/opt/spack/linux-debian12-broadwell/gcc-12.2.0/mpfr-4.1.0-tc7th5la42lcp6ontcxivowhx2kihdxi/include;/home/spack/spack/opt/spack/linux-debian12-broadwell/gcc-12.2.0/gmp-6.2.1-mkdkv7btdd2oxay6ocaqc4vbnqm5ph2u/include;/home/spack/spack/opt/spack/linux-debian12-broadwell/gcc-12.2.0/gcc-10.2.0-rngqwhteky3csksvpgujyc2dpdyqyqej/include/c++/10.2.0;/home/spack/spack/opt/spack/linux-debian12-broadwell/gcc-12.2.0/gcc-10.2.0-rngqwhteky3csksvpgujyc2dpdyqyqej/include/c++/10.2.0/x86_64-pc-linux-gnu;/home/spack/spack/opt/spack/linux-debian12-broadwell/gcc-12.2.0/gcc-10.2.0-rngqwhteky3csksvpgujyc2dpdyqyqej/include/c++/10.2.0/backward;/home/spack/spack/opt/spack/linux-debian12-broadwell/gcc-12.2.0/gcc-10.2.0-rngqwhteky3csksvpgujyc2dpdyqyqej/lib/gcc/x86_64-pc-linux-gnu/10.2.0/include;/usr/local/include;/home/spack/spack/opt/spack/linux-debian12-broadwell/gcc-12.2.0/gcc-10.2.0-rngqwhteky3csksvpgujyc2dpdyqyqej/lib/gcc/x86_64-pc-linux-gnu/10.2.0/include-fixed;/usr/include/x86_64-linux-gnu;/usr/include")
set(CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES "stdc++;m;gcc_s;gcc;c;gcc_s;gcc")
set(CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES "/usr/local/cuda-11.1/targets/x86_64-linux/lib/stubs;/usr/local/cuda-11.1/targets/x86_64-linux/lib;/home/spack/spack/opt/spack/linux-debian12-broadwell/gcc-12.2.0/gcc-10.2.0-rngqwhteky3csksvpgujyc2dpdyqyqej/lib64;/home/spack/spack/opt/spack/linux-debian12-broadwell/gcc-12.2.0/gcc-10.2.0-rngqwhteky3csksvpgujyc2dpdyqyqej/lib/gcc/x86_64-pc-linux-gnu/10.2.0;/lib/x86_64-linux-gnu;/lib64;/usr/lib/x86_64-linux-gnu;/usr/lib64;/home/spack/spack/opt/spack/linux-debian12-broadwell/gcc-12.2.0/ncurses-6.4-7lr4y2bfbsx4gbztlu7rkf27r7qbg7tu/lib;/usr/local/cuda-11.1/lib64;/home/spack/spack/opt/spack/linux-debian12-broadwell/gcc-12.2.0/gcc-10.2.0-rngqwhteky3csksvpgujyc2dpdyqyqej/lib;/home/spack/spack/opt/spack/linux-debian12-broadwell/gcc-12.2.0/zstd-1.5.2-b7naonmcsrankrpr3l2djbhe7h3fmka5/lib;/home/spack/spack/opt/spack/linux-debian12-broadwell/gcc-12.2.0/zlib-1.2.13-35xuqkethhgeiwg5icruve6x4zmlybph/lib;/home/spack/spack/opt/spack/linux-debian12-broadwell/gcc-12.2.0/mpc-1.2.1-oc4526ngoxkwu7baaoz4xi2wj5tss5fl/lib;/home/spack/spack/opt/spack/linux-debian12-broadwell/gcc-12.2.0/mpfr-4.1.0-tc7th5la42lcp6ontcxivowhx2kihdxi/lib;/home/spack/spack/opt/spack/linux-debian12-broadwell/gcc-12.2.0/gmp-6.2.1-mkdkv7btdd2oxay6ocaqc4vbnqm5ph2u/lib")
set(CMAKE_CUDA_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")

set(CMAKE_CUDA_RUNTIME_LIBRARY_DEFAULT "STATIC")

set(CMAKE_LINKER "/usr/bin/ld")
set(CMAKE_AR "/usr/bin/ar")
set(CMAKE_MT "")
