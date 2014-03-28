CUDA-OptiLaunch
===============
Finds the best kernel launch parameters/Configuration for a given kernel in runtime.

Using CUDA-OptiLaunch
---------------------


    #include<OptimalKernelLaunch.h>

    int best=get_best_TPB(kernel);
    int occupancy=get_occupancy(kernel, best)<<endl;

Compile:
--------
Before compiling, make sure **nvcc** in **$PATH** and CUDA libraries in **$LD_LIBRARY_PATH**

    nvcc -c yourfile.cu -I src
    nvcc yourfile.o OptimalKernelLaunch.o

Test sample code:
-----------------
Test sample code by running

    make clean
    make
    ./test

Credits:
--------

* [Karthikeyan](https://github.com/lxkarthi/cuda-calculator) - Improved cuda-calculator online version. 
* [Aliaksei](http://github.com/roadhump) - Original Author deleted it. :( 
* [Mihai Maruseac](https://github.com/mihaimaruseac/cuda-calculator) - Thanks for the fork. 

