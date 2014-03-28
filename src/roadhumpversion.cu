#include"OptimalKernelLaunch.h"
#include"constants.h"
// Converted by Karthikeyan
// Converted from Javascript to C
// Used as a reference for my functions
#ifndef  KERNEL_OPT_FNS
#define KERNEL_OPT_FNS
  float ceil(float a, float b) {
    return ceil(a / b) * b;
  }


//Functions
//1
    float blockWarps(struct Kernel_parameters input, struct Physical_Limits& config) {
      return ceil(float(input.threadsPerBlock) / config.threadsPerWarp);
    }
//2
    float blockRegisters(struct Kernel_parameters input, struct Physical_Limits& config) {
      if (strcmp(config.allocationGranularity,"block")==0) {
        return ceil(ceil(blockWarps(input, config), config.warpAllocationGranularity) * input.registersPerThread * config.threadsPerWarp, config.registerAllocationUnitSize);
      } else {
        return ceil(input.registersPerThread * config.threadsPerWarp, config.registerAllocationUnitSize) * blockWarps(input, config);
      }
    }
//3
    float blockSharedMemory(struct Kernel_parameters input, struct Physical_Limits& config) {
      return ceil(input.sharedMemoryPerBlock, config.sharedMemoryAllocationUnitSize);
    }
//4
    float threadBlocksPerMultiprocessorLimitedByWarpsOrBlocksPerMultiprocessor(struct Kernel_parameters input, struct Physical_Limits& config) {
      return min( float(config.threadBlocksPerMultiprocessor), floor(float(config.warpsPerMultiprocessor) / blockWarps(input, config)));
    }
//5
    float threadBlocksPerMultiprocessorLimitedByRegistersPerMultiprocessor(struct Kernel_parameters input, struct Physical_Limits& config) {
      if (input.registersPerThread > 0) {
        return floor(float(config.registerFileSize) / blockRegisters(input, config));
      } else {
        return config.threadBlocksPerMultiprocessor;
      }
    }
//6
    float threadBlocksPerMultiprocessorLimitedBySharedMemoryPerMultiprocessor(struct Kernel_parameters input, struct Physical_Limits& config) {
      if (input.sharedMemoryPerBlock > 0) {
        return floor(float(config.sharedMemoryPerMultiprocessor) / blockSharedMemory(input, config));
      } else {
        return config.threadBlocksPerMultiprocessor;
      }
    }
//9
    float activeThreadBlocksPerMultiprocessor(struct Kernel_parameters input, struct Physical_Limits& config) {
      return min(threadBlocksPerMultiprocessorLimitedByWarpsOrBlocksPerMultiprocessor(input, config), min( threadBlocksPerMultiprocessorLimitedByRegistersPerMultiprocessor(input, config), threadBlocksPerMultiprocessorLimitedBySharedMemoryPerMultiprocessor(input, config) ));
    }
//7
    float activeThreadsPerMultiprocessor(struct Kernel_parameters input, struct Physical_Limits& config) {
      return input.threadsPerBlock * activeThreadBlocksPerMultiprocessor(input, config);
    }
//8
    float activeWarpsPerMultiprocessor(struct Kernel_parameters input, struct Physical_Limits& config) {
      return activeThreadBlocksPerMultiprocessor(input, config) * blockWarps(input, config);
    }
//10
    float occupancyOfMultiprocessor(struct Kernel_parameters input, struct Physical_Limits& config) {
      return float( activeWarpsPerMultiprocessor(input, config) ) / float(config.warpsPerMultiprocessor);
    }
  void printstats(struct Kernel_parameters input, struct Physical_Limits& config) {
      std::cout<<"activeThreadsPerMultiprocessor: "<< activeThreadsPerMultiprocessor(input, config);
      std::cout<<"\nactiveWarpsPerMultiprocessor: "<< activeWarpsPerMultiprocessor(input, config);
      std::cout<<"\nactiveThreadBlocksPerMultiprocessor: "<< activeThreadBlocksPerMultiprocessor(input, config);
      std::cout<<"\noccupancyOfMultiprocessor: "<< occupancyOfMultiprocessor(input, config);
      std::cout<<"\nblockWarps: "<< blockWarps(input, config);
      std::cout<<"\nblockSharedMemory: "<< blockSharedMemory(input, config);
      std::cout<<"\nblockRegisters: "<< blockRegisters(input, config);
      std::cout<<"\nthreadBlocksPerMultiprocessorLimitedByWarpsOrBlocksPerMultiprocessor: "<< threadBlocksPerMultiprocessorLimitedByWarpsOrBlocksPerMultiprocessor(input, config);
      std::cout<<"\nthreadBlocksPerMultiprocessorLimitedByRegistersPerMultiprocessor: "<< threadBlocksPerMultiprocessorLimitedByRegistersPerMultiprocessor(input, config);
      std::cout<<"\nthreadBlocksPerMultiprocessorLimitedBySharedMemoryPerMultiprocessor: "<< threadBlocksPerMultiprocessorLimitedBySharedMemoryPerMultiprocessor(input, config)<<"\n";
  }
#endif
