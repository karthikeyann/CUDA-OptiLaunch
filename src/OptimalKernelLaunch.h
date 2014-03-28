#ifndef  KERNEL_OPT_FNS
#define KERNEL_OPT_FNS
#include<iostream>
#include<cmath>
#include<cstring>

struct Physical_Limits{
// All device parameters from xls
int version; //ComputeCapability;
int threadsPerWarp;
int warpsPerMultiprocessor;
int threadsPerMultiprocessor;
int threadBlocksPerMultiprocessor;
int sharedMemoryPerMultiprocessor;
int registerFileSize;
int registerAllocationUnitSize;
char *allocationGranularity;
int MaxRegistersPerThread;
int sharedMemoryAllocationUnitSize;
//Extras not used in code, kept for future use
int warpAllocationGranularity;
int MaxThreadBlockSize;
int SharedMemorySizeConfigurations[3];
int Warpregisterallocationgranularities[2];
};

struct Kernel_parameters{
int version; //from attribute
int threadsPerBlock; //find the best**
int registersPerThread; //from attribute (int)
int sharedMemoryPerBlock; //from attribute (size_t bytes)
};

//Constants
extern struct Physical_Limits sm_[];

//#include"functions.cpp"
int ceil(int a, int b);

//Register limitation
int RegistersPerBlock(struct Kernel_parameters& input, struct Physical_Limits& config);
int Blocks_limitbyRegisters(struct Kernel_parameters& input, struct Physical_Limits& config);
//Shared memory limitation
int SharedMemoryPerBlock(struct Kernel_parameters input, struct Physical_Limits& config);
int Blocks_limitbySharedMemory(struct Kernel_parameters input, struct Physical_Limits& config);

//Calculate the occupancy for given version, ThreadsPerBlock, Reg, Sh.Mem
float occupancy(struct Kernel_parameters input, struct Physical_Limits& config);
// Calculates occupany and print all stats during calculation
void printstats(struct Kernel_parameters input, struct Physical_Limits& config);

//Search for sm version in the GPU Physical limits array
int index_of_sm(int version);

//Library Functions

//Attributes 
void printattr(struct cudaFuncAttributes attr);
cudaError_t printKernelAtrributes(const char *  	func);

//Print all occupancies and TPBs
int print_all_occupancy(struct cudaFuncAttributes attr);
int input_kernel_params();

//Get occupancy
float get_occupancy(struct cudaFuncAttributes attr, int TPB);

//Template Functions
int print_all_occupancy(const void*  	func);
template<class T > 
int print_all_occupancy(T*  	func){
  return ::print_all_occupancy((const void*)func);
}

float get_occupancy(const void*  	func, int TPB);
template<class T > 
float get_occupancy(T*  	func, int TPB){
	return ::get_occupancy((const void*)func, TPB);
}

int get_best_TPB(const void*  	func);
template<class T > 
int get_best_TPB(T*  	func){
  return ::get_best_TPB((const void*)func);
}

#endif
