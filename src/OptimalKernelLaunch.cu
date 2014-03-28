#include"OptimalKernelLaunch.h"
#include"constants.h"

//#include"functions.cpp"
#ifdef  KERNEL_OPT_FNS

int ceil(int a, int b) {
    return ceil(float(a) / b) * b;
}

//Register limitation
int RegistersPerBlock(struct Kernel_parameters& input, struct Physical_Limits& config) {
	int WarpsPerBlock = ceil( float(input.threadsPerBlock) / config.threadsPerWarp);
	if (strcmp(config.allocationGranularity,"block") == 0)
		return ceil( ceil(WarpsPerBlock, config.warpAllocationGranularity) * input.registersPerThread * config.threadsPerWarp, config.registerAllocationUnitSize);
	else
		return ceil(input.registersPerThread * config.threadsPerWarp, config.registerAllocationUnitSize) * WarpsPerBlock;
}

int Blocks_limitbyRegisters(struct Kernel_parameters& input, struct Physical_Limits& config) {
	if (input.registersPerThread>0){
		return floor(float(config.registerFileSize) / RegistersPerBlock(input, config));
	}
	else
		return config.threadBlocksPerMultiprocessor;
}

//Shared memory limitation
int SharedMemoryPerBlock(struct Kernel_parameters input, struct Physical_Limits& config) {
	return ceil(input.sharedMemoryPerBlock, config.sharedMemoryAllocationUnitSize);
}

int Blocks_limitbySharedMemory(struct Kernel_parameters input, struct Physical_Limits& config) {
	if (input.sharedMemoryPerBlock > 0)
		return int(float(config.sharedMemoryPerMultiprocessor) / SharedMemoryPerBlock(input, config));
	else
		return config.threadBlocksPerMultiprocessor;
}

//Calculate the occupancy for given version, ThreadsPerBlock, Reg, Sh.Mem
float occupancy(struct Kernel_parameters input, struct Physical_Limits& config) {
	
	int WarpsPerBlock = ceil( float( input.threadsPerBlock) / config.threadsPerWarp);
	// Calculate limits on no_of_blocks_per_SM
	int limitbyWarps = config.warpsPerMultiprocessor/WarpsPerBlock;
	int limitbyRegisters = Blocks_limitbyRegisters(input, config);
	int limitbySharedMemory = Blocks_limitbySharedMemory(input, config);
	int ActiveBlocksPerSM =  min(	min( limitbyWarps, config.threadBlocksPerMultiprocessor),
					min( limitbyRegisters, limitbySharedMemory));
	//int ActiveThreadsPerSM =  ActiveBlocksPerSM * input.threadsPerBlock;
	int ActiveWarpsPerSM =  ActiveBlocksPerSM * WarpsPerBlock;
	float Occupancy= float(ActiveWarpsPerSM) / config.warpsPerMultiprocessor;
	return Occupancy;
    }
// Calculates occupany and print all stats during calculation
void printstats(struct Kernel_parameters input, struct Physical_Limits& config) {
	
	int WarpsPerBlock = ceil( float( input.threadsPerBlock) / config.threadsPerWarp);
	// Calculate limits on no_of_blocks_per_SM
	int limitbyWarps = config.warpsPerMultiprocessor/WarpsPerBlock;
	int limitbyRegisters = Blocks_limitbyRegisters(input, config);
	int limitbySharedMemory = Blocks_limitbySharedMemory(input, config);
	int ActiveBlocksPerSM =  min(	min( limitbyWarps, config.threadBlocksPerMultiprocessor),
					min( limitbyRegisters, limitbySharedMemory));
	int ActiveThreadsPerSM =  ActiveBlocksPerSM * input.threadsPerBlock;
	int ActiveWarpsPerSM =  ActiveBlocksPerSM * WarpsPerBlock;
	float Occupancy= float(ActiveWarpsPerSM) / config.warpsPerMultiprocessor;

	std::cout<<"activeThreadsPerMultiprocessor: "<< ActiveThreadsPerSM;
	std::cout<<"\nactiveWarpsPerMultiprocessor: "<< ActiveWarpsPerSM;
	std::cout<<"\nactiveThreadBlocksPerMultiprocessor: "<< ActiveBlocksPerSM;
	std::cout<<"\noccupancyOfMultiprocessor: "<< Occupancy;
	std::cout<<"\nblockWarps: "<< WarpsPerBlock;
	std::cout<<"\nblockSharedMemory: "<< SharedMemoryPerBlock(input, config);
	std::cout<<"\nblockRegisters: "<< RegistersPerBlock(input, config);
	std::cout<<"\nthreadBlocksPerMultiprocessorLimitedByWarpsOrBlocksPerMultiprocessor: "<< min( limitbyWarps, config.threadBlocksPerMultiprocessor);
	std::cout<<"\nthreadBlocksPerMultiprocessorLimitedByRegistersPerMultiprocessor: "<< limitbyRegisters;
	std::cout<<"\nthreadBlocksPerMultiprocessorLimitedBySharedMemoryPerMultiprocessor: "<< limitbySharedMemory;
}
//Search for sm version in the GPU Physical limits array
int index_of_sm(int version){
	int i=0;
	while( sm_[i].version != -1){
		if(sm_[i].version==version)
			return i;
	i++;
	}
	std::cerr<<"sm_"<<version<<" not found\n";
	exit(EXIT_FAILURE);
}

//Library Functions
void printattr(struct cudaFuncAttributes attr){
	std::cout<<"Binary Version:"<<attr.binaryVersion<<std::endl;
	std::cout<<"constant memory(bytes):"<<attr.constSizeBytes<<std::endl;
	std::cout<<"local memory(bytes):"<<attr.localSizeBytes<<std::endl;
	std::cout<<"Max Threads per Block:"<<attr.maxThreadsPerBlock<<std::endl;
	std::cout<<"No of Registers:"<<attr.numRegs<<std::endl;
	std::cout<<"PTX Version:"<<attr.ptxVersion<<std::endl;
	std::cout<<"Shared memory(bytes):"<<attr.sharedSizeBytes<<std::endl;
}
cudaError_t printKernelAtrributes(const char *  	func){
	struct cudaFuncAttributes attr;
	cudaError_t err = cudaFuncGetAttributes( &attr, func);
	if( err== cudaSuccess)
		printattr(attr);
	return err;
}

int print_all_occupancy(struct cudaFuncAttributes attr){
	struct Kernel_parameters parameters={attr.binaryVersion, 0, attr.numRegs, attr.sharedSizeBytes};
	#ifdef DEBUG
	printattr(attr);
	#endif
	int sm_index=index_of_sm(parameters.version);
	int maxThreadsPerBlock=sm_[sm_index].MaxThreadBlockSize;
	typedef struct occlist_{
		float occ;
		int tpb;
		static int compare(const void *a, const void *b) {
		return ((*(occlist_*)a ).occ - (*(occlist_*)b).occ)>0?-1:+1; }
	}occlist;
	occlist OccAll[maxThreadsPerBlock/32];
	for(int tpb=32,i=0; tpb<=maxThreadsPerBlock; tpb+=32,i+=1){
		parameters.threadsPerBlock=tpb;
		OccAll[i].occ=occupancy(parameters, sm_[sm_index]);
		OccAll[i].tpb=tpb;
		std::cout<<tpb<<"\t"<<OccAll[i].occ<<std::endl;
		//printstats(parameters, sm_[sm_index]);
	}
	qsort(OccAll, sizeof(OccAll)/sizeof(occlist), sizeof(occlist), OccAll[0].compare);
	int max_tpb=OccAll[0].tpb; //float max=OccAll[0].occ;
	return max_tpb;
}

int input_kernel_params(){
	struct cudaFuncAttributes attr;
	//Input parameters
	std::cout<<" SM Version(10,11,12,13,,20,21,30,35):";	std::cin>>attr.binaryVersion;
	std::cout<<" Registers per Thread:";			std::cin>>attr.numRegs;
	std::cout<<" Shared Memory per Block:";			std::cin>>attr.sharedSizeBytes;
	//std::cout<<"constant memory(bytes):";			std::cin>>attr.constSizeBytes;
	//std::cout<<"local memory(bytes):";			std::cin>>attr.localSizeBytes;
	return print_all_occupancy(attr);
}

float get_occupancy(struct cudaFuncAttributes attr, int TPB){
	struct Kernel_parameters parameters={attr.binaryVersion, TPB, attr.numRegs, attr.sharedSizeBytes};
	int sm_index=index_of_sm(parameters.version);
	return occupancy(parameters, sm_[sm_index]);
}

//Template functions are defined inside headerfile

int print_all_occupancy(const void*  	func){
	struct cudaFuncAttributes attr;
	cudaError_t err = cudaFuncGetAttributes( &attr, func);
	if( err== cudaSuccess)
		printattr(attr);
	return print_all_occupancy(attr);
}

float get_occupancy(const void*  	func, int TPB){
	struct cudaFuncAttributes attr;
	cudaError_t err = cudaFuncGetAttributes( &attr, func);
	struct Kernel_parameters parameters={attr.binaryVersion, TPB, attr.numRegs, attr.sharedSizeBytes};
	int sm_index=index_of_sm(parameters.version);
	return occupancy(parameters, sm_[sm_index]);
}

int get_best_TPB(const void*  	func){
	struct cudaFuncAttributes attr;
	cudaError_t err = cudaFuncGetAttributes( &attr, func);
	#ifdef DEBUG
	if( err== cudaSuccess)
		printattr(attr);
	#endif
	struct Kernel_parameters parameters={attr.binaryVersion, 0, attr.numRegs, attr.sharedSizeBytes};
	int sm_index=index_of_sm(parameters.version);
	int maxThreadsPerBlock=sm_[sm_index].MaxThreadBlockSize;
	float occ, max=0; int max_tpb=0;
	for(int tpb=32; tpb<=maxThreadsPerBlock; tpb+=32){
		parameters.threadsPerBlock=tpb;
		occ=occupancy(parameters, sm_[sm_index]);
		if( (!(occ<max)) || (occ==max) || (max<occ)){
			max=occ;
			max_tpb=tpb;
		}
		//std::cout<<tpb<<"\t"<<occ<<std::endl;
		//printstats(parameters, sm_[sm_index]);
	}
	return max_tpb;
}

#endif
