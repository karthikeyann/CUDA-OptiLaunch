#include<iostream>

#include<OptimalKernelLaunch.h>
using namespace std;

__global__ void kernel(int *a, int *b, int *c, int N){
	int i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i<N)
		c[i]=a[i]+b[i];
}
int main(){
	//Pass a kernel
	print_all_occupancy( kernel );
	/*
	struct cudaFuncAttributes attr;
	cudaFuncGetAttributes( &attr, kernel);
	printattr(attr);
	print_all_occupancy(attr);
	*/

	int best=get_best_TPB(kernel);
	cout<<"\nBest TPB="<<best;
	cout<<"\nOccupancy="<<get_occupancy(kernel, best)<<endl;
	
	//Get the parameters from user
	cout<<"Input Kernel Parameters\n";
	cout<<"Best TPB="<<input_kernel_params()<<endl;
	return 0;
}
