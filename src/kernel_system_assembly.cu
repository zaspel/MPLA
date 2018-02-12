#include <stdio.h>
#include "mpla.h"
#include <curand.h>

#include "system_assembly.h"
#include "generic_system_adapter.h"
#include "kernel_system_assembly.h"


__global__ void create_gaussian_kernel_system_assembly_object_kernel(struct gaussian_kernel_system_assembly** assem, double** points, int max_row_count_per_dgemv, int dim)
{
	(*assem) = new gaussian_kernel_system_assembly();
	(*assem)->points = points;
	(*assem)->max_row_count_per_dgemv = max_row_count_per_dgemv;
	(*assem)->dim = dim;
}

void create_gaussian_kernel_system_assembly_object(struct gaussian_kernel_system_assembly** assem, double** points, int max_row_count_per_dgemv, int dim)
{
	create_gaussian_kernel_system_assembly_object_kernel<<<1,1>>>(assem, points, max_row_count_per_dgemv, dim);
	cudaThreadSynchronize();
	checkCUDAError("create_gaussian_kernel_system_assembly_object");
}


__global__ void destroy_gaussian_kernel_system_assembly_object_kernel(struct gaussian_kernel_system_assembly** assem)
{
	delete *assem;
}

void destroy_gaussian_kernel_system_assembly_object(struct gaussian_kernel_system_assembly** assem)
{
	destroy_gaussian_kernel_system_assembly_object_kernel<<<1,1>>>(assem);
	cudaThreadSynchronize();
 	checkCUDAError("destroy_gaussian_kernel_system_assembly_object");
}



