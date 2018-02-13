#include <stdio.h>
#include "mpla.h"
#include <curand.h>

#include "system_assembler.h"
#include "generic_system_adapter.h"
#include "kernel_system_assembler.h"


__global__ void create_gaussian_kernel_system_assembler_object_kernel(struct gaussian_kernel_system_assembler** assem, double** points, int max_row_count_per_dgemv, int dim)
{
	(*assem) = new gaussian_kernel_system_assembler();
	(*assem)->points = points;
	(*assem)->max_row_count_per_dgemv = max_row_count_per_dgemv;
	(*assem)->dim = dim;
}

void create_gaussian_kernel_system_assembler_object(struct gaussian_kernel_system_assembler** assem, double** points, int max_row_count_per_dgemv, int dim)
{
	create_gaussian_kernel_system_assembler_object_kernel<<<1,1>>>(assem, points, max_row_count_per_dgemv, dim);
	cudaThreadSynchronize();
	checkCUDAError("create_gaussian_kernel_system_assembler_object");
}


__global__ void destroy_gaussian_kernel_system_assembler_object_kernel(struct gaussian_kernel_system_assembler** assem)
{
	delete *assem;
}

void destroy_gaussian_kernel_system_assembler_object(struct gaussian_kernel_system_assembler** assem)
{
	destroy_gaussian_kernel_system_assembler_object_kernel<<<1,1>>>(assem);
	cudaThreadSynchronize();
 	checkCUDAError("destroy_gaussian_kernel_system_assembler_object");
}



