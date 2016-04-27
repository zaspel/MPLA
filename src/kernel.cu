#include <stdio.h>
#include "mpla.h"
#include <curand.h>

#include "kernel.h"

__device__ double matern_kernel(double r)
{
	return (1.0+r)*exp(-r);
}

__global__ void gen_dis_matrix3(double* A, double** points, int row_count, int col_count, int row_offset, int col_offset)
{
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	
	if (idx<row_count*col_count)
	{
		// get row and col index in matrix
		int j = idx / row_count;
		int i = idx % row_count;

		int i_global = i+row_offset;
		int j_global = j+col_offset;

		A[idx] =  sqrt(   (points[0][i_global]-points[0][j_global])*(points[0][i_global]-points[0][j_global])	\
				+ (points[1][i_global]-points[1][j_global])*(points[1][i_global]-points[1][j_global])	\
	 			+ (points[2][i_global]-points[2][j_global])*(points[2][i_global]-points[2][j_global])  );
	}
}

__global__ void apply_kernel(double* A, int count)
{
	int idx = blockIdx.x*blockDim.x+threadIdx.x;

	if (idx<count)
	{
		A[idx] = matern_kernel(A[idx]);
	}
}

void mpla_dgemv_core_kernel_cublas(struct mpla_vector* b, struct mpla_generic_matrix* A, struct mpla_vector* x_redist, struct mpla_instance* instance)
{
	double* Amat;
	cudaMalloc((void**)&Amat, sizeof(double)*A->cur_proc_row_count*A->cur_proc_col_count);
	checkCUDAError("cudaMalloc in mpla_dgemv_core_kernel_cublas");
	
	int block_size = 1024;
	int grid_size = (A->cur_proc_row_count*A->cur_proc_col_count + block_size -1)/block_size;

	double** points = ((struct kernel_matrix_data*)(A->data))->points;

	gen_dis_matrix3<<<grid_size, block_size>>>(Amat, points, A->cur_proc_row_count, A->cur_proc_col_count, A->cur_proc_row_offset, A->cur_proc_col_offset);
	cudaThreadSynchronize();
	checkCUDAError("gen_dist_matrix3");
	
	apply_kernel<<<grid_size, block_size>>>(Amat, A->cur_proc_row_count*A->cur_proc_col_count);
	cudaThreadSynchronize();
	checkCUDAError("apply_kernel");


        double one = 1;
        double zero = 0;
        cublasDgemv((instance->cublas_handle), CUBLAS_OP_N, A->cur_proc_row_count, A->cur_proc_col_count, &one, Amat, A->cur_proc_row_count, x_redist->data, 1, &zero, b->data, 1);

	cudaFree(Amat);
}


void mpla_dgemv_core_kernel_streamed_cublas(struct mpla_vector* b, struct mpla_generic_matrix* A, struct mpla_vector* x_redist, struct mpla_instance* instance)
{
	printf("Applying MVP\n");
	int max_row_count = ((struct kernel_matrix_data*)(A->data))->max_row_count_per_dgemv;

	double* Amat;
	cudaMalloc((void**)&Amat, sizeof(double)*max_row_count*A->cur_proc_col_count);
	checkCUDAError("cudaMalloc in mpla_dgemv_core_kernel_cublas");

	int block_size = 1024;
	int grid_size = (max_row_count*A->cur_proc_col_count + block_size -1)/block_size;

	double** points = ((struct kernel_matrix_data*)(A->data))->points;

	int row_block_count = (A->cur_proc_row_count + max_row_count - 1) / max_row_count;

	int curr_row_block_offset = A->cur_proc_row_offset;

	printf("rbc %d  mrc %d\n", row_block_count, max_row_count);

	for (int curr_row_block=0; curr_row_block<row_block_count; curr_row_block++)
	{
		int curr_row_block_size = (curr_row_block < row_block_count-1) ? max_row_count : A->cur_proc_row_count-(max_row_count*(row_block_count-1));
		gen_dis_matrix3<<<grid_size, block_size>>>(Amat, points, curr_row_block_size, A->cur_proc_col_count, curr_row_block_offset, A->cur_proc_col_offset);
		cudaThreadSynchronize();
		checkCUDAError("gen_dist_matrix3");
	
		apply_kernel<<<grid_size, block_size>>>(Amat, curr_row_block_size*A->cur_proc_col_count);
		cudaThreadSynchronize();
		checkCUDAError("apply_kernel");

		double one = 1;
	        double zero = 0;
        	cublasDgemv((instance->cublas_handle), CUBLAS_OP_N, curr_row_block_size, A->cur_proc_col_count, &one, Amat, curr_row_block_size, x_redist->data, 1, &zero, &((b->data)[curr_row_block_offset]), 1);

		curr_row_block_offset = curr_row_block_offset + curr_row_block_size;
	}

	cudaFree(Amat);
}


