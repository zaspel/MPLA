#include <stdio.h>
#include "mpla.h"
#include <curand.h>

#include "kernel.h"

__device__ double matern_kernel(double r)
{
	return (1.0+r)*exp(-r);
}

__device__ double gaussian_kernel(double r)
{
	return exp(-r*r);
}

__global__ void gen_dis_matrix(double* A, double** points, int dim, int row_count, int col_count, int row_offset, int col_offset)
{
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	
	if (idx<row_count*col_count)
	{
		// get row and col index in matrix
		int j = idx / row_count;
		int i = idx % row_count;

		int i_global = i+row_offset;
		int j_global = j+col_offset;

		double result = 0.0;

		for (int d=0; d<dim; d++)
			result = result + (points[d][i_global]-points[d][j_global])*(points[d][i_global]-points[d][j_global]);

		A[idx] =  sqrt( result );
	}
}

__global__ void apply_kernel(double* A, int count)
{
	int idx = blockIdx.x*blockDim.x+threadIdx.x;

	if (idx<count)
	{
		if (fabs(A[idx])>=1.0e-14)
			A[idx] = gaussian_kernel(A[idx]);
		else
			A[idx] = gaussian_kernel(A[idx])+1.0e-8;
	}
}

__global__ void gen_gaussian_kernel_rhs(double* rhs, double** points, int dim, int row_count, int row_offset)
{
	int idx = blockIdx.x*blockDim.x+threadIdx.x;

	if (idx<row_count)
	{
		// get row index in rhs
		int i_global = idx + row_offset;
	
		double result = 1.0;

		for (int d=0; d<dim; d++)
			result = result * (sqrt(M_PI)/2.0) * (erf(1.0- points[d][i_global]) - erf(0.0 - points[d][i_global]));

		rhs[idx] = result;
	}
}

void mpla_set_gaussian_kernel_rhs(struct mpla_vector* b, struct mpla_generic_matrix* A, struct mpla_instance* instance)
{
        int block_size = 1024;
        int grid_size = (A->cur_proc_row_count + block_size -1)/block_size;

        double** points = ((struct kernel_matrix_data*)(A->data))->points;
	int dim = ((struct kernel_matrix_data*)(A->data))->dim;

        gen_gaussian_kernel_rhs<<<grid_size, block_size>>>(b->data, points, dim, A->cur_proc_row_count, A->cur_proc_row_offset);
        cudaThreadSynchronize();
        checkCUDAError("gen_gaussian_kernel_rhs");

}




void mpla_dgemv_core_kernel_cublas(struct mpla_vector* b, struct mpla_generic_matrix* A, struct mpla_vector* x_redist, struct mpla_instance* instance)
{
	double* Amat;
	cudaMalloc((void**)&Amat, sizeof(double)*A->cur_proc_row_count*A->cur_proc_col_count);
	checkCUDAError("cudaMalloc in mpla_dgemv_core_kernel_cublas");
	
	int block_size = 1024;
	int grid_size = (A->cur_proc_row_count*A->cur_proc_col_count + block_size -1)/block_size;

	double** points = ((struct kernel_matrix_data*)(A->data))->points;
	int dim = ((struct kernel_matrix_data*)(A->data))->dim;

	gen_dis_matrix<<<grid_size, block_size>>>(Amat, points, dim, A->cur_proc_row_count, A->cur_proc_col_count, A->cur_proc_row_offset, A->cur_proc_col_offset);
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
	int max_row_count = ((struct kernel_matrix_data*)(A->data))->max_row_count_per_dgemv;

	max_row_count = min(max_row_count, A->cur_proc_row_count);
	
//	printf("%d\n",max_row_count);

	double* Amat;
	cudaMalloc((void**)&Amat, sizeof(double)*max_row_count*A->cur_proc_col_count);
	checkCUDAError("cudaMalloc in mpla_dgemv_core_kernel_cublas");

	int block_size = 1024;
	int grid_size = (max_row_count*A->cur_proc_col_count + block_size -1)/block_size;

	double** points = ((struct kernel_matrix_data*)(A->data))->points;

	int dim = ((struct kernel_matrix_data*)(A->data))->dim;

	int row_block_count = (A->cur_proc_row_count + max_row_count - 1) / max_row_count;

	int curr_row_block_offset = A->cur_proc_row_offset;

//	printf("mrc %d   rbc: %d\n", max_row_count, row_block_count);

	for (int curr_row_block=0; curr_row_block<row_block_count; curr_row_block++)
	{
		int curr_row_block_size = (curr_row_block < row_block_count-1) ? max_row_count : A->cur_proc_row_count-(max_row_count*(row_block_count-1));
	//	printf("crbs: %d\n", curr_row_block_size);
//		printf("stream %d: %d %d %d\n", instance->cur_proc_rank, curr_row_block_size, curr_row_block_offset, row_block_count);
		gen_dis_matrix<<<grid_size, block_size>>>(Amat, points, dim, curr_row_block_size, A->cur_proc_col_count, curr_row_block_offset, A->cur_proc_col_offset);
		cudaThreadSynchronize();
		checkCUDAError("gen_dist_matrix3");
	
		apply_kernel<<<grid_size, block_size>>>(Amat, curr_row_block_size*A->cur_proc_col_count);
		cudaThreadSynchronize();
		checkCUDAError("apply_kernel");

		double one = 1;
	        double zero = 0;
        	cublasDgemv((instance->cublas_handle), CUBLAS_OP_N, curr_row_block_size, A->cur_proc_col_count, &one, Amat, curr_row_block_size, x_redist->data, 1, &zero, &((b->data)[curr_row_block_offset-A->cur_proc_row_offset]), 1);

//		double* tmp=new double[curr_row_block_size];
//		cudaMemcpy(tmp, &((b->data)[curr_row_block_offset-A->cur_proc_row_offset]), sizeof(double)*curr_row_block_size, cudaMemcpyDeviceToHost);
//		for (int ii = 0; ii< curr_row_block_size; ii++)
//			printf("%le\n", tmp[ii]);
//		delete [] tmp;
		

		curr_row_block_offset = curr_row_block_offset + curr_row_block_size;
	}

	cudaFree(Amat);
}


