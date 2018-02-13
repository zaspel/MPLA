#include <stdio.h>
#include "mpla.h"
#include <curand.h>

#include "generic_system_adapter.h"
#include "kernel_system_assembler.h"
__global__ void gen_generic_system_matrix(double* A, system_assembler* assem, int row_count, int col_count, int row_offset, int col_offset)
{
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	
	if (idx<row_count*col_count)
	{
		// get row and col index in matrix
		int j = idx / row_count;
		int i = idx % row_count;

		int i_global = i+row_offset;
		int j_global = j+col_offset;

		double result =  assem->get_matrix_entry(i_global, j_global);
		A[idx] = result;
	}
}


__global__ void gen_generic_system_rhs(double* rhs, system_assembler* assem, int row_count, int row_offset)
{
	int idx = blockIdx.x*blockDim.x+threadIdx.x;

	if (idx<row_count)
	{
		// get row index in rhs
		int i_global = idx + row_offset;

		rhs[idx] = assem->get_rhs_entry(i_global);
	}
}

void mpla_set_generic_system_rhs(struct mpla_vector* b, struct mpla_generic_matrix* A, struct mpla_instance* instance)
{
        int block_size = 1024;
        int grid_size = (A->cur_proc_row_count + block_size -1)/block_size;

	system_assembler* assem = (system_assembler*)(A->data);

        gen_generic_system_rhs<<<grid_size, block_size>>>(b->data, assem, A->cur_proc_row_count, A->cur_proc_row_offset);
        cudaThreadSynchronize();
        checkCUDAError("gen_gaussian_kernel_rhs");

}


void mpla_dgemv_core_generic_system_matrix_cublas(struct mpla_vector* b, struct mpla_generic_matrix* A, struct mpla_vector* x_redist, struct mpla_instance* instance)
{
	double* Amat;
	cudaMalloc((void**)&Amat, sizeof(double)*A->cur_proc_row_count*A->cur_proc_col_count);
	checkCUDAError("cudaMalloc in mpla_dgemv_core_kernel_cublas");
	
	int block_size = 1024;
	int grid_size = (A->cur_proc_row_count*A->cur_proc_col_count + block_size -1)/block_size;

	system_assembler* assem = (system_assembler*)(A->data);

	gen_generic_system_matrix<<<grid_size, block_size>>>(Amat, assem, A->cur_proc_row_count, A->cur_proc_col_count, A->cur_proc_row_offset, A->cur_proc_col_offset);
	cudaThreadSynchronize();
	checkCUDAError("gen_generic_system_matrix_matrix3");
	
        double one = 1;
        double zero = 0;
        cublasDgemv((instance->cublas_handle), CUBLAS_OP_N, A->cur_proc_row_count, A->cur_proc_col_count, &one, Amat, A->cur_proc_row_count, x_redist->data, 1, &zero, b->data, 1);

	cudaFree(Amat);

}

__global__ void get_max_row_count_per_dgemv_kernel(struct system_assembler* assem, int* max_row_count_per_dgemv)
{
	max_row_count_per_dgemv[0] = assem->max_row_count_per_dgemv;
}



void mpla_dgemv_core_generic_system_matrix_streamed_cublas(struct mpla_vector* b, struct mpla_generic_matrix* A, struct mpla_vector* x_redist, struct mpla_instance* instance)
{
	int* max_row_count_per_dgemv_d;
	cudaMalloc((void**)&max_row_count_per_dgemv_d, sizeof(int));
	checkCUDAError("bbbdf");
	get_max_row_count_per_dgemv_kernel<<<1,1>>>((system_assembler*)A->data, max_row_count_per_dgemv_d);
	cudaThreadSynchronize();
	checkCUDAError("fasdfasf");
	int max_row_count_per_dgemv_h;
	cudaMemcpy(&max_row_count_per_dgemv_h, max_row_count_per_dgemv_d, sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(max_row_count_per_dgemv_d);	

	int max_row_count = max_row_count_per_dgemv_h;

	max_row_count = min(max_row_count, A->cur_proc_row_count);
	
	double* Amat;
	cudaMalloc((void**)&Amat, sizeof(double)*max_row_count*A->cur_proc_col_count);
	checkCUDAError("cudaMalloc in mpla_dgemv_core_kernel_cublas");

	int block_size = 1024;
	int grid_size = (max_row_count*A->cur_proc_col_count + block_size -1)/block_size;

	system_assembler* assem = (system_assembler*)(A->data);

	int row_block_count = (A->cur_proc_row_count + max_row_count - 1) / max_row_count;
	int curr_row_block_offset = A->cur_proc_row_offset;

//	printf("mrc %d   rbc: %d\n", max_row_count, row_block_count);

	for (int curr_row_block=0; curr_row_block<row_block_count; curr_row_block++)
	{
		int curr_row_block_size = (curr_row_block < row_block_count-1) ? max_row_count : A->cur_proc_row_count-(max_row_count*(row_block_count-1));
	//	printf("crbs: %d\n", curr_row_block_size);
//		printf("stream %d: %d %d %d\n", instance->cur_proc_rank, curr_row_block_size, curr_row_block_offset, row_block_count);
		gen_generic_system_matrix<<<grid_size, block_size>>>(Amat, assem, curr_row_block_size, A->cur_proc_col_count, curr_row_block_offset, A->cur_proc_col_offset);
		cudaThreadSynchronize();
		checkCUDAError("gen_generic_system_matrix3");

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

