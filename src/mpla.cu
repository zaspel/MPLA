// Copyright (C) 2016 Peter Zaspel
//
// This file is part of MPLA.
//
// MPLA is free software: you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License as published by the Free
// Software Foundation, either version 3 of the License, or (at your option) any
// later version.
//
// MPLA is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
// A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
// details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with MPLA.  If not, see <http://www.gnu.org/licenses/>.



#include <stdio.h> 
#include "cublas_v2.h"

void info()
{
	printf("Hello world\n");
}

void mpla_init_instance(struct mpla_instance* instance, MPIComm comm)
{
	instance->comm = comm;	

	// get number of process
	MPI_Comm_size(comm, &(instance->proc_count));

	// find number of current process
	MPI_Comm_rank(comm, &(instance->cur_proc_rank));
	if (instance->current_proc==0)
		instance->is_parent = true;
	else
		instance->is_parent = false;

	// compute the process grid
	int dims[2]; 
	dims[0]=dims[1]=0;
	MPI_DIMS_CREATE(instance->proc_count, 2, dims);
	instance->proc_rows = dims[0];
	instance->proc_cols = dims[1];

	// create cartesian communicator and retrieve cartesian coordinates
	int periods[2];
	periods[0]=periods[1]=0;
	MPI_CART_CREATE(comm, 2, dims, periods, 0, &(instance->comm));
	int cur_proc_coord[2];
	MPI_CART_GET(instance->comm, 2, dims, periods, cur_proc_coord);
	instance->cur_proc_row = cur_proc_coord[0];
	instance->cur_proc_col = cur_proc_coord[1];
	
}

void mpla_init_matrix(struct mpla_matrix* matrix, struct mpla_instance* instance, int mat_row_count, int mat_col_count)
{
	// setting global matrix dimensions
	matrix->mat_row_count = mat_row_count;
	matrix->mat_col_count = mat_col_count;

	// allocating memory for process-wise matrix information
	matrix->proc_row_count = new int*[instance->proc_rows];
	matrix->proc_col_count = new int*[instance->proc_rows];
	matrix->proc_row_offset = new int*[instance->proc_rows];
	matrix->proc_col_offset = new int*[instance->proc_rows];
	for (int i=0; i<instance->proc_rows; i++)
	{
		matrix->proc_row_count[i] = new int[instance->proc_cols];
		matrix->proc_col_count[i] = new int[instance->proc_cols];
		matrix->proc_row_offset[i] = new int[instance->proc_cols];
		matrix->proc_col_offset[i] = new int[instance->proc_cols];
	}

	// computing general row block sizes
	int filled_row_block_size = ceil((float)mat_row_count / (float)(instance->proc_rows));
	int filled_row_block_count = mat_row_count / filled_row_block_size;
	int last_row_block_size =  mat_row_count % filled_row_block_size;

	// computing general column block sizes
	int filled_col_block_size = ceil((float)mat_col_count / (float)(instance->proc_rows));
	int filled_col_block_count = mat_col_count / filled_col_block_size;
	int last_col_block_size =  mat_col_count % filled_col_block_size;


	// computing process-wise block row / column counts
	for (int i=0; i < instance->proc_rows; i++)
	{
		for (int j=0; j < instance->proc_cols; j++)
		{
			if ((i==(instance->proc_rows-1)) && (last_row_block_size>0)) // handling last row block which is only partially filled
				matrix->proc_row_count[i][j] = last_row_block_size;
			else
				matrix->proc_row_count[i][j] = filled_row_block_size;
	
			if ((j==(instance->proc_cols-1)) && (last_col_block_size>0)) // handling last column block which is only partially filled
				matrix->proc_col_count[i][j] = last_col_block_size;
			else
				matrix->proc_col_count[i][j] = filled_col_block_size;
		}
	}

	// computing process-wise block row / column offsets
	matrix->proc_row_offset[0][0] = 0;
	matrix->proc_col_offset[0][0] = 0;
	for (int i=1; i < instance->proc_rows; i++)
		for (int j=1; j < instance->proc_cols; j++)
			matrix->proc_row_offset[i][j] = matrix->proc_row_offset[i-1][j] + matrix->proc_row_count[i-1][j];
	for (int j=1; j < instance->pro_cols; j++)
		for (int i=1; i < instance->proc_rows; i++)
			matrix->proc_col_offset[i][j] = matrix->proc_col_offset[i][j-1] + matrix->proc_col_count[i][j-1];
	
	// retrieving local data for the current process
	matrix->cur_proc_row_count = matrix->proc_row_count[instance->cur_proc_row][instance->cur_proc_col];
	matrix->cur_proc_col_count = matrix->proc_col_count[instance->cur_proc_row][instance->cur_proc_col];
	matrix->cur_proc_row_offset = matrix->proc_row_offset[instance->cur_proc_row][instance->cur_proc_col];
	matrix->cur_proc_col_offset = matrix->proc_col_offset[instance->cur_proc_row][instance->cur_proc_col];

	// allocating matrix data storage
	cudaMalloc((void**)&(matrix->data), sizeof(double)*matrix->cur_proc_row_count*matrix->cur_proc_col_count);
}


void mpla_dgemv(struct mpla_vector* b, struct mpla_matrix* A, struct mpla_vector* x)
{
	


}




