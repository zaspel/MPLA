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

#include "cublas_v2.h"
#include <mpi.h>

#ifndef mpla_h__
#define mpla_h__


struct mpla_instance
{
	MPI_Comm comm;
	int proc_count;
	int proc_rows;
	int proc_cols;
	int cur_proc_coord[2];
	int cur_proc_row;
	int cur_proc_col;
	int cur_proc_rank;	
	bool is_parent;
	cublasHandle_t cublas_handle;
};


struct mpla_matrix
{
	double* data;
	int mat_row_count, mat_col_count;
	int cur_proc_row_count;
	int cur_proc_col_count;
	int cur_proc_row_offset;
	int cur_proc_col_offset;
	int** proc_row_count;
	int** proc_col_count;
	int** proc_row_offset;
	int** proc_col_offset;
	
};

struct mpla_vector
{
	double* data;
	int vec_row_count, vec_col_count;
	int cur_proc_row_count;
	int cur_proc_row_offset;
	int** proc_row_count;
	int** proc_row_offset;
};


extern void info();

extern void mpla_init_instance(struct mpla_instance* instance, MPI_Comm comm);

extern void mpla_init_matrix(struct mpla_matrix* matrix, struct mpla_instance* instance, int mat_row_count, int mat_col_count);

extern void mpla_init_vector(struct mpla_vector* vector, struct mpla_instance* instance, int vec_row_count);

extern void mpla_init_vector_for_block_rows(struct mpla_vector* vector, struct mpla_instance* instance, int vec_row_count);

extern void mpla_redistribute_vector_for_dgesv(struct mpla_vector* b_redist, struct mpla_vector* b, struct mpla_matrix* A, struct mpla_instance* instance);

extern void mpla_free_vector(struct mpla_vector* x, struct mpla_instance* instance);

extern void mpla_free_matrix(struct mpla_matrix* A, struct mpla_instance* instance);

extern void mpla_dgemv(struct mpla_vector* b, struct mpla_matrix* A, struct mpla_vector* x, struct mpla_instance* instance);

extern void mpla_ddot(double* xy, struct mpla_vector* x, struct mpla_vector* y, struct mpla_instance* instance);

#endif

