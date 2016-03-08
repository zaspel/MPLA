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
	bool is_parent; 
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
	int m;
	int* row_distr;
	int* row_offset;	
};


extern void info();


#endif

