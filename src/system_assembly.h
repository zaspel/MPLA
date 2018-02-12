#ifndef system_assembly_h__
#define system_assembly_h__

#include "mpla.h"

class system_assembly
{
	public:
		int max_row_count_per_dgemv;
		virtual	__device__ double get_matrix_entry(int i, int j) =0;
		virtual __device__ double get_rhs_entry(int i) =0;
};


#endif


