#include <stdio.h>

struct kernel_matrix_data
{
	double** points;
	int dim;
	int max_row_count_per_dgemv;
};


extern void mpla_set_gaussian_kernel_rhs(struct mpla_vector* b, struct mpla_generic_matrix* A, struct mpla_instance* instance);

extern void mpla_dgemv_core_kernel_cublas(struct mpla_vector* b, struct mpla_generic_matrix* A, struct mpla_vector* x_redist, struct mpla_instance* instance);
extern void mpla_dgemv_core_kernel_streamed_cublas(struct mpla_vector* b, struct mpla_generic_matrix* A, struct mpla_vector* x_redist, struct mpla_instance* instance);
