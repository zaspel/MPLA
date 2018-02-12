#ifndef generic_system_adapter_h__
#define generic_system_adapter_h__


#include <stdio.h>
#include "mpla.h"
#include <curand.h>

#include "system_assembly.h"


extern void mpla_set_generic_system_rhs(struct mpla_vector* b, struct mpla_generic_matrix* A, struct mpla_instance* instance);
extern void mpla_dgemv_core_generic_system_matrix_cublas(struct mpla_vector* b, struct mpla_generic_matrix* A, struct mpla_vector* x_redist, struct mpla_instance* instance);
extern void mpla_dgemv_core_generic_system_matrix_streamed_cublas(struct mpla_vector* b, struct mpla_generic_matrix* A, struct mpla_vector* x_redist, struct mpla_instance* instance);
//extern void create_gaussian_kernel_system_assembly_object(struct gaussian_kernel_system_assembly** assem, double** points, int max_row_count_per_dgemv, int dim);
//extern void destroy_gaussian_kernel_system_assembly_object(struct gaussian_kernel_system_assembly** assem);

#endif
