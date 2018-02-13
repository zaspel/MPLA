#ifndef generic_system_adapter_h__
#define generic_system_adapter_h__


#include <stdio.h>
#include "mpla.h"
#include <curand.h>

#include "system_assembler.h"


extern void mpla_set_generic_system_rhs(struct mpla_vector* b, struct mpla_generic_matrix* A, struct mpla_instance* instance);
extern void mpla_dgemv_core_generic_system_matrix_cublas(struct mpla_vector* b, struct mpla_generic_matrix* A, struct mpla_vector* x_redist, struct mpla_instance* instance);
extern void mpla_dgemv_core_generic_system_matrix_streamed_cublas(struct mpla_vector* b, struct mpla_generic_matrix* A, struct mpla_vector* x_redist, struct mpla_instance* instance);

#endif
