// Copyright (C) 2016 Peter Zaspel
//
// This file is part of MPLA.
//
// hmglib is free software: you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License as published by the Free
// Software Foundation, either version 3 of the License, or (at your option) any
// later version.
//
// hmglib is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
// A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
// details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with MPLA.  If not, see <http://www.gnu.org/licenses/>.

#include "mpla.h"
#include <stdio.h>

extern void mpla_init_hmglib(struct mpla_generic_matrix* A, int global_point_count[2], double** global_coords[2], int dim, int bits, int c_leaf, int k, double epsilon, struct mpla_instance* instance);

extern void mpla_init_hmglib_coords_from_host(struct mpla_generic_matrix* A, int global_point_count[2], double** all_coords[2], int dim, int bits, int c_leaf, int k, double epsilon, struct mpla_instance* instance);

extern void mpla_destroy_hmglib(struct mpla_generic_matrix* A, struct mpla_instance* instance);

extern void mpla_dgemv_core_hmglib_h_matrix(struct mpla_vector* b, struct mpla_generic_matrix* A, struct mpla_vector* x_redist, struct mpla_instance* instance);

extern void mpla_dgemv_core_hmglib_full_matrix(struct mpla_vector* b, struct mpla_generic_matrix* A, struct mpla_vector* x_redist, struct mpla_instance* instance);

