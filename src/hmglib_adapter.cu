#include <curand.h>
#include <hmglib.h>

#include "hmglib_adapter.h"

void mpla_init_hmglib(struct mpla_generic_matrix* A, int global_point_count[2], double** all_coords[2], int dim, int bits, int c_leaf, int k, double epsilon, struct mpla_instance* instance)
{
	// get H matrix data structure
	struct h_matrix_data* data = (struct h_matrix_data*)A->data;

	int local_point_count[2];
	local_point_count[0] = A->cur_proc_row_count;
	local_point_count[1] = A->cur_proc_col_count;

        // initialize data structure
        init_h_matrix_data(data, local_point_count, dim, bits);

        // set balance  
        data->eta=1.0;

        // set maximum level
        data->max_level=50; // DEBUG

        // set maximum leaf size
        data->c_leaf=c_leaf;

        // set rank in ACA
        data->k = k;

        // set threshold for ACA (currently not use)
        data->epsilon = epsilon;

	// copy global coordinates to local coordinate lists
	for (int d=0; d<dim; d++)
	{
		cudaMemcpy(data->coords_d[0][d], &(all_coords[0][d][A->cur_proc_row_offset]), local_point_count[0]*sizeof(double), cudaMemcpyDeviceToDevice);
		cudaMemcpy(data->coords_d[1][d], &(all_coords[1][d][A->cur_proc_col_offset]), local_point_count[1]*sizeof(double), cudaMemcpyDeviceToDevice);
	}

        // run setup of H matrix
        setup_h_matrix(data);
}

void mpla_init_hmglib_coords_from_host(struct mpla_generic_matrix* A, int global_point_count[2], double** all_coords[2], int dim, int bits, int c_leaf, int k, double epsilon, struct mpla_instance* instance)
{
        // get H matrix data structure
        struct h_matrix_data* data = (struct h_matrix_data*)A->data;

        int local_point_count[2];
        local_point_count[0] = A->cur_proc_row_count;
        local_point_count[1] = A->cur_proc_col_count;

        // initialize data structure
        init_h_matrix_data(data, local_point_count, dim, bits);

        // set balance  
        data->eta=1.0;

        // set maximum level
        data->max_level=50; // DEBUG

        // set maximum leaf size
        data->c_leaf=c_leaf;

        // set rank in ACA
        data->k = k;

        // set threshold for ACA (currently not use)
        data->epsilon = epsilon;

        // copy global coordinates to local coordinate lists
        for (int d=0; d<dim; d++)
        {
                cudaMemcpy(data->coords_d[0][d], &(all_coords[0][d][A->cur_proc_row_offset]), local_point_count[0]*sizeof(double), cudaMemcpyHostToDevice);
                cudaMemcpy(data->coords_d[1][d], &(all_coords[1][d][A->cur_proc_col_offset]), local_point_count[1]*sizeof(double), cudaMemcpyHostToDevice);
        }

        // run setup of H matrix
        setup_h_matrix(data);
}



void mpla_destroy_hmglib(struct mpla_generic_matrix* A, struct mpla_instance* instance)
{
        // get H matrix data structure
        struct h_matrix_data* data = (struct h_matrix_data*)A->data;
        
	destroy_h_matrix_data(data);
}


void mpla_dgemv_core_hmglib_h_matrix(struct mpla_vector* b, struct mpla_generic_matrix* A, struct mpla_vector* x_redist, struct mpla_instance* instance)
{
        // get H matrix data structure
        struct h_matrix_data* data = (struct h_matrix_data*)A->data;

	apply_h_matrix_mvp(x_redist->data, b->data, data);
}


void mpla_dgemv_core_hmglib_full_matrix(struct mpla_vector* b, struct mpla_generic_matrix* A, struct mpla_vector* x_redist, struct mpla_instance* instance)
{
        // get H matrix data structure
        struct h_matrix_data* data = (struct h_matrix_data*)A->data;

	apply_full_mvp(x_redist->data, b->data, data);
}


