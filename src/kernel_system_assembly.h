#ifndef kernel_system_assembly_h__
#define kernel_system_assembly_h__
        


#include "mpla.h"
#include "system_assembly.h"


class gaussian_kernel_system_assembly : public system_assembly
{
	public:

		double** points;
		int dim;
		double regularization;


		__device__ double gaussian_kernel(double r)
		{
			return exp(-r*r);
		}

		__device__ double get_matrix_entry(int i, int j)
		{
			double result = 0.0;
			for (int d=0; d<dim; d++)
			{
				result += (points[d][i]-points[d][j])*(points[d][i]-points[d][j]);
			}
			result =  sqrt(result);

			if (i==j)
				return gaussian_kernel(result) + regularization;
			else
				return gaussian_kernel(result);

		}

		__device__ double get_rhs_entry(int i)
		{
			double result = 1.0;

			for (int d=0; d<dim; d++)
				result = result * (sqrt(M_PI)/2.0) * (erf(1.0- points[d][i]) - erf(0.0 - points[d][i]));

			return result;
		}

};

 extern void create_gaussian_kernel_system_assembly_object(struct gaussian_kernel_system_assembly** assem, double** points, int max_row_count_per_dgemv, int dim);

extern void destroy_gaussian_kernel_system_assembly_object(struct gaussian_kernel_system_assembly** assem);

#endif
