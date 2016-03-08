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



struct mpla_matrix
{
	double* data;
	int m,n;
	int* row_distr;
	int* row_offset;	
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

