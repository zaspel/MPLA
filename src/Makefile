# Copyright (C) 2016 Peter Zaspel
#
# This file is part of MPLA.
#
# MPLA is free software: you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# MPLA is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with MPLA.  If not, see <http://www.gnu.org/licenses/>.


default: shared
	nvcc -g -I/home/zaspel/libs/openmpi/include -L/home/zaspel/libs/openmpi/lib -lcurand -lmpi_cxx -lmpi -L. -lmpla -lcublas -o benchmark_dgemv benchmark_dgemv.cu

shared:
	nvcc -g -I/home/zaspel/libs/openmpi/include -I/usr/local/cuda/include -L/home/zaspel/libs/openmpi/lib -lmpi_cxx -lmpi --ptxas-options=-v --compiler-options '-fPIC' -o libmpla.so --shared mpla.cu
