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

# default: test
#
#benchmark: shared
#	g++ -c -I/usr/local/cuda/include -I/home/zaspel/libs/openmpi/include -o benchmark_dgemv.o benchmark_dgemv.cpp
#	g++ -L/usr/local/cuda/lib64 -L/home/zaspel/libs/openmpi/lib -L. -o benchmark_dgemv benchmark_dgemv.o -lcudart -lcurand -lmpi_cxx -lmpi -lmpla -lcublas
#	g++ -c -I/usr/local/cuda/include -I/home/zaspel/libs/openmpi/include -o benchmark_cg.o benchmark_cg.cpp
#	g++ -L/usr/local/cuda/lib64 -L/home/zaspel/libs/openmpi/lib -L. -o benchmark_cg benchmark_cg.o -lcudart -lcurand -lmpi_cxx -lmpi -lmpla -lcublas
#	$(NVCC) -O3 -c -arch=sm_30 -I/home/zaspel/libs/openmpi/include kernel.cu -o kernel.o
#	g++ -c -I/usr/local/cuda/include -I/home/zaspel/libs/openmpi/include -o benchmark_kernel_cg.o benchmark_kernel_cg.cpp
#	g++ -L/usr/local/cuda/lib64 -L/home/zaspel/libs/openmpi/lib -L. -o benchmark_kernel_cg benchmark_kernel_cg.o kernel.o -lcudart -lcurand -lmpi_cxx -lmpi -lmpla -lcublas
#
#
# shared:
#	g++ -I/usr/local/cuda/include -I/home/zaspel/libs/openmpi/include -pthread  -L/usr/local/cuda/lib64 -lcudart -fPIC -o libmpla.so --shared mpla.cpp
#
#titan:
# 	$(CC) -g -lcudart -fPIC -o libmpla.so --shared mpla.cpp
##	$(CC) -lcurand -L. -lmpla -lcublas -o benchmark_dgemv benchmark_dgemv.cpp
##	$(CC) -lcurand -L. -lmpla -lcublas -o benchmark_cg benchmark_cg.cpp
##	$(NVCC) -g -O3 -c -arch=sm_35 -I$(MPICH_DIR)/include kernel.cu -o kernel.o
##	$(CC) -lcurand -L. -lmpla -lcublas kernel.o -o benchmark_kernel_cg benchmark_kernel_cg.cpp
##	$(CC) -g -lcurand -L. -lmpla -lcublas ${GSL_LIB} kernel.o -o generate_quadrature_weights generate_quadrature_weights.cpp
#	$(NVCC) -O3 -c -arch=sm_35 -I$(MPICH_DIR)/include hmglib_adapter.cu -I/lustre/atlas/proj-shared/csc238/hmglib/src -o hmglib_adapter.o
#	$(NVCC) -O3 -arch=sm_35 -lcurand -L. -L/lustre/atlas/proj-shared/csc238/hmglib/src -lmpla -lhmglib -lcublas ${GSL_LIB} -I$(MPICH_DIR)/include -I/lustre/atlas/proj-shared/csc238/hmglib/src hmglib_adapter.o -o h_matrix_test h_matrix_test.cu
#
#clean:
#	rm *.o *.so test_library benchmark_dgemv
#
#test: shared
#	g++ -c -I/usr/local/cuda/include -I/home/zaspel/libs/openmpi/include -o test_library.o test_library.cpp
#	g++ -L/usr/local/cuda/lib64 -L/home/zaspel/libs/openmpi/lib -L. -o test_library test_library.o -lcudart -lcurand -lmpi_cxx -lmpi -lmpla -lcublas
#	g++ -c -I/usr/local/cuda/include -I/home/zaspel/libs/openmpi/include -o test_cg.o test_cg.cpp
#	g++ -L/usr/local/cuda/lib64 -L/home/zaspel/libs/openmpi/lib -L. -o test_cg test_cg.o -lcudart -lcurand -lmpi_cxx -lmpi -lmpla -lcublas
#	g++ -c -I/usr/local/cuda/include -I/home/zaspel/libs/openmpi/include -o test_generic_cg.o test_generic_cg.cpp
#	g++ -L/usr/local/cuda/lib64 -L/home/zaspel/libs/openmpi/lib -L. -o test_generic_cg test_generic_cg.o -lcudart -lcurand -lmpi_cxx -lmpi -lmpla -lcublas
#	$(NVCC) -c -arch=sm_30 -I/home/zaspel/libs/openmpi/include kernel.cu -o kernel.o
#	g++ -c -I/usr/local/cuda/include -I/home/zaspel/libs/openmpi/include -o test_kernel_cg.o test_kernel_cg.cpp
#	g++ -L/usr/local/cuda/lib64 -L/home/zaspel/libs/openmpi/lib -L. -o test_kernel_cg test_kernel_cg.o kernel.o -lcudart -lcurand -lmpi_cxx -lmpi -lmpla -lcublas

CUDA_DIR = /usr/local/cuda

NVCC = $(CUDA_DIR)/bin/nvcc
CC = /users/staff/dmi-dmi/zaspel/opt/openmpi/bin/mpiCC

GSL_LIB_DIR = /users/staff/dmi-dmi/zaspel/opt/gsl
HMGLIB_DIR = /users/staff/dmi-dmi/zaspel/hmglib/src

GPU_ARCH_SETTINGS = -arch=sm_60

MPI_CFLAGS = -I/users/staff/dmi-dmi/zaspel/opt/openmpi/include
MPI_LIBS = -L/users/staff/dmi-dmi/zaspel/opt/openmpi/lib -lmpi


NVCC_CXXFLAGS += -O3 $(GPU_ARCH_SETTINGS)
NVCC_CXXFLAGS += $(MPI_CFLAGS) -I$(GSL_LIB_DIR)/include
CXXFLAGS += -O3 -I$(CUDA_DIR)/include -L$(CUDA_DIR)/lib64

NVCC_LDFLAGS += $(MPI_LIBS)
NVCC_LDFLAGS += -lmpi

NVCC_CXXFLAGS_with_hmglib = $(NVCC_CXXFLAGS) -I$(HMGLIB_DIR) -I/users/staff/dmi-dmi/zaspel/opt/magma/include -DADD_

LDFLAGS_benchmarks += -L. -lmpla -lcublas -L$(GSL_LIB_DIR)/lib -lgslcblas -lgsl -lmpi -lcudart -lcurand 
LDFLAGS_benchmarks_with_hmglib += $(LDFLAGS_benchmarks) -L$(HMGLIB_DIR) -L/users/staff/dmi-dmi/zaspel/opt/magma/lib -lhmglib $(MPI_LIBS)

all: generate_quadrature_weights benchmark_generic_system_cg benchmark_kernel_cg benchmark_cg

hmglib: h_matrix_test h_matrix_test_mrhs h_matrix_test_2d_parallel

libmpla.so: mpla.cpp mpla.h
	$(CC) $(CXXFLAGS) -lcudart -fPIC -o libmpla.so -shared mpla.cpp

hmglib_adapter.o: hmglib_adapter.h hmglib_adapter.cu
	$(NVCC) $(NVCC_CXXFLAGS_with_hmglib) -c hmglib_adapter.cu -o hmglib_adapter.o

kernel.o: kernel.h kernel.cu libmpla.so
	$(NVCC) $(NVCC_CXXFLAGS) -c kernel.cu -o kernel.o

generic_system_adapter.o: generic_system_adapter.h generic_system_adapter.cu libmpla.so
	$(NVCC) $(NVCC_CXXFLAGS) -dc generic_system_adapter.cu -o generic_system_adapter.o

kernel_system_assembler.o: kernel_system_assembler.h kernel_system_assembler.cu generic_system_adapter.o libmpla.so
	$(NVCC) $(NVCC_CXXFLAGS) -dc kernel_system_assembler.cu -o kernel_system_assembler.o

benchmark_cg: benchmark_cg.cpp kernel.o libmpla.so
	$(CC) kernel.o -o benchmark_cg benchmark_cg.cpp $(CXXFLAGS) $(LDFLAGS_benchmarks) 

benchmark_kernel_cg: benchmark_kernel_cg.cpp kernel.o libmpla.so
	$(CC) kernel.o -o benchmark_kernel_cg benchmark_kernel_cg.cpp $(CXXFLAGS) $(LDFLAGS_benchmarks) 

generic_system_adapter_kernel_system_assembler_device.o: generic_system_adapter.o kernel_system_assembler.o 
	$(NVCC) $(NVCC_CXXFLAGS) --device-link generic_system_adapter.o kernel_system_assembler.o -o generic_system_adapter_kernel_system_assembler_device.o

benchmark_generic_system_cg.o: benchmark_generic_system_cg.cpp
	$(CC) -c benchmark_generic_system_cg.cpp -o benchmark_generic_system_cg.o $(CXXFLAGS) $(LDFLAGS_benchmarks)

benchmark_generic_system_cg: benchmark_generic_system_cg.o kernel_system_assembler.o generic_system_adapter.o generic_system_adapter_kernel_system_assembler_device.o libmpla.so
	$(CC) kernel_system_assembler.o generic_system_adapter.o generic_system_adapter_kernel_system_assembler_device.o benchmark_generic_system_cg.o -o benchmark_generic_system_cg $(CXXFLAGS) $(LDFLAGS_benchmarks) 
clean:
	rm *.o *.so


generate_quadrature_weights: generate_quadrature_weights.cu kernel.o libmpla.so
	$(NVCC) $(NVCC_CXXFLAGS_with_hmglib) $(LDFLAGS_benchmarks_with_hmglib) kernel.o -o generate_quadrature_weights generate_quadrature_weights.cu

h_matrix_test: h_matrix_test.cu hmglib_adapter.o libmpla.so
	$(NVCC) $(NVCC_CXXFLAGS_with_hmglib) $(LDFLAGS_benchmarks_with_hmglib) hmglib_adapter.o -o h_matrix_test h_matrix_test.cu 

h_matrix_test_mrhs: h_matrix_test_mrhs.cu hmglib_adapter.o libmpla.so
	$(NVCC) $(NVCC_CXXFLAGS_with_hmglib) $(LDFLAGS_benchmarks_with_hmglib)  hmglib_adapter.o -o h_matrix_test_mrhs h_matrix_test_mrhs.cu




h_matrix_test_2d_parallel: h_matrix_test_2d_parallel.cu hmglib_adapter.o libmpla.so
	$(NVCC) $(NVCC_CXXFLAGS_with_hmglib) $(LDFLAGS_benchmarks_with_hmglib) hmglib_adapter.o -o h_matrix_test_2d_parallel h_matrix_test_2d_parallel.cu



