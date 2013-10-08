/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <iostream>
#include <cstdlib>
#include <cmath>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>

#include "PCR.h"
#include "constants/alloc.h"
#include "constants/cutil_math.h"

using namespace std;

#define CHUNK_MAX 256
#define CR_BUFF_MAX 128 // set statically since my current card doesn't support dynamic memory
						  // allocation

#define TESTING

/* Constructor */
PCR::PCR(int N_tmp) {
	if (N_tmp > CHUNK_MAX*CR_BUFF_MAX) {
		cout << "Error: system dimension exceeds allowance of " << CHUNK_MAX*CR_BUFF_MAX;
		cout << " equations!" << endl;
		exit(1);
	}

	N = N_tmp;
	check_return(cudaMalloc((float**) &A1, N*sizeof(float)));
	check_return(cudaMalloc((float**) &A2, N*sizeof(float)));
	check_return(cudaMalloc((float**) &A3, N*sizeof(float)));
	check_return(cudaMalloc((float**) &b, N*sizeof(float)));
}

/* Destructor */
PCR::~PCR() {
	check_return(cudaFree(A1));
	check_return(cudaFree(A2));
	check_return(cudaFree(A3));
	check_return(cudaFree(b));
}

/* Public Methods */

/* PCR solver method */
__host__ void PCR::PCR_solve(float* A1_tmp, float* A2_tmp, float* A3_tmp,
	float* b_tmp, float* x_tmp) {

	PCR_init(A1_tmp, A2_tmp, A3_tmp, b_tmp);

	/* Launch solver here */

	PCR_solver<<<1, CHUNK_MAX>>>(A1, A2, A3, b, N);

	check_return(cudaMemcpy(x_tmp, b, N*sizeof(float), cudaMemcpyDeviceToHost));
}

/* Private Methods */

/* Allocates device memory */
__host__ void PCR::PCR_init(float* A1_tmp, float* A2_tmp, float* A3_tmp,
		float* b_tmp) {
	check_return(cudaMemcpy(A1, A1_tmp, N*sizeof(float), cudaMemcpyHostToDevice));
	check_return(cudaMemcpy(A2, A2_tmp, N*sizeof(float), cudaMemcpyHostToDevice));
	check_return(cudaMemcpy(A3, A3_tmp, N*sizeof(float), cudaMemcpyHostToDevice));
	check_return(cudaMemcpy(b, b_tmp, N*sizeof(float), cudaMemcpyHostToDevice));
}

/* Copies reduced matrix A' to host memory A for testing purposes */
__host__ void PCR::PCR_A_tester(float* A1_tmp, float* A2_tmp, float* A3_tmp,
	float* b_tmp) {
	check_return(cudaMemcpy(A1_tmp, A1, N*sizeof(float), cudaMemcpyDeviceToHost));
	check_return(cudaMemcpy(A2_tmp, A2, N*sizeof(float), cudaMemcpyDeviceToHost));
	check_return(cudaMemcpy(A3_tmp, A3, N*sizeof(float), cudaMemcpyDeviceToHost));
	check_return(cudaMemcpy(b_tmp, b, N*sizeof(float), cudaMemcpyDeviceToHost));
}

/* Global functions */

/* Global solver function called from PCR Method PCR_solve(...) */
__global__ void PCR_solver(float* A1, float* A2, float* A3, float* B,
	int N) {

	int chunks = (N + CHUNK_MAX - 1)/CHUNK_MAX;
	int delta = 1;
	int sys_offset = blockIdx.x*N;

	int cycles = ceil(log2f(N));

	for (int i = 0; i < cycles; i++) {
		PCR_reduce(A1, A2, A3, B, N, chunks, delta, sys_offset);
		delta *= 2;
		__syncthreads();
	}

	PCR_solve_eqn(A2, B, N, chunks, sys_offset);
}

/* Device functions */

/* Carries reduction on the system for a specified distance between equations (delta) */
__device__ void PCR_reduce(float* A1, float* A2, float* A3, float* B,
	int N, int chunks, int delta, int sys_offset) {

	float4 eqn1[CR_BUFF_MAX];

	/* Fetch top line values that will get overwritten on chunk boundaries */
	for (int i = 0; i < chunks; i++) {
		int eqn_num = i*CHUNK_MAX + threadIdx.x;
		if (eqn_num < N) {
			int id = sys_offset + eqn_num;
			eqn1[i] = (eqn_num-delta >= 0) ?
				make_float4(A1[id-delta], A2[id-delta], A3[id-delta], B[id-delta]) :
				make_float4(0.0);
		}
	}

	/* Reduce */
	for (int i = 0; i < chunks; i++) {
		int eqn_num = i*CHUNK_MAX + threadIdx.x;
		if (eqn_num < N) {
			int id = sys_offset + eqn_num;

			float4 eqn2 = make_float4(A1[id], A2[id], A3[id], B[id]);

			float4 eqn3 = (eqn_num+delta < N) ?
				make_float4(A1[id+delta], A2[id+delta], A3[id+delta], B[id+delta]) :
				make_float4(0.0);

			__syncthreads();

			float l2 = eqn2.x; float Lam1 = (eqn_num-delta >= 0) ? eqn1[i].y : 1.0;
			float m2 = eqn2.z; float Lam3 = (eqn_num+delta < N) ? eqn3.y : 1.0;

			eqn1[i] *= (l2*Lam3);
			eqn2 *= (-Lam1*Lam3);
			eqn3 *= (Lam1*m2);

			eqn2.y += eqn1[i].z + eqn3.x;
			eqn2.x = eqn1[i].x;
			eqn2.z = eqn3.z;
			eqn2.w += eqn1[i].w + eqn3.w;
			A1[id] = eqn2.x;
			A2[id] = eqn2.y;
			A3[id] = eqn2.z;
			B[id] = eqn2.w;
		}
	}
}

/* Solves the 1 unknown system */
__device__ void PCR_solve_eqn(float* A2, float* B, int N, int chunks,
	int sys_offset) {
	for (int i = 0; i < chunks; i++) {
		int eqn_num = i*CHUNK_MAX + threadIdx.x;
		if (eqn_num < N) {
			int id = sys_offset + eqn_num;
			B[id] /= A2[id];
		}
	}
}

#ifdef TESTING

int main()
{
	cout << "Hello World!" << endl;
	cout << "This is PCR solver!" << endl;

	int N = 5;

	float A1 [] = {0.0, 3.0, 2.0, 1.0, 6.0};
	float A2 [] = {1.0, 4.0, -1.0, -2.0, 11};
	float A3 [] = {7.0, 1.0, 7.0, 5.0, 0.0};
	float B [] = {1.0, 4.0, -1.0, 7.0, 18.0};
	float X [] = {0.0, 0.0, 0.0, 0.0, 0.0};

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			if ((i != 0) && (j == i-1)) cout << A1[i] << " ";
			else if (j == i) cout << A2[i] << " ";
			else if ((i != N-1) && (j == i+1)) cout << A3[i] << " ";
			else cout << "0 ";
		}
		cout << endl;
	}
	cout << endl;

	PCR* pcr= new PCR(N);

	pcr->PCR_solve(A1, A2, A3, B, X);

	for (int i = 0; i < N; i++) {
		cout << X[i] << " ";
	}
	cout << endl;

	delete pcr;

	return 0;
}

#endif
