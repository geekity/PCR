/*
 * PCR.h
 *
 *  Created on: 4 Oct 2013
 *      Author: geekity
 *
 *  Parallel Cyclic Reduction Solver. This solver solves
 *  the equation
 *  	Ax = b
 *  where A is a tridiagonal matrix, b is supplied and x is unknown,
 *  using parallel cyclic reduction method. It mostly uses ideas from
 *
 *  Y. Zhang, J. Cohen, A. Davidson, J. Owens GPU Computing Gems Jade
 *  Edition (2011), Chapter 11
 *
 *  and
 *
 *  Zhangping Wei, Byunghyun Jang, Yaoxin Zhang, Yafei Jia, Procedia
 *  Computer Science 18 (2013) 389 - 398, Section 4.2.1, method 2(b).
 */

#ifndef PCR_H_
#define PCR_H_

class PCR {
private:
	int N;	/* system dimension */
	int S;
	float* A1;	/* below diagonal in tridiagonal system A */
	float* A2;	/* diagonal of tridiagonal system A */
	float* A3;	/* above diagonal in tridiagonal system A */
	float* b;	/* source vector */

	/* Allocates device memory */
	__host__ void PCR_init(float* A1_tmp, float* A2_tmp, float* A3_tmp,
		float* b_tmp);
	/* Copies reduced matrix A' to host memory A for testing purposes */
	__host__ void PCR_A_tester(float* A1_tmp, float* A2_tmp, float* A3_tmp,
		float* b_tmp);
public:
	/* Constructors */
	PCR(int N_tmp, int S_tmp);

	/* Destructor */
	virtual ~PCR();

	/* PCR solver method */
	__host__ void PCR_solve(float* A1_tmp, float* A2_tmp, float* A3_tmp,
		float* b_tmp, float* x_tmp);
};

/* Global solver function called from PCR Method PCR_solve(...) */
__global__ void PCR_solver(float* A1, float* A2, float* A3, float* B,
	int N);

/* Carries reduction on the system for a specified distance between equations (delta) */
__device__ void PCR_reduce(float* A1, float* A2, float* A3, float* B,
	int N, int chunks, int delta, int sys_offset);

/* Solves the 1 unknown system */
__device__ void PCR_solve_eqn(float* A2, float* B, int N, int chunks,
	int sys_offset);

#endif /* PCR_H_ */
