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

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>

#include "PCR.h"

using namespace std;

#define TESTING

#ifdef TESTING

int main()
{
	cout << "Hello World!" << endl;
	cout << "This is PCR solver!" << endl;


	return 0;
}

#endif
