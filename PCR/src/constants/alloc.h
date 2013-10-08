/*
 * alloc.h
 *
 *  Created on: 28 Sep 2012
 *      Author: geekity
 *
 *  Checkers for error codes in dynamic memory allocation
 *  functions.
 */

#ifndef ALLOC_H_
#define ALLOC_H_

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <cuda.h>
using namespace std;

#define safe_calloc( num , size ) \
    calloc_check_return( (num) , (size) , __FILE__ , __LINE__ )

#define safe_malloc( size ) \
    malloc_check_return( (size) , __FILE__ , __LINE__ )

#define safe_realloc(pointer, size ) \
    realloc_check_return(pointer, (size) , __FILE__ , __LINE__ )

#define safe_memalign( size , align ) \
    memalign_check_return( (size) , (align) , __FILE__ , __LINE__ )

#define safe_free( pointer ) free( (pointer) )

#define check_return( err ) (HandleError( err, __FILE__, __LINE__ ))

void *calloc_check_return(size_t num,size_t size,char const*file
	,int line_no);

void *malloc_check_return(size_t size,char const*file,int line_no);

void *realloc_check_return(void *p, size_t size,char const*file,int line_no);

void *memalign_check_return(size_t size,size_t alignment,char const*file
	,int line_no);

static void HandleError( cudaError_t err, const char *file, int line )
{
	if (err != cudaSuccess)
	{
		cout << "Error: " << cudaGetErrorString( err ) << " in " << file << " at line "
					<< line << endl;
		exit( EXIT_FAILURE );
	}
}



#endif /* ALLOC_H_ */
