/*
 * alloc.cu
 *
 *  Created on: 28 Sep 2012
 *      Author: geekity
 *
 *  Checkers for error codes in dynamic memory allocation
 *  functions.
 */

#include <cerrno>
#include <cstdlib>
#include <iostream>
#include <cstring>
#include <fstream>
using namespace std;

#include "alloc.h"

void alloc_error(char const*file,int line_no,int err_code)
{
	(void)fprintf(stderr,"%s %s %d\n",strerror(err_code),file,line_no);
	exit(EXIT_FAILURE);
}

void *calloc_check_return(size_t num,size_t size,char const*file
	,int line_no)
{
	void *p= calloc(num,size);

	if(p==0)
	{
		alloc_error(file,line_no,ENOMEM);
	}
	return p;
}

void *malloc_check_return(size_t size,char const*file,int line_no)
{
	void*p= malloc(size);

	if(p==0)
	{
		alloc_error(file,line_no,ENOMEM);
	}
	return p;
}

void *realloc_check_return(void *p, size_t size,char const*file,int line_no)
{
	p = realloc(p, size);

	if(p==0)
	{
		alloc_error(file,line_no,ENOMEM);
	}
	return p;
}

void *memalign_check_return(size_t size,size_t alignment,char const*file
	,int line_no)
{
	void *p;
	int err= posix_memalign(&p,alignment,size);

	if(err!=0)
	{
		alloc_error(file,line_no,err);
	}
	return p;
}



