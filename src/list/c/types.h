#ifndef TYPES_H
#define TYPES_H

#include <stdio.h>
#include <stdlib.h>

typedef enum
{
	FALSE,
	TRUE
} bool_t;


typedef enum
{
	ST_OK,
	ST_NO_MEM,
	ST_ILLEGAL,
	ST_NULL_POINTER,
	ST_LIST_EMPTY,
} status_t;

#endif