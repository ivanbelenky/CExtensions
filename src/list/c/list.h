#ifndef LIST_H
#define LIST_H

#include <stdio.h>
#include <stdlib.h>
#include "types.h"

typedef struct node
{
	struct node *next;
	void *data;
} node_t, *list_t;


bool_t list_is_empty(const list_t *ptr); 
status_t list_create(list_t *pl); 
status_t list_create_node(node_t **node, void *data, size_t data_size);
void list_destroy_node(node_t **node, void (*destroy_node)(void *));
void list_destroy(list_t *pl, void (*destroy_node)(void *));
status_t list_insert_end(list_t *pl, void *data, size_t data_size);
void destroy_node_simple(void* node_data);


#endif














