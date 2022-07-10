#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "types.h"
#include "list.h"


bool_t list_is_empty(const list_t *ptr){
	return (*ptr == NULL) ? TRUE : FALSE;
}

status_t list_create(list_t *pl){
	if(!pl){
		return ST_ILLEGAL;
	}

	*pl = NULL;
	return ST_OK;
}

status_t list_create_node(node_t **node, void *data, size_t data_size){
	if (!node){
		return ST_ILLEGAL;
	}
	if((*node = (node_t *)malloc(sizeof(node_t))) == NULL ){
		return ST_NO_MEM;
	}
	(*node)->next = NULL;
	
	if((((*node)->data) = (void*)malloc(data_size)) == NULL){
		return ST_NO_MEM;
	}
	(*node)->data = memcpy((*node)->data, data, data_size);
	return ST_OK;
}

void list_destroy_node(node_t **node, void (*destroy_node)(void*)){
	if(list_is_empty (node)){
		return;
	}

	if (destroy_node != NULL){
		(*destroy_node)((*node)->data);
	}

	(*node)->data = NULL;
	(*node)->next = NULL;
	free(*node);
	*node = NULL;
}

void destroy_node_simple(void* node_data){
	free(node_data);
}

void list_destroy(list_t *pl, void (*destroy_node)(void*)){
	node_t *aux;
	while(*pl){
		aux = *pl;
		*pl = (*pl)->next;
		list_destroy_node(&aux, destroy_node);
	}
}


status_t list_insert_end(list_t *pl, void *data, size_t data_size){
	node_t *aux, *last;
	status_t st;
	if (!pl){
		return ST_ILLEGAL;
	}
	if((st = list_create_node(&aux, data, data_size)) != ST_OK){
		return st;
	}

	if (list_is_empty(pl)){
		*pl = aux;
	}
	else{
		last = *pl;
		while(last->next){
			last = last->next;
		}	
		last->next = aux;
	}
	return ST_OK;
}


