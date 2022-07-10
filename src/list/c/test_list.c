#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "types.h"
#include "list.h"


int main(int argc, char const *argv[])
{
    list_t list;
    list_create(&list);

    int i;
    
    printf("The list at the beginning is allocated to %x\n", &list);
    

    for (i = 0; i < 10; i++)
    {
        list_insert_end(&list, &i, sizeof(int));
    }
    printf("The list is allocated at %x\n", &list);
    list_destroy(&list, &destroy_node_simple);

    return 0;
}


