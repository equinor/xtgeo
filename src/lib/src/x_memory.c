#include <stdbool.h>
#include <stdlib.h>

#include "common.h"
#include "logger.h"

/* allocate double 2D pointers to be contiguous in memory */

double **
x_allocate_2d_double(int n1, int n2)
{
    int i;

    double *data = malloc(sizeof(double) * n1 * n2);

    double **ptr_array = malloc(sizeof(double *) * n1);
    for (i = 0; i < n1; i++) {
        ptr_array[i] = data + (i * n2);
    }
    return ptr_array;
}

void
x_free_2d_double(double **ptr_array)
{
    if (!ptr_array)
        return;
    if (ptr_array[0])
        free(ptr_array[0]);
    free(ptr_array);
}

int **
x_allocate_2d_int(int n1, int n2)
{
    int i;

    int *data = malloc(sizeof(int) * n1 * n2);

    int **ptr_array = malloc(sizeof(int *) * n1);
    for (i = 0; i < n1; i++) {
        ptr_array[i] = data + (i * n2);
    }
    return ptr_array;
}

void
x_free_2d_int(int **ptr_array)
{
    if (!ptr_array)
        return;
    if (ptr_array[0])
        free(ptr_array[0]);
    free(ptr_array);
}

bool **
x_allocate_2d_bool(int n1, int n2)
{
    int i;

    bool *data = malloc(sizeof(bool) * n1 * n2);

    bool **ptr_array = malloc(sizeof(bool *) * n1);
    for (i = 0; i < n1; i++) {
        ptr_array[i] = data + (i * n2);
    }
    return ptr_array;
}

void
x_free_2d_bool(bool **ptr_array)
{
    if (!ptr_array)
        return;
    if (ptr_array[0])
        free(ptr_array[0]);
    free(ptr_array);
}
