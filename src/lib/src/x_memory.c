#include <stdbool.h>
#include <stdlib.h>
#include "common.h"

/* allocate double 2D pointers to be contiguous in memory */

double **
x_allocate_2d_double(int n1, int n2)
{
    double **ptr_array = calloc((size_t)n1, sizeof(double *));

    double *data = calloc((size_t)n1 * (size_t)n2, sizeof(double));
    for (int i = 0; i < n1; i++) {
        ptr_array[i] = data + ((size_t)i * (size_t)n2);
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

bool **
x_allocate_2d_bool(int n1, int n2)
{
    int i;

    bool *data = malloc(sizeof(bool) * (size_t)n1 * (size_t)n2);

    bool **ptr_array = malloc(sizeof(bool *) * (size_t)n1);
    for (i = 0; i < n1; i++) {
        ptr_array[i] = data + ((size_t)i * (size_t)n2);
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
