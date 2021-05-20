/* File : except.c */

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char error_message[256];
static int error_status = 0;

void
throw_exception(char *msg)
{
    strncpy(error_message, msg, 256);
    error_status = 1;
}

void
clear_exception()
{
    error_status = 0;
}

char *
check_exception()
{
    if (error_status)
        return error_message;
    else
        return NULL;
}
