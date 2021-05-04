/* File : except.c */

#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

static char error_message[256];
static int error_status = 0;

void throw_exception( const char *fmt, ...) {
    char message[550], msg[547];
    va_list ap;
    va_start(ap, fmt);
    vsprintf(msg, fmt, ap);
    va_end(ap);
    sprintf(message, "%s", msg);
	strncpy(error_message,message,256);
	error_status = 1;
}

void clear_exception() {
	error_status = 0;
}

char *check_exception() {
	if (error_status) return error_message;
	else return NULL;
}

