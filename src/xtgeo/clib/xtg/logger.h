#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>


#define LI __LINE__
#define FI __FILE__
#define FU __FUNCTION__

void logger_info(int line, char *file, const char *func, const char *fmt, ...);
void logger_debug(int line, char *file, const char *func, const char *fmt, ...);
void logger_warn(int line, char *file, const char *func, const char *fmt, ...);
void logger_error(int line, char *file, const char *func, const char *fmt, ...);
void logger_critical(int line, char *file, const char *func, const char *fmt, ...);


/* A cross platform monotonic timer. Copyright 2013 Alex Reece. */

#ifndef MONOTONIC_TIMER_H_
#define MONOTONIC_TIMER_H_

/* Returns seconds since some unspecified start time (guaranteed to be */
/* monotonicly increasing). */
double monotonic_seconds();

#endif  // MONOTONIC_TIMER_H_
