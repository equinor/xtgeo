#ifndef LOGGER_H_
#define LOGGER_H_

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LI __LINE__
#define FI __FILE__
#define FU __FUNCTION__

#ifdef __cplusplus
extern "C"
{
#endif  // __cplusplus

    void logger_info(int line, char *file, const char *func, const char *fmt, ...);
    void logger_debug(int line, char *file, const char *func, const char *fmt, ...);
    void logger_warn(int line, char *file, const char *func, const char *fmt, ...);
    void logger_error(int line, char *file, const char *func, const char *fmt, ...);
    void logger_critical(int line, char *file, const char *func, const char *fmt, ...);

    /* Returns seconds since some unspecified start time (guaranteed to be */
    /* monotonicly increasing). */
    double monotonic_seconds();

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // LOGGER_H_
