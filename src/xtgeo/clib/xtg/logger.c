#include "logger.h"

/* Logging levels: */

/* CRITICAL: 50 */
/* ERROR: 40 */
/* WARNING: 30 */
/* INFO: 20 */
/* DEBUG: 10 */
/* NOTSET: 0 */


int DEBUG = 0;

static int XTG_LOGGING_SET = 0;

static int XTG_LOGGING_LEVEL = 30;
static int XTG_LOGGING_FORMAT = 1;
static double XTG_START_TIME = 0.0;


const char* _basename(const char *path)
{
    const char *name = NULL, *tmp = NULL;
    if (path && *path) {
        name = strrchr(path, '/');
        tmp = strrchr(path, '\\');
        if (tmp) {
             return name && name > tmp ? name + 1 : tmp + 1;
        }
    }
    return name ? name + 1 : path;
}

int _logger_init()
{
    /* This function shall only be ran once */
    char *llevel;
    char *lfmt;
    int llev = 30;

    if (XTG_LOGGING_SET == 1) return 0;

    XTG_LOGGING_SET = 1;
    XTG_START_TIME = monotonic_seconds();

    llevel = getenv("XTG_LOGGING_LEVEL");

    if (llevel == NULL) {
        return -1;
    }

    if (strcmp(llevel, "INFO") == 0) llev = 20;
    if (strcmp(llevel, "DEBUG") == 0) llev = 10;
    if (strcmp(llevel, "WARN") == 0) llev = 30;
    if (strcmp(llevel, "WARNING") == 0) llev = 30;
    if (strcmp(llevel, "ERROR") == 0) llev = 40;
    if (strcmp(llevel, "CRITICAL") == 0) llev = 50;

    XTG_LOGGING_LEVEL = llev;

    lfmt = getenv("XTG_LOGGING_FORMAT");
    if (lfmt != NULL) {
        if (strncmp(lfmt, "1", 1) == 0) XTG_LOGGING_FORMAT = 1;
        if (strncmp(lfmt, "2", 1) == 0) XTG_LOGGING_FORMAT = 2;
    }

    if (DEBUG == 1) {
        printf("Logging details:\n");
        printf("Logging level: %d\n", XTG_LOGGING_LEVEL);
        printf("Logging format: %d\n", XTG_LOGGING_FORMAT);
        printf("Start time: %lf\n", XTG_START_TIME);
    }
    return -1;
}


void _logger_tell(int line, const char *filename, const char *func, const char *message,
                   const char *ltype, int level)
{

    double dtime = monotonic_seconds() - XTG_START_TIME;

    if (level >= XTG_LOGGING_LEVEL) {
        if (XTG_LOGGING_FORMAT == 1) {
            printf("%8s: (%3.2fs) \t%s\n", ltype, dtime, message);
        }
        else if (XTG_LOGGING_FORMAT == 2) {
            printf("%8s (%3.2lfs) %44s [%42s] %4d >> \t%s\n", ltype, dtime, filename,
                   func, line, message);
        }

    }
    XTG_START_TIME = monotonic_seconds();
}

void logger_debug(int line, char *file, const char *func, const char *fmt, ...)
{

    if (_logger_init() == -1) return;

    va_list ap;
    char message[550], msg[547];

    const char *filename = _basename(file);

    va_start(ap, fmt);
    vsprintf(msg, fmt, ap);
    va_end(ap);
    sprintf(message, "%s", msg);
    _logger_tell(line, filename, func, message, "DEBUG", 10);
}

void logger_info(int line, char *file, const char *func, const char *fmt, ...)
{

    if (_logger_init() == -1) return;

    va_list ap;
    char message[550], msg[547];

    const char *filename = _basename(file);

    va_start(ap, fmt);
    vsprintf(msg, fmt, ap);
    va_end(ap);
    sprintf(message, "%s", msg);
    _logger_tell(line, filename, func, message, "INFO", 20);
}

void logger_warn(int line, char *file, const char *func, const char *fmt, ...)
{

    if (_logger_init() == -1) return;

    va_list ap;
    char message[550], msg[547];

    const char *filename = _basename(file);

    va_start(ap, fmt);
    vsprintf(msg, fmt, ap);
    va_end(ap);
    sprintf(message, "%s", msg);
    _logger_tell(line, filename, func, message, "WARNING", 30);
}

void logger_error(int line, char *file, const char *func, const char *fmt, ...)
{

    if (_logger_init() == -1) return;

    va_list ap;
    char message[550], msg[547];

    const char *filename = _basename(file);

    va_start(ap, fmt);
    vsprintf(msg, fmt, ap);
    va_end(ap);
    sprintf(message, "%s", msg);
    _logger_tell(line, filename, func, message, "ERROR", 40);
}

void logger_critical(int line, char *file, const char *func, const char *fmt, ...)
{

    if (_logger_init() == -1) return;

    va_list ap;
    char message[550], msg[547];

    const char *filename = _basename(file);

    va_start(ap, fmt);
    vsprintf(msg, fmt, ap);
    va_end(ap);
    sprintf(message, "%s", msg);
    _logger_tell(line, filename, func, message, "CRTITICAL", 50);
    exit(666);
}

/*
 *====================================================================================
 * Based on:
 * https://github.com/awreece/monotonic_timer/blob/master/monotonic_timer.c
 * Copyright 2013 Alex Reece. A cross platform monotonic timer.
 * But slightly adapted
 */

#define NANOS_PER_SECF 1000000000.0
#define USECS_PER_SEC 1000000

#if defined(_WIN32)

/* On Windows, use QueryPerformanceCounter and QueryPerformanceFrequency. */
#include <windows.h>

#define BILLION (1E9)

static BOOL g_first_time = 1;
static LARGE_INTEGER g_counts_per_sec;

struct timespec { long tv_sec; long tv_nsec; };

int _clock_gettime(int dummy, struct timespec *ct)
{
    LARGE_INTEGER count;

    if (g_first_time)
    {
        g_first_time = 0;

        if (0 == QueryPerformanceFrequency(&g_counts_per_sec))
        {
            g_counts_per_sec.QuadPart = 0;
        }
    }

    if ((NULL == ct) || (g_counts_per_sec.QuadPart <= 0) ||
            (0 == QueryPerformanceCounter(&count)))
    {
        return -1;
    }

    ct->tv_sec = count.QuadPart / g_counts_per_sec.QuadPart;
    ct->tv_nsec = ((count.QuadPart % g_counts_per_sec.QuadPart) * BILLION) /
        g_counts_per_sec.QuadPart;

    return 0;
}
double monotonic_seconds() {
    struct timespec time;
    _clock_gettime(0, &time);
    return ((double) time.tv_sec) + ((double) time.tv_nsec / (NANOS_PER_SECF));
}

#else

#include <time.h>

double monotonic_seconds() {
    struct timespec time;
    // Note: Make sure to link with -lrt to define clock_gettime.
    clock_gettime(CLOCK_MONOTONIC, &time);
    return ((double) time.tv_sec) + ((double) time.tv_nsec / (NANOS_PER_SECF));
}


#endif
