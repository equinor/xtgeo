#include "logger.h"

/* Logging levels: */

/* CRITICAL: 50 */
/* ERROR: 40 */
/* WARNING: 30 */
/* INFO: 20 */
/* DEBUG: 10 */
/* NOTSET: 0 */


int DEBUG = 1;

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
        printf("Start time: %ld\n", XTG_START_TIME);
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
 * But remade to milliseconds
 */

#define NANOS_PER_SECF 1000000000.0
#define USECS_PER_SEC 1000000

#if _POSIX_TIMERS > 0 && defined(_POSIX_MONOTONIC_CLOCK)
  // If we have it, use clock_gettime and CLOCK_MONOTONIC.

#include <time.h>

double monotonic_seconds() {
    struct timespec time;
    // Note: Make sure to link with -lrt to define clock_gettime.
    clock_gettime(CLOCK_MONOTONIC, &time);
    return ((double) time.tv_sec) + ((double) time.tv_nsec / (NANOS_PER_SECF));

}

#elif defined(__APPLE__)
// If we don't have CLOCK_MONOTONIC, we might be on a Mac. There we instead
// use mach_absolute_time().

#include <mach/mach_time.h>

static mach_timebase_info_data_t info;
static void __attribute__((constructor)) init_info() {
    mach_timebase_info(&info);
}

double monotonic_seconds() {
    uint64_t time = mach_absolute_time();
    double dtime = (double) time;
    dtime *= (double) info.numer;
    dtime /= (double) info.denom;
    return dtime / NANOS_PER_SECF;
}

#elif defined(_MSC_VER)
// On Windows, use QueryPerformanceCounter and QueryPerformanceFrequency.

#include <windows.h>

static double PCFreq = 0.0;

// According to http://stackoverflow.com/q/1113409/447288, this will
// make this function a constructor.
// TODO(awreece) Actually attempt to compile on windows.
static void __cdecl init_pcfreq();
__declspec(allocate(".CRT$XCU")) void (__cdecl*init_pcfreq_)() = init_pcfreq;
static void __cdecl init_pcfreq() {
    // Accoring to http://stackoverflow.com/a/1739265/447288, this will
    // properly initialize the QueryPerformanceCounter.
    LARGE_INTEGER li;
    int has_qpc = QueryPerformanceFrequency(&li);
    assert(has_qpc);

    PCFreq = (double) li.QuadPart;
}

double monotonic_seconds() {
    LARGE_INTEGER li;
    QueryPerformanceCounter(&li);
    return ((double) li.QuadPart) / PCFreq;
}

#else
// Fall back to rdtsc. The reason we don't use clock() is this scary message
// from the man page:
//     "On several other implementations, the value returned by clock() also
//      includes the times of any children whose status has been collected via
//      wait(2) (or another wait-type call)."
//
// Also, clock() only has microsecond accuracy.
//
// This whitepaper offered excellent advice on how to use rdtscp for
// profiling: http://download.intel.com/embedded/software/IA/324264.pdf
//
// Unfortunately, we can't follow its advice exactly with our semantics,
// so we're just going to use rdtscp with cpuid.
//
// Note that rdtscp will only be available on new processors.

#include <stdint.h>

static inline uint64_t rdtsc() {
    uint32_t hi, lo;
    asm volatile("rdtscp\n"
                 "movl %%edx, %0\n"
                 "movl %%eax, %1\n"
                 "cpuid"
                 : "=r" (hi), "=r" (lo) : : "%rax", "%rbx", "%rcx", "%rdx");
    return (((uint64_t)hi) << 32) | (uint64_t)lo;
}

static uint64_t rdtsc_per_sec = 0;
static void __attribute__((constructor)) init_rdtsc_per_sec() {
    uint64_t before, after;

    before = rdtsc();
    usleep(USECS_PER_SEC);
    after = rdtsc();

    rdtsc_per_sec = after - before;
}

double monotonic_seconds() {
    return (double) rdtsc() / (double) rdtsc_per_sec;
}

#endif
