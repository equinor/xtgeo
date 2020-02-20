#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <libgen.h>
#include <sys/time.h>
#include <time.h>


#define LI __LINE__
#define FI __FILE__
#define FU __FUNCTION__

void logger_info(int line, char *file, const char *func, const char *fmt, ...);
void logger_debug(int line, char *file, const char *func, const char *fmt, ...);
void logger_warn(int line, char *file, const char *func, const char *fmt, ...);
void logger_error(int line, char *file, const char *func, const char *fmt, ...);
void logger_critical(int line, char *file, const char *func, const char *fmt, ...);
