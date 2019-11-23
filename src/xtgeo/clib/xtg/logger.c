#include "logger.h"

/* Logging levels: */

/* CRITICAL: 50 */
/* ERROR: 40 */
/* WARNING: 30 */
/* INFO: 20 */
/* DEBUG: 10 */
/* NOTSET: 0 */


int DEBUG = 0;

static int XTG_LOGGING_LEVEL = 30;
static int XTG_LOGGING_FORMAT = 1;
static char XTG_FUNCTION[33] = "unknown_function";
static char XTG_FILE[51] = "unknown_file";


const char* _basefilename(const char *path)
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


/* typically: logger_init(__FILE__, __FUNCTION__) */
void logger_init(const char *filename, const char *funcname)

{
    char *llevel;
    char *lfmt;
    int llev = 30;

    llevel = getenv("XTG_LOGGING_LEVEL");

    if (llevel != NULL) {
        if (strcmp(llevel, "INFO") == 0) llev = 20;
        if (strcmp(llevel, "DEBUG") == 0) llev = 10;
        if (strcmp(llevel, "WARN") == 0) llev = 30;
        if (strcmp(llevel, "WARNING") == 0) llev = 30;
        if (strcmp(llevel, "ERROR") == 0) llev = 40;
        if (strcmp(llevel, "CRITICAL") == 0) llev = 50;
    }

    XTG_LOGGING_LEVEL = llev;

    strncpy(XTG_FUNCTION, funcname, 32);
    strncpy(XTG_FILE, _basefilename(filename), 50);

    lfmt = getenv("XTG_LOGGING_FORMAT");
    if (lfmt != NULL) {
        if (strncmp(lfmt, "1", 1) == 0) XTG_LOGGING_FORMAT = 1;
        if (strncmp(lfmt, "2", 1) == 0) XTG_LOGGING_FORMAT = 2;
    }

    if (DEBUG == 1) {
        printf("Logging details:\n");
        printf("Logging level: %d\n", XTG_LOGGING_LEVEL);
        printf("Logging format: %d\n", XTG_LOGGING_FORMAT);
        printf("Logging function: %s\n", XTG_FUNCTION);
        printf("Logging file: %s\n", XTG_FILE);

    }
}


void _logger_tell(int line, const char *message, const char *ltype, int level)
{

    if (level >= XTG_LOGGING_LEVEL) {
        if (XTG_LOGGING_FORMAT == 1) {
            printf("%8s: (*****) \t%s\n", ltype, message);
        }
        else if (XTG_LOGGING_FORMAT == 2) {
            printf("%8s (*****) %44s [%42s] %4d >> \t%s\n", ltype, XTG_FILE,
                   XTG_FUNCTION, line, message);
        }

    }
}

void logger_debug(int line, const char *fmt, ...)
{
    va_list ap;
    char message[550], msg[547];

    va_start(ap, fmt);
    vsprintf(msg, fmt, ap);
    va_end(ap);
    sprintf(message, "C! %s", msg);
    _logger_tell(line, message, "DEBUG", 10);
}

void logger_info(int line, const char *fmt, ...)
{
    va_list ap;
    char message[550], msg[547];

    va_start(ap, fmt);
    vsprintf(msg, fmt, ap);
    va_end(ap);
    sprintf(message, "C! %s", msg);
    _logger_tell(line, message, "INFO", 20);
}

void logger_warn(int line, const char *fmt, ...)
{
    va_list ap;
    char message[550], msg[547];

    va_start(ap, fmt);
    vsprintf(msg, fmt, ap);
    va_end(ap);
    sprintf(message, "C! %s", msg);
    _logger_tell(line, message, "WARNING", 30);
}

void logger_error(int line, const char *fmt, ...)
{
    va_list ap;
    char message[550], msg[547];

    va_start(ap, fmt);
    vsprintf(msg, fmt, ap);
    va_end(ap);
    sprintf(message, "C! %s", msg);
    _logger_tell(line, message, "ERROR", 40);
}

void logger_critical(int line, const char *fmt, ...)
{
    va_list ap;
    char message[550], msg[547];

    va_start(ap, fmt);
    vsprintf(msg, fmt, ap);
    va_end(ap);
    sprintf(message, "C! %s", msg);
    _logger_tell(line, message, "CRITICAL", 50);
}
