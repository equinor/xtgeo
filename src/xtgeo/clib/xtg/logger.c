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
static unsigned long XTG_START_TIME = 0;

unsigned long _get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    unsigned long ret = tv.tv_usec;
    ret /= 1000;
    ret += (tv.tv_sec * 1000);
    return ret;
}

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
    XTG_START_TIME = _get_time();

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
        printf("Start time: %lu\n", XTG_START_TIME);
    }
    return -1;
}


void _logger_tell(int line, const char *filename, const char *func, const char *message,
                   const char *ltype, int level)
{

    unsigned long deltatime = _get_time() - XTG_START_TIME;

    float dtime = deltatime / 1000.0;

    if (level >= XTG_LOGGING_LEVEL) {
        if (XTG_LOGGING_FORMAT == 1) {
            printf("%8s: (%3.2fs) \t%s\n", ltype, dtime, message);
        }
        else if (XTG_LOGGING_FORMAT == 2) {
            printf("%8s (%3.2fs) %44s [%42s] %4d >> \t%s\n", ltype, dtime, filename,
                   func, line, message);
        }

    }
    XTG_START_TIME = _get_time();
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
