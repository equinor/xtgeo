#ifdef _DEBUG
  #undef _DEBUG
  #include <python.h>
  #define _DEBUG
#else
  #include <python.h>


/* using pythons logging! */
void logger_init(const char *fname);
void logger_info(char *msg);
void logger_warn(char *msg);
