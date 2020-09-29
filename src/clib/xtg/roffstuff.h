
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <time.h>

void
strwrite(int mode, const char *str, FILE *stream);

void
boolwrite(int mode, int value, FILE *stream);

void
intwrite(int mode, int value, FILE *stream);

void
fltwrite(int mode, float value, FILE *stream);
