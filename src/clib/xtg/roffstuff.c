/*
 ***************************************************************************************
 * Special generic utilities for ROFF ascii or binary write
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ***************************************************************************************
 */

#include "roffstuff.h"

static int
replacechar_bin(char *buffer, const char *str)
{
    int len = strlen(str);
    int i;
    for (i = 0; i < len; i++) {
        buffer[i] = str[i];
        if (str[i] == '^' || str[i] == '$') {
            buffer[i] = '\0';
        }
    }
    return len;
}

static int
replacechar_asc(char *buffer, const char *str)
{
    int len = strlen(str);
    int i;
    for (i = 0; i < len; i++) {
        buffer[i] = str[i];
        if (str[i] == '^') {
            buffer[i] = ' ';
        }
        if (str[i] == '$') {
            buffer[i] = '\n';
        }
    }
    return len;
}

void
strwrite(int mode, const char *str, FILE *stream)
{
    // write a character string to binary or ascii
    // "tag^filetype$" is converted to:
    //      "tag\0filetype\0" for binary
    //      "tag filetype\n" for ascii
    if (mode == 0) {
        char buffer[200] = "";
        int nlen = replacechar_bin(buffer, str);
        fwrite(buffer, 1, nlen, stream);
    } else {
        char buffer[200] = "";
        replacechar_asc(buffer, str);
        fputs(buffer, stream);
    }
}

void
boolwrite(int mode, int value, FILE *stream)
{
    // write an integer as bool'ish, if ascii followed by newline or blank
    if (mode == 0 || mode == 10) {
        char bvalue = value;
        fwrite(&bvalue, 1, 1, stream);
    } else {
        if (mode == 1) {
            fprintf(stream, "%1d\n", value);
        } else {
            fprintf(stream, "%1d ", value);
        }
    }
}

void
intwrite(int mode, int value, FILE *stream)
{
    // write an integer, if ascii followed by newline or blank
    if (mode == 0 || mode == 10) {
        fwrite(&value, 4, 1, stream);
    } else {
        if (mode == 1) {
            fprintf(stream, "%d\n", value);
        } else {
            fprintf(stream, "%d ", value);
        }
    }
}

void
fltwrite(int mode, float value, FILE *stream)
{
    // write a float, if ascii followed by newline or blank
    if (mode == 0 || mode == 10) {
        fwrite(&value, 4, 1, stream);
    } else {
        if (mode == 1) {
            fprintf(stream, "%.8e\n", value);
        } else {
            // no newline but space
            fprintf(stream, "%.8e ", value);
        }
    }
}
