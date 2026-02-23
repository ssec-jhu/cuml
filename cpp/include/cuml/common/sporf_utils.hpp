/*
 * sporf_utils.hpp
 *
 * Copyright (c) 2026, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <stdio.h>
#include <stdarg.h>
#include <time.h>
#include <sys/time.h>


/* common implementation for dtPrint and dtLog */
static int dtEmit( const char sep, const char* fmt, va_list& args )
{
    // time since epoch
    struct timeval tv;
    gettimeofday( &tv, NULL );

    // current local time
    struct tm lt = *localtime( &tv.tv_sec );

    // emit YYYYMMDD HHmmss.ffffff
    int cch = fprintf( stdout,
                        "%04d%02d%02d %02d%02d%02d.%06ld%c",
                        lt.tm_year+1900, lt.tm_mon+1, lt.tm_mday, lt.tm_hour, lt.tm_min, lt.tm_sec,
                        tv.tv_usec,
                        sep
                     );

    // emit formatted parameters
    cch += vfprintf( stdout, fmt, args );
    va_end( args );

    // emit trailing LF
    cch += fprintf( stdout, "\n" );

    return cch;
}

/* emit timestamp and formatted string, separated by tab (ASCII 9) */
int dtPrint( const char * fmt, ... )
{
    va_list args;
    va_start( args, fmt );
    int cch = dtEmit( '\t', fmt, args );
    va_end( args );

    return cch;
}

/* emit timestamp and formatted string, separated by space (ASCII 32) */
int dtLog( const char * fmt, ... )
{
    va_list args;
    va_start( args, fmt );
    int cch = dtEmit( ' ', fmt, args );
    va_end( args );

    return cch;
}

/* emit byte count formatted as bytes and gigabytes

   (not implemented because the only safe way to do it is to start passing string
    pointers around)

   it's easy enough to write something like

    char s[64];
    sprintf( s, "%ld %2.1f", nBytes, nBytes / (1024.0 * 1024 * 1024));

   as hardcoded C/C++ or as a C macro if we ever need it
*/