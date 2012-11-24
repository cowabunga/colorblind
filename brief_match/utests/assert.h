#ifndef _ORAZAEV_UTIL_COMMON_ASSERT_H_
#define _ORAZAEV_UTIL_COMMON_ASSERT_H_


#include <cstdio>
#include <cstdlib>
#include <execinfo.h>




#ifndef STACK_TRACE_SIZE
#   define STACK_TRACE_SIZE 10
#endif



#ifdef _MSC_VER
#   define FUNCTION_NAME __FUNCTION__
#endif // _MSCVER

#ifdef __GNUC__
#   define FUNCTION_NAME __PRETTY_FUNCTION__
#endif




#define UTASSERT(CONDITION)                                         \
    if (!(CONDITION)) {                                             \
        fprintf(stderr, "%s:%d: %s: Assertion '%s' failed.\n",      \
                __FILE__, __LINE__, FUNCTION_NAME, #CONDITION);     \
        fprintf(stderr, "\nStack trace:\n");                        \
                                                                    \
        void * array[STACK_TRACE_SIZE];                             \
        size_t size;                                                \
                                                                    \
        size = backtrace(array, STACK_TRACE_SIZE);                  \
        backtrace_symbols_fd(array, size, 2);                       \
                                                                    \
        abort();                                                    \
    }




#ifdef DEBUG
#   define ASSERT(CONDITION) UTASSERT(CONDITION)
#else
#   define ASSERT(CONDITION)
#endif // DEBUG



#endif // _ORAZAEV_UTIL_COMMON_ASSERT_H_
