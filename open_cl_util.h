#ifndef OPEN_CL_UTIL_H
#define OPEN_CL_UTIL_H

#include "graph_kernel.h"

namespace OCLUtil {
    void checkError(cl_int errorNum, cl_int expected, const char* msg);
}



#endif //OPEN_CL_UTIL_H
