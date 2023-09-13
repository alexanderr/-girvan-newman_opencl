#include "open_cl_util.h"
#include <iostream>

void OCLUtil::checkError(cl_int errorNum, cl_int expected, const char* msg)
{
    if(errorNum != expected)
        std::cerr << "Error occured: " << errorNum << " : " << msg << std::endl;
}



