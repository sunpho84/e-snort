#ifdef HAVE_CONFIG_H
# include "config.hpp"
#endif

/// \file minimalCrash.cpp

#include <cstdarg>

#include <debug/crash.hpp>
#include <ios/logger.hpp>

namespace grill
{
  void minimalCrash(const char* path,
		    const int line,
		    const char* funcName,
		    const char* format,
		    ...)
  {
    /// Starts the variadic arguments
    va_list ap;
    va_start(ap,format);
    
    (logger()<<Crasher(path,line,funcName)).printVariadicMessage(format,ap);
    
    va_end(ap);
  }
}
