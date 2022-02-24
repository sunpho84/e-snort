#ifdef HAVE_CONFIG_H
 #include "config.hpp"
#endif

/// \file minimalLogger.hpp

#include <ios/logger.hpp>
#include <ios/minimalLogger.hpp>

#include <cstdarg>

namespace esnort
{
  void minimalLogger(Logger& logger,
		     const char* format,
		     ...)
  {
    /// Starts the variadic arguments
    va_list ap;
    va_start(ap,format);
    
    logger.printVariadicMessage(format,ap);
    
    va_end(ap);
  }
}
