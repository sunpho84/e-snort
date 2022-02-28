#ifdef HAVE_CONFIG_H
 #include "config.hpp"
#endif

/// \file minimalLogger.hpp

#include <ios/logger.hpp>

#include <cstdarg>

namespace esnort
{
  namespace Logger
  {
    extern File logFile;
  }
  
  void minimalLogger(const char* format,
		     ...)
  {
    /// Starts the variadic arguments
    va_list ap;
    va_start(ap,format);
    
    logger().printVariadicMessage(format,ap);
    
    va_end(ap);
  }
}
