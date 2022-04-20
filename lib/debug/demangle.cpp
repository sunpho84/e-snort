#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file demangle.cpp

#if CAN_DEMANGLE
# include <cxxabi.h>
#endif

#include <debug/demangle.hpp>

namespace grill
{
  std::string demangle(const std::string& what)
  {
    
#if CAN_DEMANGLE
  
    /// Returned status of demangle
    int status;
    
    /// Demangled
    char* name=
      abi::__cxa_demangle(what.c_str(),0,0,&status);
    
    /// Copy the result
    std::string out=
      (status==0)?
      name:
      what+" (failed demangle)";
    
    // Free if succeded
    if(status==0)
      free(name);
    
    return
      out;
    
#else
    
    return
      "(unable to demangle)";
    
#endif
    
  }
}
