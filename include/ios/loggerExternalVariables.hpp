#ifndef _LOGGEREXTERNALVARIABLES_HPP
#define _LOGGEREXTERNALVARIABLES_HPP

#ifdef HAVE_CONFIG_H
# include "config.hpp"
#endif

/// \file loggerexternalvariables.hpp

#include <ios/file.hpp>

namespace esnort
{
  namespace Logger
  {
    /// Access to the logger as if it was a file
    DEFINE_OR_DECLARE_EXTERNAL_VARIABLE(File,logFile,{"/dev/stdout","w"});
    
    /// Indentation level
    DEFINE_OR_DECLARE_EXTERNAL_VARIABLE(int,indentLev,{0});
    
    /// Determine wheter the new line includes time
    DEFINE_OR_DECLARE_EXTERNAL_VARIABLE(bool,prependTime,{true});
    
    ///Verbosity level
    DEFINE_OR_DECLARE_EXTERNAL_VARIABLE(int,verbosityLv,{0});
    
    /// Decide whether only master thread can write here
    DEFINE_OR_DECLARE_EXTERNAL_VARIABLE(bool,onlyMasterThreadPrint,{true});
    
    /// Decide whether only master MPI can write here
    DEFINE_OR_DECLARE_EXTERNAL_VARIABLE(bool,onlyMasterRankPrint,{true});
  }
}

#endif
