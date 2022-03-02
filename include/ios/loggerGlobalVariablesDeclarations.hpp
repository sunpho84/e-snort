#ifndef _LOGGERGLOBALVARIABLESDECLARATIONS_HPP
#define _LOGGERGLOBALVARIABLESDECLARATIONS_HPP

#ifdef HAVE_CONFIG_H
# include "config.hpp"
#endif

/// \file loggerGlobalVariablesDeclarations.hpp

#include <ios/file.hpp>
#include <resources/threads.hpp>
#include <metaprogramming/globalVariableProvider.hpp>

namespace esnort::Logger
{
  /// Access to the logger as if it was a file
  DEFINE_OR_DECLARE_GLOBAL_VARIABLE(File,logFile,{"/dev/stdout","w"});
  
  /// Indentation level
  DEFINE_OR_DECLARE_GLOBAL_VARIABLE(int,indentLev,{0});
  
  /// Determine wheter the new line includes time
  DEFINE_OR_DECLARE_GLOBAL_VARIABLE(bool,prependTime,{true});
  
  ///Verbosity level
  DEFINE_OR_DECLARE_GLOBAL_VARIABLE(int,verbosityLv,{0});
  
  /// Decide whether only master thread can write here
  DEFINE_OR_DECLARE_GLOBAL_VARIABLE(bool,onlyMasterThreadPrint,{true});
  
  /// Decide whether only master MPI can write here
  DEFINE_OR_DECLARE_GLOBAL_VARIABLE(bool,onlyMasterRankPrint,{true});
  
  /// Lock the logger
  DEFINE_OR_DECLARE_GLOBAL_VARIABLE(threads::Lock,lock,);
}

#endif
