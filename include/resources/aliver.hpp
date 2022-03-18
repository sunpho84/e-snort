#ifndef _ALIVER_HPP
#define _ALIVER_HPP

#ifdef HAVE_CONFIG_H
# include "config.hpp"
#endif

/// \file aliver.hpp

#include <ios/logger.hpp>
#include <metaprogramming/singleInstance.hpp>

namespace esnort
{
  /// Initialize
  void initialize(int narg,char** arg);
  
  /// Finalize
  void finalize();
  
  /// Run the program
  template <typename F>
  void runProgram(int narg,char** arg,F f)
  {
    initialize(narg,arg);
    
    f(narg,arg);
    
    finalize();
  }
}

#endif
