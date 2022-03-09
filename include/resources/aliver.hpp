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
  /// Class used to provocate initialization of Mpi
  struct Aliver :
    public SingleInstance<Aliver>
  {
    /// Creates
    Aliver();
    
    /// Destroys
    ~Aliver();
  };
}

#endif
