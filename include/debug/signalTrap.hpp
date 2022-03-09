#ifndef _SIGNALTRAP_HPP
#define _SIGNALTRAP_HPP

#ifdef HAVE_CONFIG_H
# include "config.hpp"
#endif

/// \file signalTrap.hpp

namespace esnort
{
  /// Called when signal received
  void signalHandler(int sig);
  
  /// Set all traps
  void setSignalTraps();
}

#endif
