#ifndef _TIMERGLOBALVARIABLESDECLARATIONS_HPP
#define _TIMERGLOBALVARIABLESDECLARATIONS_HPP

/// \file timer.hpp

#ifdef HAVE_CONFIG_H
 #include "config.hpp"
#endif

#include <system/timer.hpp>

namespace esnort
{
  DEFINE_OR_DECLARE_GLOBAL_VARIABLE(Timer,timings,("Total time",Timer::NO_FATHER,Timer::UNSTOPPABLE));
}

#endif
