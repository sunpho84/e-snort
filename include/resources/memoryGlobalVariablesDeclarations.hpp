#ifndef _MEMORYGLOBALVARIABLESDECLARATIONS_HPP
#define _MEMORYGLOBALVARIABLESDECLARATIONS_HPP

/// \file memoryGlobalVariablesDeclarations.hpp

#ifdef HAVE_CONFIG_H
# include "config.hpp"
#endif

#include <metaprogramming/globalVariableProvider.hpp>

namespace grill::memory
{
  DEFINE_OR_DECLARE_GLOBAL_VARIABLE(bool,useCache,("true"));
}

#endif
