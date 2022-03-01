#ifndef _THREADSHIDDENVARIABLESDECLARATIONS_HPP
#define _THREADSHIDDENVARIABLESDECLARATIONS_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file threadsHiddenVariablesDeclarations.hpp

#include <metaprogramming/hiddenVariableProvider.hpp>

namespace esnort::threads
{
  DEFINE_OR_DECLARE_HIDDEN_VARIABLE_WITH_CONST_ACCESSOR(int,nThreads,(1));
}

#include <metaprogramming/hiddenVariableProviderTail.hpp>

#endif
