#ifndef _MPIHIDDENVARIABLES_HPP
#define _MPIHIDDENVARIABLES_HPP

#ifdef HAVE_CONFIG_H
# include "config.hpp"
#endif

/// \file MpiHiddenVariables.hpp

#include <metaprogramming/hiddenVariableProvider.hpp>

namespace esnort::Mpi
{
  DEFINE_OR_DECLARE_HIDDEN_VARIABLE_WITH_CONST_ACCESSOR(bool,inited,{false});
  
#if ENABLE_MPI
  
  DEFINE_OR_DECLARE_HIDDEN_VARIABLE_WITH_CONST_ACCESSOR(int,rank,{0});
  
  DEFINE_OR_DECLARE_HIDDEN_VARIABLE_WITH_CONST_ACCESSOR(int,nRanks,{1});
  
  DEFINE_OR_DECLARE_HIDDEN_VARIABLE_WITH_CONST_ACCESSOR(bool,isMaster,{false});
  
#else
  
  constexpr int rank=0;
  
  constexpr int nRanks=1;
  
  constexpr bool isMaster=true;
  
#endif
}

#include <metaprogramming/hiddenVariableProviderTail.hpp>

#endif
