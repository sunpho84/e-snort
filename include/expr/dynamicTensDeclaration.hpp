#ifndef _DYNAMICTENSORDECLARATION_HPP
#define _DYNAMICTENSORDECLARATION_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file expr/dynamicTensorDeclaration.hpp

#include <expr/executionSpace.hpp>
#include <metaprogramming/detectableAs.hpp>

namespace esnort
{
  PROVIDE_DETECTABLE_AS(DynamicTens);
  
  /// Tensor
  ///
  /// Forward declaration
  template <typename C,
	    typename F,
	    ExecSpace ES>
  struct DynamicTens;
}

#endif
