#ifndef _DYNAMICTENSORDECLARATION_HPP
#define _DYNAMICTENSORDECLARATION_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file expr/nodes/dynamicTensorDeclaration.hpp

#include <expr/assign/executionSpace.hpp>
#include <metaprogramming/detectableAs.hpp>

namespace grill
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
