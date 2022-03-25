#ifndef _DYNAMICTENSORDECLARATION_HPP
#define _DYNAMICTENSORDECLARATION_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file expr/dynamicTensorDeclaration.hpp

#include <expr/executionSpace.hpp>

namespace esnort
{
  /// Tensor
  ///
  /// Forward declaration
  template <typename C,
	    typename F,
	    ExecSpace ES,
	    bool _IsRef=false>
  struct DynamicTens;
}

#endif
