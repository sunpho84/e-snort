#ifndef _DYNAMICTENSORDEFINITION_HPP
#define _DYNAMICTENSORDEFINITION_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file expr/dynamicTensordefinition.hpp

#include <expr/executionSpace.hpp>

namespace esnort
{
  /// Tensor
  ///
  /// Forward declaration
  template <typename C,
	    typename F,
	    ExecutionSpace ES,
	    bool _IsRef=false>
  struct DynamicTens;
}

#endif
