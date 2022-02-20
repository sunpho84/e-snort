#ifndef _ASSIGNBASE_HPP
#define _ASSIGNBASE_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file assign/assignBase.hpp
///
/// \brief Forward declaration of assignment

#include <cstdio>
#include <utility>

#if ENABLE_DEVICE_CODE
# include <cuda/cuda.hpp>
#endif
#include <expr/executionSpace.hpp>

namespace esnort
{
  /// Structure to decide the correct path of assignement
  ///
  /// Forward declaration
  template <ExecutionSpace LhsSpace,
	    ExecutionSpace RhsSpace,
	    WhichSideToChange WhichSide>
  struct Assign;
}

#endif
