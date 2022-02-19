#ifndef _ASSIGNBASE_HPP
#define _ASSIGNBASE_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

#include <cstdio>
#include <utility>

#if ENABLE_CUDA_CODE
# include <cuda/cuda.hpp>
#endif
#include <expr/executionSpace.hpp>

namespace esnort
{
  template <ExecutionSpace LhsSpace,
	    ExecutionSpace RhsSpace,
	    WhichSideToChange WhichSide>
  struct Assign;
}

#endif
