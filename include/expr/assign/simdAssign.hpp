#ifndef _SIMDASSIGN_HPP
#define _SIMDASSIGN_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file expr/assign/simdAssign.hpp
///
/// \brief Assign two expressions usng SIMD

#include <expr/assign/assignerFactory.hpp>
#include <expr/comps/compLoops.hpp>
#include <ios/logger.hpp>

namespace esnort
{
  /// Assign two expressions using SIMD
  template <typename Lhs,
	    typename Rhs>
  INLINE_FUNCTION
  void simdAssign(Lhs& lhs,
		  const Rhs& rhs)
  {
    LOGGER_LV3_NOTIFY("Using simd assign");
    
    lhs.simdify()=rhs.simdify();
  }
}

#endif
