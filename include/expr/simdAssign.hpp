#ifndef _SIMDASSIGN_HPP
#define _SIMDASSIGN_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file expr/simdAssign.hpp
///
/// \brief Assign two expressions usng SIMD

#include <expr/assignerFactory.hpp>
#include <expr/compLoops.hpp>
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
    /*LOGGER<<"Using simd assign";*/
    
    lhs.simdify()=rhs.simdify();
  }
}

#endif
