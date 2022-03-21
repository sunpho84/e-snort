#ifndef _DIRECTASSIGN_HPP
#define _DIRECTASSIGN_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file expr/directAssign.hpp
///
/// \brief Assign two expressions directly

#include <expr/assignerFactory.hpp>
#include <expr/compLoops.hpp>
#include <ios/logger.hpp>

namespace esnort
{
  /// Assign two expressions directly
  template <typename Lhs,
	    typename Rhs>
  INLINE_FUNCTION
  void directAssign(Lhs& lhs,
		    const Rhs& rhs)
  {
    LOGGER<<"Using direct assign";
    
    loopOnAllComps<typename Lhs::Comps>(lhs.dynamicSizes,getAssigner(lhs,rhs));
  }
}

#endif
