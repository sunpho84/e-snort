#ifndef _DIRECTASSIGN_HPP
#define _DIRECTASSIGN_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file expr/assign/directAssign.hpp
///
/// \brief Assign two expressions directly

#include <expr/assign/assignerFactory.hpp>
#include <expr/comps/compLoops.hpp>
#include <ios/logger.hpp>

namespace grill
{
  /// Assign two expressions directly
  template <typename Lhs,
	    typename Rhs>
  INLINE_FUNCTION constexpr
  void directAssign(Lhs& lhs,
		    const Rhs& rhs)
  {
    LOGGER_LV3_NOTIFY("Using direct assign");
    
    loopOnAllComps<typename Lhs::Comps>(lhs.getDynamicSizes(),getAssigner(lhs,rhs));
  }
}

#endif
