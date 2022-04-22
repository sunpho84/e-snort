#ifndef _ASSIGNERFACTORY_HPP
#define _ASSIGNERFACTORY_HPP

#ifdef HAVE_CONFIG_H
# include "config.hpp"
#endif

/// \file assign/assignerFactory.hpp

#include <metaprogramming/inline.hpp>
#include <metaprogramming/arithmeticTraits.hpp>

namespace grill
{
  /// Returns a lambda function to perform assignment
  template <typename Lhs,
	    typename Rhs>
  INLINE_FUNCTION constexpr
  auto getAssigner(Lhs& lhs,
		   const Rhs& rhs)
  {
    return
      [&lhs,&rhs](const auto&...comps) CONSTEXPR_INLINE_ATTRIBUTE
      {
	assign(lhs(comps...),rhs(comps...));
      };
  }
}

#endif
