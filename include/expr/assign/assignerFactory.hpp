#ifndef _ASSIGNERFACTORY_HPP
#define _ASSIGNERFACTORY_HPP

#include <metaprogramming/inline.hpp>

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
	lhs(comps...)=rhs(comps...);
      };
  }
}

#endif
