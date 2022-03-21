#ifndef _ASSIGNERFACTORY_HPP
#define _ASSIGNERFACTORY_HPP

#include <metaprogramming/inline.hpp>

namespace esnort
{
  /// Returns a lambda function to perform assignment
  template <typename Lhs,
	    typename Rhs>
  INLINE_FUNCTION
  auto getAssigner(Lhs& lhs,
		   const Rhs& rhs)
  {
    return
      [&lhs,&rhs](const auto&...comps) INLINE_ATTRIBUTE
      {
	lhs(comps...)=rhs(comps...);
      };
  }
}

#endif
