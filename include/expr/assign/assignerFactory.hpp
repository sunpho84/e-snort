#ifndef _ASSIGNERFACTORY_HPP
#define _ASSIGNERFACTORY_HPP

#ifdef HAVE_CONFIG_H
# include "config.hpp"
#endif

/// \file assign/assignerFactory.hpp

#include <metaprogramming/inline.hpp>
#include <metaprogramming/arithmeticTraits.hpp>
#include <tuples/tupleSubset.hpp>

namespace grill
{
  /// Returns a lambda function to perform assignment
  ///
  /// Filters out the missing elements to allow direct assignment if types aere missing on rhs
  template <typename Lhs,
	    typename Rhs>
  INLINE_FUNCTION constexpr
  auto getAssigner(Lhs& lhs,
		   const Rhs& rhs)
  {
    return
      [&lhs,&rhs](const auto&...comps) CONSTEXPR_INLINE_ATTRIBUTE
      {
	assign(lhs(comps...),
	       std::apply(rhs,tupleGetSubset<typename Rhs::Comps>(std::make_tuple(comps...))));
      };
  }
}

#endif
