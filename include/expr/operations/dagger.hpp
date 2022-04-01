#ifndef _DAGGER_HPP
#define _DAGGER_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file expr/operations/dagger.hpp

#include <expr/nodes/conj.hpp>
#include <expr/nodes/expr.hpp>
#include <expr/nodes/transp.hpp>

namespace esnort
{
  /// Take the dagger of an expression
  template <typename _E,
	    ENABLE_THIS_TEMPLATE_IF(isExpr<_E>)>
  HOST_DEVICE_ATTRIB INLINE_FUNCTION constexpr
  decltype(auto) dag(_E&& e)
  {
    using E=std::decay_t<_E>;
    
    if constexpr(isConjugator<E>)
      return transpose(FORWARD_MEMBER_VAR(_E,e,conjExpr));
    else
      if constexpr(isTransposer<E>)
	return conj(FORWARD_MEMBER_VAR(_E,e,transpExpr));
      else
	return transp(conj(std::forward<_E>(e)));
  }
}

#endif
