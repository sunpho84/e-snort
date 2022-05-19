#ifndef _TENS_HPP
#define _TENS_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file expr/nodes/tens.hpp

#include <tuple>
#include <expr/nodes/stackTens.hpp>

namespace grill
{
  enum class Dynamicity{NOT_DYNAMICAL,DYNAMICAL};

  template <typename Comps>
  constexpr Dynamicity inferDynamicity()
  {
    constexpr bool stackable=
      std::apply([](const auto&...c)
      {
	return (c.sizeIsKnownAtCompileTime and ... and true);
      },Comps{});
    
    if constexpr(stackable)
      return Dynamicity::NOT_DYNAMICAL;
    else
      return Dynamicity::DYNAMICAL;
  }
  
  template <typename Comps,
	    typename Fund,
	    ExecSpace ES=defaultExecSpace,
	    Dynamicity D=inferDynamicity<Comps>(),
	    typename...DynamicSize>
  auto getTens(DynamicSize&&...dynamicSize)
  {
    if constexpr(D==Dynamicity::NOT_DYNAMICAL)
      return StackTens<Comps,Fund>();
    else
      return DynamicTens<Comps,Fund,ES>(std::make_tuple(dynamicSize...));
  }
}

#endif
