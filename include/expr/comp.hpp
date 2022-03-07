#ifndef _COMP_HPP
#define _COMP_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file comp.hpp

#include <expr/baseComp.hpp>
#include <expr/transposableComp.hpp>

namespace esnort
{
  template <compFeat::IsTransposable IsTransposable,
	    typename Index,
	    typename Derived>
  struct Comp  :
    compFeat::Transposable<IsTransposable,Derived>,
    BaseComp<Derived,Index>
  {
    using Base=BaseComp<Derived,Index>;
    
    using Base::Base;
  };
}

#endif
