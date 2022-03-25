#ifndef _BINDCOMPS_HPP
#define _BINDCOMPS_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file bindComps.hpp

#include <expr/comps.hpp>
#include <expr/expr.hpp>

namespace esnort
{
#warning maybe simplifiable

  /// Component binder
  ///
  /// Forward declaration to capture the index sequence
  template <typename _BC,
	    typename _E,
	    typename _Comps,
	    typename _EvalTo>
  struct CompsBinder;
  
#define THIS					\
  CompsBinder<CompsList<Bc...>,_E,CompsList<C...>,_EvalTo>

#define BASE					\
    Expr<THIS>
  
  /// Component binder
  ///
  template <typename...Bc,
	    typename _E,
	    typename...C,
	    typename _EvalTo>
  struct THIS :
    BASE
  {
    /// Import the base expression
    using Base=BASE;
    
    using This=THIS;
    
#undef BASE
    
#undef THIS
    
    /// Import assignemnt operator
    using Base::operator=;
    
    /// Components
    using Comps=
      CompsList<C...>;

}

#endif
