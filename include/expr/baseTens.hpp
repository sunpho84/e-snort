#ifndef _BASETENS_HPP
#define _BASETENS_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file expr/baseTens.hpp

#include <expr/comps.hpp>
#include "expr/dynamicCompsProvider.hpp"
#include <expr/executionSpace.hpp>
#include <expr/expr.hpp>
#include <resources/memory.hpp>

namespace esnort
{
  /// Base Tensor
  ///
  /// Forward declaration
  template <typename T,
	    typename C,
	    typename F,
	    ExecutionSpace ES>
  struct BaseTens;

  /// Base Tensor
  template <typename T,
	    typename...C,
	    typename F,
	    ExecutionSpace ES>
  struct BaseTens<T,CompsList<C...>,F,ES> :
    DynamicCompsProvider<C...>,
    Expr<T>
  {
    using Expr<T>::operator=;
    
    /// Assign from another dynamic tensor of the very same type
    template <typename OtherT,
	      typename OtherC,
	      ExecutionSpace OtherES>
    BaseTens& operator=(const BaseTens<OtherT,OtherC,F,OtherES>& _oth)
    {
      T& t=this->crtp();
      const OtherT& oth=_oth.crtp();
      
      if(t.storageSize!=oth.storageSize)
	CRASH<<"Storage size not agreeing";
      
      LOGGER<<"Copying a "<<execSpaceName<ES><<" tensor into a "<<execSpaceName<OtherES><<" one";
      
      memory::memcpy<ES,OtherES>(t.storage,oth.storage,oth.storageSize);
      
      return *this;
    }
    
    // /// Initialize knowing the dynamic comps
    // explicit constexpr
    // BaseTens(const CompsList<C...>& c) :
    //   DynamicCompsProvider<C...>{c}
    // {
    // }
    
    /// Default constructor
    constexpr BaseTens()
    {
    }
  };
}

#endif
