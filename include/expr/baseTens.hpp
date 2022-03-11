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
    /// Import assignment operator from Expr
    using Expr<T>::operator=;
    
    /// Assign from another dynamic tensor of the very same type
    template <typename OtherT,
	      ExecutionSpace OtherES>
    BaseTens& operator=(const BaseTens<OtherT,CompsList<C...>,F,OtherES>& _oth)
    {
      T& t=this->crtp();
      const OtherT& oth=_oth.crtp();
      
      if(t.storageSize!=oth.storageSize)
	CRASH<<"Storage sizes not agreeing";
      
      LOGGER<<"Copying a "<<execSpaceName<OtherES><<" tensor into a "<<execSpaceName<ES><<" one";
      
      memory::memcpy<ES,OtherES>(t.storage,oth.storage,oth.storageSize);
      
      return *this;
    }
    
    
    /// Return whether can be assigned at compile time
    static constexpr bool canAssignAtCompileTime=
      not std::is_const_v<F>;
    
    /// Returns a const reference
    auto getRef() const;
    
    /// Returns a reference
    auto getRef();
    
    /// Default constructor
    constexpr BaseTens()
    {
    }
  };
}

#endif
