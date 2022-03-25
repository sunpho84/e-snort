#ifndef _BASETENS_HPP
#define _BASETENS_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file expr/baseTens.hpp

#include <expr/comps.hpp>
#include <expr/dynamicCompsProvider.hpp>
#include <expr/dynamicTensDeclaration.hpp>
#include <expr/executionSpace.hpp>
#include <expr/expr.hpp>
#include <metaprogramming/crtp.hpp>
#include <resources/memory.hpp>
#include <resources/SIMD.hpp>

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
    // /// Assign from another dynamic tensor
    // template <typename OtherT,
    // 	      typename OtherF,
    // 	      ExecutionSpace OtherES>
    // INLINE_FUNCTION
    // BaseTens& assign(const BaseTens<OtherT,CompsList<C...>,OtherF,OtherES>& _oth)
    // {
    //   this->Expr<T>::operator=(_oth);
      
    //   return *this;
    // }
    
    using Expr<T>::operator=;
    
    /// Copy-assign
    INLINE_FUNCTION
    BaseTens& operator=(const BaseTens& oth)
    {
      Expr<T>::operator=(oth);
      
      return *this;
    }
    
    /// Move-assign
    INLINE_FUNCTION
    BaseTens& operator=(BaseTens&& oth)
    {
      Expr<T>::operator=(std::move(oth));
      
      return *this;
    }
    
    /// Copy from other execution space
    template <typename U>
    INLINE_FUNCTION
    T& operator=(const BaseTens<U,CompsList<C...>,F,otherExecSpace<ES>>& _oth)
    {
      /// Derived class of this
      T& t=DE_CRTPFY(T,this);
      
      /// Derived class of _oth
      const U& oth=DE_CRTPFY(const U,&_oth);
      
      if(t.getDynamicSizes()!=oth.getDynamicSizes())
	CRASH<<"Not matching dynamic sizes";
	
      device::memcpy<ES,otherExecSpace<ES>>(t.storage,oth.storage,t.storageSize*sizeof(F));
      
      return t;
    }
    
    using _SimdifyTraits=
      CompsListSimdifiableTraits<CompsList<C...>,F>;
    
    /// States whether the tensor can be simdified
    static constexpr int canSimdify=
#if ENABLE_DEVICE_CODE
      (ES==ExecutionSpace::HOST) and
#endif
      _SimdifyTraits::canSimdify;
    
    /// Return whether can be assigned at compile time
    static constexpr bool canAssignAtCompileTime=
      not std::is_const_v<F>;
    
    /// Returns a const reference
    auto getRef() const;
    
    /// Returns a reference
    auto getRef();
    
    /// Returns a const simdified view
    auto simdify() const;
    
    /// Returns a simdified view
    auto simdify();
    
    /// Gets a copy on a specific execution space
    template <ExecutionSpace OES>
    auto getCopyOnExecSpace() const;
    
    /// Default constructor
    constexpr BaseTens()
    {
    }
  };
}

#endif
