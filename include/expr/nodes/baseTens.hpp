#ifndef _BASETENS_HPP
#define _BASETENS_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file expr/nodes/baseTens.hpp

#include <expr/comps/comps.hpp>
#include <expr/comps/dynamicCompsProvider.hpp>
#include <expr/nodes/dynamicTensDeclaration.hpp>
#include <expr/assign/executionSpace.hpp>
#include <expr/nodes/node.hpp>
#include <metaprogramming/crtp.hpp>
#include <resources/memory.hpp>
#include <resources/SIMD.hpp>

namespace esnort
{
  PROVIDE_DETECTABLE_AS(BaseTens);
  
  /// Base Tensor
  ///
  /// Forward declaration
  template <typename T,
	    typename C,
	    typename F,
	    ExecSpace ES>
  struct BaseTens;
  
  /// Base Tensor
  template <typename T,
	    typename...C,
	    typename F,
	    ExecSpace ES>
  struct BaseTens<T,CompsList<C...>,F,ES> :
    DynamicCompsProvider<CompsList<C...>>,
    DetectableAsBaseTens,
    Node<T>
  {
    // /// Assign from another dynamic tensor
    // template <typename OtherT,
    // 	      typename OtherF,
    // 	      ExecSpace OtherES>
    // INLINE_FUNCTION
    // BaseTens& assign(const BaseTens<OtherT,CompsList<C...>,OtherF,OtherES>& _oth)
    // {
    //   this->Node<T>::operator=(_oth);
      
    //   return *this;
    // }
    
    using Node<T>::operator=;
    
    /// Copy-assign
    INLINE_FUNCTION
    BaseTens& operator=(const BaseTens& oth)
    {
      Node<T>::operator=(oth);
      
      return *this;
    }
    
    /// Move-assign
    INLINE_FUNCTION
    BaseTens& operator=(BaseTens&& oth)
    {
      Node<T>::operator=(std::move(oth));
      
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
	
      device::memcpy<ES,otherExecSpace<ES>>(t.storage,oth.storage,t.nElements*sizeof(F));
      
      return t;
    }
    
    /// Return whether can be assigned at compile time
    static constexpr bool canAssignAtCompileTime=
      not std::is_const_v<F>;
    
    /// Traits of simdifying
    using _SimdifyTraits=
      CompsListSimdifiableTraits<CompsList<C...>,F>;
    
    /// Components on which simdifying
    using SimdifyingComp=
      typename _SimdifyTraits::LastComp;
    
    /// States whether the tensor can be simdified
    static constexpr bool canSimdify=
#if ENABLE_DEVICE_CODE
      (ES==ExecSpace::HOST) and
#endif
      _SimdifyTraits::canSimdify;
    
    /// Returns a const reference
    auto getRef() const;
    
    /// Returns a reference
    auto getRef();
    
    /// Returns a const simdified view
    auto simdify() const;
    
    /// Returns a simdified view
    auto simdify();
    
    /// Gets a copy on a specific execution space
    template <ExecSpace OES>
    DynamicTens<CompsList<C...>,F,OES> getCopyOnExecSpace() const;
    
    /// Default constructor
    constexpr BaseTens()
    {
    }
  };
}

#endif