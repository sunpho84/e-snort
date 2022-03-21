#ifndef _BASETENS_HPP
#define _BASETENS_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file expr/baseTens.hpp

#include <expr/comps.hpp>
#include <expr/dynamicCompsProvider.hpp>
#include <expr/executionSpace.hpp>
#include <expr/expr.hpp>
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
    // /// Import assignment operator from Expr
    // using Expr<T>::operator=;
    
    // INLINE_FUNCTION
    // BaseTens& operator=(const BaseTens& oth)
    // {
    //   this->Expr<T>::operator=(oth);
      
    //   return *this;
    // }
    /// Assign from another dynamic tensor of the very same type
    template <typename OtherT,
	      typename OtherF,
	      ExecutionSpace OtherES>
    INLINE_FUNCTION
    BaseTens& assign(const BaseTens<OtherT,CompsList<C...>,OtherF,OtherES>& _oth)
    {
      this->Expr<T>::operator=(_oth);
      // T& t=DE_CRTPFY(T,this);
      // const OtherT& oth=DE_CRTPFY(const OtherT,&_oth);
      
      // if(t.storageSize!=oth.storageSize)
      // 	CRASH<<"Storage sizes not agreeing";
      
      // LOGGER<<"Copying a "<<execSpaceName<OtherES><<" tensor into a "<<execSpaceName<ES><<" one";
      
      // memory::memcpy<ES,OtherES>(t.storage,oth.storage,oth.storageSize);
      
      return *this;
    }
    
    /// Define the copy-assignment operator
    INLINE_FUNCTION
    BaseTens& operator=(const BaseTens& oth)
    {
      return this->assign(oth);
    }
    
    /// Define the move-assignment operator
    INLINE_FUNCTION
    BaseTens& operator=(BaseTens&& oth)
    {
      return this->assign(std::move(oth));
    }
    
    /// Assign from another dynamic tensor of the very same type
    template <typename OtherT,
	      typename OtherF,
	      ExecutionSpace OtherES>
    INLINE_FUNCTION
    BaseTens& operator=(const BaseTens<OtherT,CompsList<C...>,OtherF,OtherES>& _oth)
    {
      return this->assign(_oth);
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
    
    /// Default constructor
    constexpr BaseTens()
    {
    }
  };
}

#endif
