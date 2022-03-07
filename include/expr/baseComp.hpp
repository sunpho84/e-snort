#ifndef _BASE_COMP_HPP
#define _BASE_COMP_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file baseComp.hpp
///
/// \brief Implements a tensor comp

#include <metaprogramming/inline.hpp>
#include <metaprogramming/nonConstMethod.hpp>
#include <metaprogramming/arithmeticOperatorsViaCast.hpp>
#include <metaprogramming/typeConversion.hpp>

namespace esnort
{
  /// A component
  template <typename _C,
	    typename _Index>
  struct BaseComp:
    ArithmeticOperators<_Index,_C>
  {
    /// Value type
    using Index=_Index;
    
    /// Component
    using C=_C;
    
    /// Value
    Index i;
    
    /// Returns the size at compile time, with assert
    static constexpr Index sizeAtCompileTimeAssertingNotDynamic()
    {
      static_assert(sizeIsKnownAtCompileTime,"Size not known at compile time!");
      
      return C::sizeAtCompileTime;
    }
    
    /// Determine whether the size is known at compile time
    static constexpr bool sizeIsKnownAtCompileTime=
      (C::sizeAtCompileTime!=0);
    
    /// Default constructor
    INLINE_FUNCTION HOST_DEVICE_ATTRIB constexpr
    BaseComp() : i(0)
    {
    }
    
    /// Init from value
    template <typename T=Index,
	      ENABLE_THIS_TEMPLATE_IF(isSafeNumericConversion<Index,T>)>
    INLINE_FUNCTION HOST_DEVICE_ATTRIB constexpr
    BaseComp(T&& i) : i(i)
    {
    }
    
    /// Assignment operator
    INLINE_FUNCTION HOST_DEVICE_ATTRIB constexpr
    BaseComp& operator=(const Index& oth) &
    {
      i=oth;
      
      return
	*this;
    }
    
    /// Assignment operator of a TensComp
    INLINE_FUNCTION HOST_DEVICE_ATTRIB constexpr
    BaseComp& operator=(const BaseComp& oth) &
    {
      return
	(*this)=oth.i;
    }
    
    /// Forbid assignement to a temporary
    BaseComp& operator=(const BaseComp& oth) && = delete;
    
    /// Convert to actual reference with const attribute
    INLINE_FUNCTION constexpr HOST_DEVICE_ATTRIB
    const Index& operator()() const
    {
      return i;
    }
    
    PROVIDE_ALSO_NON_CONST_METHOD_WITH_ATTRIB(toPod,HOST_DEVICE_ATTRIB);
    
#define PROVIDE_CAST_TO_VALUE(ATTRIB)					\
    /*! Convert to actual reference with or without const attribute */	\
    INLINE_FUNCTION HOST_DEVICE_ATTRIB constexpr			\
    explicit operator ATTRIB Index&() ATTRIB				\
    {									\
      return (*this)();							\
    }
    
    PROVIDE_CAST_TO_VALUE(const);
    PROVIDE_CAST_TO_VALUE(/* non const */);
    
#undef PROVIDE_CAST_TO_VALUE
    
    /// Convert to actual reference with const attribute, to be removed
    INLINE_FUNCTION constexpr HOST_DEVICE_ATTRIB
    const Index& nastyConvert() const
    {
      return toPod();
    }
    
    PROVIDE_ALSO_NON_CONST_METHOD_WITH_ATTRIB(nastyConvert,HOST_DEVICE_ATTRIB);
    
    auto transp() const
    {
      return (typename _C::Transp)(*this)();
    }
  };
}

#endif
