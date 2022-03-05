#ifndef _ARITHMETICOPERATORSVIACAST_HPP
#define _ARITHMETICOPERATORSVIACAST_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file arithmeticOperatorsViaCast.hpp

#include <metaprogramming/crtp.hpp>

namespace esnort
{
  DEFINE_CRTP_INHERITANCE_DISCRIMINER_FOR_TYPE(ArithmeticOprators);
  
  /// Provides the arithmetic operators via cast
  template <typename CastToExec,
	    typename ReturnedType>
  struct ArithmeticOperators :
    Crtp<ReturnedType,crtp::ArithmeticOpratorsDiscriminer>
  {
#define PROVIDE_POSTFIX_OPERATOR(OP)			\
    INLINE_FUNCTION constexpr HOST_DEVICE_ATTRIB	\
    ReturnedType& operator OP (int)			\
    {							\
      auto& This=this->crtp();				\
      							\
      ((CastToExec&)This) OP;				\
      							\
      return This;					\
    }
    
    PROVIDE_POSTFIX_OPERATOR(++);
    PROVIDE_POSTFIX_OPERATOR(--);
    
#undef PROVIDE_POSTFIX_OPERATOR
    
#define PROVIDE_OPERATOR(OP,RETURNED_TYPE)				\
    INLINE_FUNCTION constexpr HOST_DEVICE_ATTRIB			\
    RETURNED_TYPE operator OP(const ArithmeticOperators& oth) const	\
    {									\
      return ((CastToExec)(this->crtp())) OP ((CastToExec)oth.crtp());	\
    }
    
    PROVIDE_OPERATOR(+,ReturnedType);
    PROVIDE_OPERATOR(-,ReturnedType);
    PROVIDE_OPERATOR(*,ReturnedType);
    PROVIDE_OPERATOR(/,ReturnedType);
    PROVIDE_OPERATOR(%,ReturnedType);
    
    PROVIDE_OPERATOR(==,bool);
    PROVIDE_OPERATOR(!=,bool);
    PROVIDE_OPERATOR(<,bool);
    PROVIDE_OPERATOR(<=,bool);
    PROVIDE_OPERATOR(>,bool);
    PROVIDE_OPERATOR(>=,bool);
    
#undef PROVIDE_OPERATOR
    
#define PROVIDE_SELF_OPERATOR(OP)					\
    ReturnedType& operator OP ##=(const ArithmeticOperators& oth)	\
    {									\
      auto& This=this->crtp();						\
									\
      ((CastToExec&)this->crtp()) OP ## =(CastToExec)oth.crtp();	\
									\
      return This;							\
    }
    
    PROVIDE_SELF_OPERATOR(+);
    PROVIDE_SELF_OPERATOR(-);
    PROVIDE_SELF_OPERATOR(*);
    PROVIDE_SELF_OPERATOR(/);
    PROVIDE_SELF_OPERATOR(%);
    
#undef PROVIDE_SELF_OPERATOR
  };
}

#endif
