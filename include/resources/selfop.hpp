#ifndef _SELFOP_HPP
#define _SELFOP_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

#include <metaprogramming/crtp.hpp>

#include <type_traits>

/// \file selfop.hpp
///
/// \brief Provides a number of self-assignment operator

namespace esnort
{
  DEFINE_CRTP_INHERITANCE_DISCRIMINER_FOR_TYPE(SelfOp);
  
  template <typename T>
  struct SelfOp :
    Crtp<T,crtp::SelfOpDiscriminer>
  {
#define PROVIDE_SELFOP(OP)			\
    template <typename U>			\
    T& operator OP(const U& rhs)		\
    {						\
      auto& lhs=this->crtp();			\
    						\
      lhs() OP rhs;				\
						\
      return lhs;				\
    }
    
    PROVIDE_SELFOP(=)
    PROVIDE_SELFOP(+=)
    PROVIDE_SELFOP(-=)
    PROVIDE_SELFOP(*=)
    PROVIDE_SELFOP(/=)
    
#undef PROVIDE_SELFOP
  };
}

#endif
