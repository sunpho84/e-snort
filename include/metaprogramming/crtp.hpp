#ifndef _CRTP_HPP
#define _CRTP_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

#include <metaprogramming/inline.hpp>

namespace esnort
{
  /// Implements the CRTP pattern
  template <typename T,
	    typename Discriminer>
  struct Crtp
  {
#define PROVIDE_CRTP(ATTRIB)				\
    /*! Crtp access the type */				\
    CUDA_HOST CUDA_DEVICE INLINE_FUNCTION constexpr	\
    ATTRIB T& crtp() ATTRIB				\
    {							\
      return						\
	*static_cast<ATTRIB T*>(this);			\
    }
    
    PROVIDE_CRTP(const);
    
    PROVIDE_CRTP(/* not const*/ );
    
#undef PROVIDE_CRTP
  };
  
#define DEFINE_CRTP_INHERITANCE_DISCRIMINER_FOR_TYPE(NAME)	\
  namespace crtp						\
  {								\
    struct NAME ## Discriminer{};				\
  }
  
}

#endif
