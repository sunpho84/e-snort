#ifndef _REFPROXY_HPP
#define _REFPROXY_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

#include <utility>

#include <metaprogramming/inline.hpp>

namespace esnort
{
  template <typename T,
	    typename DestructionAction>
  class RefProxy
  {
    T& value;
    
    const DestructionAction destruct;
    
  public:
    
    template <typename U>
    INLINE_FUNCTION const
    RefProxy& operator=(const U& oth)
    {
      value=oth;
      
      return *this;
    }
    
    operator T&()
    {
      return value;
    }
    
    RefProxy(T& value,
	     DestructionAction&& destruct) :
      value(value),
      destruct(destruct)
    {
    }
    
    ~RefProxy()
    {
      destruct();
    }
  };
}

#endif
