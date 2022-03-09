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
    /// Stored reference
    T& ref;
    
    /// Action to be called at destroy
    const DestructionAction destruct;
    
  public:
    
    /// Assign from other, for some reason does not pass through decay to T
    template <typename U>
    INLINE_FUNCTION const
    RefProxy& operator=(const U& oth)
    {
      ref=oth;
      
      return *this;
    }
    
    /// Implicit convert to T
    INLINE_FUNCTION
    operator auto&()
    {
      return ref;
    }
    
    /// Create from referencee and action
    RefProxy(T& value,
	     DestructionAction&& destruct) :
      ref(value),
      destruct(destruct)
    {
    }
    
    /// Destroy calling the action
    ~RefProxy()
    {
      destruct();
    }
  };
}

#endif
