#ifndef _VALUE_WITH_EXTREME_HPP
#define _VALUE_WITH_EXTREME_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file valueWithExtreme.hpp

#include <limits>

#include <resources/refProxy.hpp>
#include <resources/selfop.hpp>

namespace grill
{
  /// Possible extreme types
  enum class Extreme{MINIMUM,MAXIMUM};
  
  /// Class which keeps track of extreme values of a given type
  template <typename T,
	    Extreme E>
  class ValWithExtreme :
    public SelfOp<ValWithExtreme<T,E>>
  {
    /// Stored value
    T val;
    
    /// Extreme value
    T extr;
    
    /// Update the extreme
    ValWithExtreme& updateExtreme()
    {
      /// Result of whether it's extreme or not
      bool is;
      
      switch(E)
	{
	case Extreme::MINIMUM:
	  is=(val<extr);
	  break;
	case Extreme::MAXIMUM:
	  is=(val>extr);
	  break;
	}
      
      if(is)
	extr=val;
      
      return *this;
    }
    
  public:
    
    using SelfOp<ValWithExtreme<T,E>>::operator=;
    
    /// Retrurn extreme value
    INLINE_FUNCTION
    const T& extreme() const
    {
      return extr;
    }
    
    /// Reset to standard value
    template <typename V=T>
    INLINE_FUNCTION
    void reset(const V& init)
    {
      switch(E)
	{
	case Extreme::MINIMUM:
	  extr=val=init;
	  break;
	case Extreme::MAXIMUM:
	  extr=val=init;
	  break;
	}
    }
    
    /// Constructor
    template <typename V=T>
    INLINE_FUNCTION
    explicit ValWithExtreme(const V& init=0)
    {
      reset(init);
    }
    
    /// Cast to const value reference
    INLINE_FUNCTION
    const T& operator()() const
    {
      return val;
    }
    
    /// Implicit cast to const value reference
    INLINE_FUNCTION
    operator const T&() const
    {
      return (*this)();
    }
    
    /// Cast to a proxy
    INLINE_FUNCTION
    auto operator()()
    {
      return RefProxy(val,[this]() INLINE_ATTRIBUTE {updateExtreme();});
    }
    
    // Causing crash before gcc11, try changing the lambda maybe
    // /// Implicit cast to a proxy
    // INLINE_FUNCTION
    // operator auto()
    // {
    //   return RefProxy(val,[this]() INLINE_ATTRIBUTE {updateExtreme();});
    // }
  };
  
  /// Class to keep a value and its maximum
  template <typename T>
  using ValWithMax=ValWithExtreme<T,Extreme::MAXIMUM>;
}

#endif
