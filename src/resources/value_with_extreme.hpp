#ifndef _VALUE_WITH_EXTREME_HPP
#define _VALUE_WITH_EXTREME_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

#include <limits>

#include <resources/refProxy.hpp>
#include <resources/selfop.hpp>

namespace esnort
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
    const T& extreme() const
    {
      return extr;
    }
    
    /// Reset to standard value
    template <typename V=T>
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
    explicit ValWithExtreme(const V& init=0)
    {
      reset(init);
    }
    
    /// Cast to const value reference
    const T& operator()() const
    {
      return val;
    }
    
    /// Implicit cast to const value reference
    operator const T&() const
    {
      return (*this)();
    }
    
    /// Cast to a proxy
    auto operator()()
    {
      return RefProxy(val,[this](){updateExtreme();});
    }
    
    /// Implicit cast to a proxy
    operator auto()
    {
      return RefProxy(val,[this](){updateExtreme();});
    }
  };
  
  /// Class to keep a value and its maximum
  template <typename T>
  using ValWithMax=ValWithExtreme<T,Extreme::MAXIMUM>;
}

#endif
