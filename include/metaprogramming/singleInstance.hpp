#ifndef _SINGLE_INSTANCE_HPP
#define _SINGLE_INSTANCE_HPP

#ifdef HAVE_CONFIG_H
# include "config.hpp"
#endif

/// \file singleInstance.hpp
///
/// \brief Defines a simple class, with a counter wich enable only one
/// instance at the time

#include <debug/minimalCrash.hpp>
#include <metaprogramming/staticMemberWithInitializator.hpp>

namespace esnort
{
  /// Single Instance enforcer class
  ///
  /// To enforce a class to be instantiated at most once, inherit this
  /// class. Example:
  ///
  /// \code
  /// class A :
  ///   public SingleInstance<A>
  /// {
  /// };
  ///
  /// A a1; // ok
  /// A a2; // error!
  /// \endcode
  template <typename T> // Type from which to inherit, to distinguish each count
  class SingleInstance
  {
    
    PROVIDE_STATIC_MEMBER_WITH_INITIALIZATOR(int,count,0,Counter of instances);
    
    /// Check that no instance exists
    void crashIfInstancesExists()
    {
      if(count()!=0)
	MINIMAL_CRASH("Count is not zero: %d",count());
    }
    
  public:
    
    /// Constructor, checking no instance exists before
    SingleInstance()
    {
      crashIfInstancesExists();
      
      count()++;
    }
    
    /// Destructor, checking that no instance exists at the end
    ~SingleInstance()
    {
      count()--;
      
      crashIfInstancesExists();
    }
  };
}

#endif
