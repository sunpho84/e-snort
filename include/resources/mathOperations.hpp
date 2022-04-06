#ifndef _MATHOPERATIONS_HPP
#define _MATHOPERATIONS_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file routines/mathOperations.hpp

#include <metaprogramming/inline.hpp>

namespace esnort
{
  /// Modulo operator, based on remainder operator %
  ///
  /// Valid on negative and positive numbers
  ///
  /// Example:
  ///
  /// \code
  /// safeModulo(5,3);  // 2
  /// safeModulo(-2,3); // 1
  /// safeModulo(-3,3); // 0
  /// \endcode
  ///
  template <typename T>
  constexpr INLINE_FUNCTION HOST_DEVICE_ATTRIB
  T safeModulo(const T& val,   ///< Value of which to take the modulo
	       const T& mod)   ///< Modulo
  {
    /// Remainder
    const T r=
      val%mod;
    
    return
      (val<0 and r!=0)?
      T(r+mod):
      r;
  }
  
}

#endif
