#ifndef _ARITHMETICTRAITS_HPP
#define _ARITHMETICTRAITS_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file arithmeticTraits.hpp

#if ENABLE_SIMD
# include <immintrin.h>
#endif

#include <metaprogramming/inline.hpp>

namespace esnort
{
  /// Struct to hold info on arithmetic types
  
  /// Forward declaration
  template <typename T>
  struct ArithmeticTypeTraits;
  
#define PROVIDE_ZERO(TYPE,OP)			\
  static INLINE_FUNCTION HOST_DEVICE_ATTRIB	\
  TYPE zero()					\
  {						\
    return OP;					\
  }						\
  
  /// Double
  template<>
  struct ArithmeticTypeTraits<double>
  {
    PROVIDE_ZERO(constexpr double,0.0);
  };
  
  /// Float
  template<>
  struct ArithmeticTypeTraits<float>
  {
    PROVIDE_ZERO(constexpr float,0.0f);
  };
  
#if ENABLE_AVX
  
  /// Avx vector
  template<>
  struct ArithmeticTypeTraits<__m256d>
  {
    PROVIDE_ZERO(__m256d,_mm256_setzero_pd());
  };
  
  /// Avx vector
  template<>
  struct ArithmeticTypeTraits<__m256>
  {
    PROVIDE_ZERO(__m256,_mm256_setzero_ps());
  };
  
#endif
  
#if ENABLE_AVX512
  
  /// Avx 512 vector
  template<>
  struct ArithmeticTypeTraits<__m512d>
  {
    PROVIDE_ZERO(__m512d,_mm512_setzero_pd());
  };
  
  /// Avx 512 vector
  template<>
  struct ArithmeticTypeTraits<__m512>
  {
    PROVIDE_ZERO(__m512,_mm512_setzero_ps());
  };
  
#endif
  
  /// Returns the zero for the passed type
  template <typename T>
  constexpr INLINE_FUNCTION HOST_DEVICE_ATTRIB
  T zero()
  {
    return ArithmeticTypeTraits<T>::zero();
  }
}

#endif
