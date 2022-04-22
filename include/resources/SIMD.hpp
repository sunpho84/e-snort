#ifndef _SIMD_HPP
#define _SIMD_HPP

#ifdef HAVE_CONFIG_H
# include "config.hpp"
#endif

/// \file SIMD.hpp

#include <metaprogramming/inline.hpp>

#if ENABLE_SIMD
# include <immintrin.h>
#endif

#include <type_traits>
#include <utility>

#include <tuples/tupleCat.hpp>

constexpr bool haveMmxInstructions=
#if ENABLE_SIMD and ENABLE_MMX
	    true
#else
	    false
#endif
	    ;

constexpr bool haveAvxInstructions=
#if ENABLE_SIMD and ENABLE_AVX
	    true
#else
	    false
#endif
	    ;

constexpr bool haveAvx512Instructions=
#if ENABLE_SIMD and ENABLE_AVX512
	    true
#else
	    false
#endif
	    ;

namespace grill
{
  // using m512d=SIMDVector<double,8>;
  // using m512=SIMDVector<float,16>;
  // using m256d=SIMDVector<double,4>;
  // using m256=SIMDVector<float,8>;
  // using m128d=SIMDVector<double,2>;
  // using m128=SIMDVector<float,4>;
  
  namespace internal
  {
#define CASE(SIZE_PER_REAL,REG_SIZE,SUFF,ELSE)				\
    if constexpr((NReals*SIZE_PER_REAL)%REG_SIZE ==0)			\
      return __m ## REG_SIZE ## SUFF{};					\
    else								\
      ELSE								\
	
#define CASE_ELSE(SIZE_PER_REAL,REG_SIZE,SUFF,ELSE)	\
    ELSE
    
#if ENABLE_SIMD and ENABLE_AVX512
# define CASE_AVX512 CASE
#else
# define CASE_AVX512 CASE_ELSE
#endif
    
#if ENABLE_SIMD and ENABLE_AVX
# define CASE_AVX CASE
#else
# define CASE_AVX CASE_ELSE
#endif

#if ENABLE_SIMD and ENABLE_MMX
# define CASE_MMX CASE
#else
# define CASE_MMX CASE_ELSE
#endif
    
#define CASES(TYPE,SIZE_PER_EL,SUFF,ELSE)				\
    if constexpr(std::is_same_v<F,TYPE>)				\
      {									\
	CASE_AVX512(SIZE_PER_EL,512,SUFF,				\
		    CASE_AVX(SIZE_PER_EL,256,SUFF,			\
			     CASE_MMX(SIZE_PER_EL,128,SUFF,)));		\
      }									\
    else								\
      ELSE;
    
    /// Provides the largest SIMD type supporting a vector of type F and size Size
    ///
    /// Internal implementation
    template <typename F,
	      int NReals>
    constexpr auto _simdTypeProvider()
    {
      CASES(float,32,,
	    CASES(double,64,d,));
    }
    
#undef CASES
#undef CASE_MMX
#undef CASE_AVX
#undef CASE_AVX512
#undef CASE
#undef CASE_ELSE
  }
  
  /// Provides the largest SIMD type supporting a vector of type F and size Size
  template <typename F,
	    int Size>
  struct SimdOfTypeTraits
  {
    using type=decltype(internal::_simdTypeProvider<F,Size>());
    
    /// Check if there is a SIMD type for the asked type and size
    static constexpr bool canSimdify()
    {
      return Size!=0 and not std::is_same_v<void,type>;
    }
    
    /// Returns the number of non-simdified elements
    static constexpr int nNonSimdifiedElements()
    {
      if constexpr(canSimdify())
	return Size*sizeof(F)/sizeof(type);
      else
	return Size;
    }
  };
  
  constexpr int maxAvailableSimdSize=
#if   ENABLE_AVX512
	      16
#elif ENABLE_AVX
	      8
#elif ENABLE_MMX
	      4
#else
	      1
#endif
	      ;
}

#endif
