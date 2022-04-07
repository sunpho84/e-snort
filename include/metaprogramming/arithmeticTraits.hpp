#ifndef _ARITHMETICTRAITS_HPP
#define _ARITHMETICTRAITS_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file arithmeticTraits.hpp

#if ENABLE_SIMD
# include <immintrin.h>
#endif

#include <ios/logger.hpp>
#include <metaprogramming/inline.hpp>

namespace esnort
{
  /// Struct to hold info on arithmetic types
  
  /// Forward declaration
  template <typename T>
  struct ArithmeticTypeTraits;
  
#define PROVIDE_ZERO(PREFIX,TYPE,OP)		\
  static INLINE_FUNCTION HOST_DEVICE_ATTRIB	\
  PREFIX TYPE zero()				\
  {						\
    return OP;					\
  }						\
  
#define PROVIDE_SUBSUMASSIGN_THE_PROD(SUBSUM,PREFIX,TYPE,OP)	\
  static INLINE_FUNCTION HOST_DEVICE_ATTRIB			\
  PREFIX TYPE SUBSUM ## AssignTheProd(TYPE& out,		\
				      const TYPE& f1,		\
				      const TYPE& f2)		\
  {								\
    return							\
      OP;							\
  }
  
#define PROVIDE_SUMASSIGN_THE_PROD(PREFIX,TYPE,OP)	\
  PROVIDE_SUBSUMASSIGN_THE_PROD(sum,PREFIX,TYPE,OP)
  
#define PROVIDE_SUBASSIGN_THE_PROD(PREFIX,TYPE,OP)	\
  PROVIDE_SUBSUMASSIGN_THE_PROD(sub,PREFIX,TYPE,OP)
  
#define PROVIDE_PRINTER(TYPE)						\
  INLINE_FUNCTION							\
  Logger::LoggerLine& operator<<(Logger::LoggerLine& os,TYPE m)		\
  {									\
    constexpr int n=sizeof(m)/sizeof(m[0]);				\
									\
    os<<"("<<m[0];							\
    for(int i=1;i<n;i++)						\
      os<<","<<m[i];							\
    os<<")";								\
									\
    return os;								\
  }
  
  /// long int
  template<>
  struct ArithmeticTypeTraits<long int>
  {
    PROVIDE_ZERO(constexpr,long int,0);
    PROVIDE_SUMASSIGN_THE_PROD(constexpr,long int,out+=f1*f2);
    PROVIDE_SUBASSIGN_THE_PROD(constexpr,long int,out-=f1*f2);
  };
  
  /// Double
  template<>
  struct ArithmeticTypeTraits<double>
  {
    PROVIDE_ZERO(constexpr,double,0.0);
    PROVIDE_SUMASSIGN_THE_PROD(constexpr,double,out+=f1*f2);
    PROVIDE_SUBASSIGN_THE_PROD(constexpr,double,out-=f1*f2);
  };
  
  /// Float
  template<>
  struct ArithmeticTypeTraits<float>
  {
    PROVIDE_ZERO(constexpr,float,0.0f);
    PROVIDE_SUMASSIGN_THE_PROD(constexpr,float,out+=f1*f2);
    PROVIDE_SUBASSIGN_THE_PROD(constexpr,float,out-=f1*f2);
  };
  
#if ENABLE_SIMD
  
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"

#if ENABLE_MMX
  
  /// Mmx double vector
  template<>
  struct ArithmeticTypeTraits<__m128d>
  {
    PROVIDE_ZERO(,__m128d,_mm_setzero_pd());
    PROVIDE_SUMASSIGN_THE_PROD(,__m128d,out=_mm_fmadd_pd(f1,f2,out));
    PROVIDE_SUBASSIGN_THE_PROD(,__m128d,out=_mm_fnmadd_pd(f1,f2,out));
  };
  
  PROVIDE_PRINTER(__m128d);
  
  /// Mmx float vector
  template<>
  struct ArithmeticTypeTraits<__m128>
  {
    PROVIDE_ZERO(,__m128,_mm_setzero_ps());
    PROVIDE_SUMASSIGN_THE_PROD(,__m128,out=_mm_fmadd_ps(f1,f2,out));
    PROVIDE_SUBASSIGN_THE_PROD(,__m128,out=_mm_fnmadd_ps(f1,f2,out));
  };
  
  PROVIDE_PRINTER(__m128);
  
#endif
  
#if ENABLE_AVX
  
  /// Avx double vector
  template<>
  struct ArithmeticTypeTraits<__m256d>
  {
    PROVIDE_ZERO(,__m256d,_mm256_setzero_pd());
    PROVIDE_SUMASSIGN_THE_PROD(,__m256d,out=_mm256_fmadd_pd(f1,f2,out));
    PROVIDE_SUBASSIGN_THE_PROD(,__m256d,out=_mm256_fnmadd_pd(f1,f2,out));
  };
  
  PROVIDE_PRINTER(__m256d);
  
  /// Avx float vector
  template<>
  struct ArithmeticTypeTraits<__m256>
  {
    PROVIDE_ZERO(,__m256,_mm256_setzero_ps());
    PROVIDE_SUMASSIGN_THE_PROD(,__m256,out=_mm256_fmadd_ps(f1,f2,out));
    PROVIDE_SUBASSIGN_THE_PROD(,__m256,out=_mm256_fnmadd_ps(f1,f2,out));
  };
  
  PROVIDE_PRINTER(__m256);
  
#endif
  
#if ENABLE_AVX512
  
  /// Avx 512 double vector
  template<>
  struct ArithmeticTypeTraits<__m512d>
  {
    PROVIDE_ZERO(__m512d,_mm512_setzero_pd());
    PROVIDE_SUMASSIGN_THE_PROD(,__m512d,out=_mm512_fmadd_pd(f1,f2,out));
    PROVIDE_SUBASSIGN_THE_PROD(,__m512d,out=_mm512_fnmadd_pd(f1,f2,out));
  };
  
  PROVIDE_PRINTER(__m512d);
  
  /// Avx 512 float vector
  template<>
  struct ArithmeticTypeTraits<__m512>
  {
    PROVIDE_ZERO(__m512,_mm512_setzero_ps());
    PROVIDE_SUMASSIGN_THE_PROD(,__m512,out=_mm512_fmadd_ps(f1,f2,out));
    PROVIDE_SUBASSIGN_THE_PROD(,__m512,out=_mm512_fnmadd_ps(f1,f2,out));
  };
  
  PROVIDE_PRINTER(__m512);
  
#endif
  
#pragma GCC diagnostic pop
  
#endif
  
#undef PROVIDE_ZERO
#undef PROVIDE_SUMASSIGN_THE_PROD
#undef PROVIDE_SUBASSIGN_THE_PROD
#undef PROVIDE_SUBSUMASSIGN_THE_PROD
#undef PROVIDE_PRINTER
  
  /// Returns the zero for the passed type
  template <typename T>
  constexpr INLINE_FUNCTION HOST_DEVICE_ATTRIB
  T zero()
  {
    return ArithmeticTypeTraits<T>::zero();
  }
  
  /////////////////////////////////////////////////////////////////
  
  /// Returns the result of summing a product
  template <typename T>
  constexpr INLINE_FUNCTION HOST_DEVICE_ATTRIB
  auto sumAssignTheProd(T& a,const T& f1,const T& f2)
  {
    return ArithmeticTypeTraits<T>::sumAssignTheProd(a,f1,f2);
  }
  
  /// Returns the result of subracting a product
  template <typename T>
  constexpr INLINE_FUNCTION HOST_DEVICE_ATTRIB
  auto subAssignTheProd(T& a,const T& f1,const T& f2)
  {
    return ArithmeticTypeTraits<T>::subAssignTheProd(a,f1,f2);
  }
  
  /////////////////////////////////////////////////////////////////
  
  /// Returns the result of summing a product
  template <typename T1,
	    typename T2,
	    typename T3>
  constexpr INLINE_FUNCTION HOST_DEVICE_ATTRIB
  auto sumAssignTheProd(T1& a,const T2& f1,const T3& f2)
  {
    return a+=f1*f2;
  }
  
  /// Returns the result of subracting a product
  template <typename T1,
	    typename T2,
	    typename T3>
  constexpr INLINE_FUNCTION HOST_DEVICE_ATTRIB
  auto subAssignTheProd(T1& a,const T2& f1,const T3& f2)
  {
    return a-=f1*f2;
  }
}

#endif
