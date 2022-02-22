#ifndef _NONCONSTMETHOD_HPP
#define _NONCONSTMETHOD_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

#include <utility>

#include <metaprogramming/inline.hpp>

namespace esnort
{
  namespace details
  {
    template <bool IsConst,
	      typename T>
    using _ConditionalConst=
      std::conditional_t<IsConst,const T,T>;
  }
  
  template <bool IsConst,
	    typename T>
  using ConditionalConst=
    details::_ConditionalConst<IsConst,std::remove_const_t<T>>;
  
  /////////////////////////////////////////////////////////////////
  
  /// Returns true if T is a const lvalue reference
  template <typename T>
  constexpr bool is_const_lvalue_reference_v=
    std::is_lvalue_reference<T>::value and std::is_const<std::remove_reference_t<T>>::value;

  template <typename T>
  CUDA_HOST_DEVICE INLINE_FUNCTION
  T* remove_const_if_ref_or_pointer(const T* a)
  {
    return (T*)a;
  }
  
  template <typename T>
  CUDA_HOST_DEVICE INLINE_FUNCTION
  decltype(auto) remove_const_if_ref_or_pointer(T&& a)
  {
    return a;
  }
  
  template <typename T>
  CUDA_HOST_DEVICE INLINE_FUNCTION
  T& remove_const_if_ref_or_pointer(const T& a)
  {
    return (T&)a;
  }
  
  /// Returns the type without "const" attribute if it is a reference
  template <typename T>
  INLINE_FUNCTION constexpr CUDA_HOST_DEVICE
  decltype(auto) remove_const_if_ref(T&& t)
  {
    using Tv=std::remove_const_t<std::remove_reference_t<T>>;
    
    return (std::conditional_t<is_const_lvalue_reference_v<T>,Tv&,Tv>)t;
  }
  
  /// Provides also a non-const version of the method \c NAME
  ///
  /// See
  /// https://stackoverflow.com/questions/123758/how-do-i-remove-code-duplication-between-similar-const-and-non-const-member-func
  /// A const method NAME must be already present Example
  ///
  /// \code
  // class nissa
  /// {
  ///   double e{0};
  ///
  /// public:
  ///
  ///   const double& get() const
  ///   {
  ///     return e;
  ///   }
  ///
  ///   PROVIDE_ALSO_NON_CONST_METHOD(get);
  /// };
  /// \endcode
#define PROVIDE_ALSO_NON_CONST_METHOD_WITH_ATTRIB(NAME,ATTRIB)		\
  /*! Overload the \c NAME const method passing all args             */ \
  template <typename...Ts> /* Type of all arguments                  */	\
  ATTRIB								\
  decltype(auto) NAME(Ts&&...ts) /*!< Arguments                      */ \
  {									\
    return remove_const_if_ref_or_pointer(std::as_const(*this).NAME(std::forward<Ts>(ts)...)); \
  }
  
#define PROVIDE_ALSO_NON_CONST_METHOD(NAME)		\
  PROVIDE_ALSO_NON_CONST_METHOD_WITH_ATTRIB(NAME,)
}

#endif
