#ifndef _COMP_HPP
#define _COMP_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file comp.hpp
///
/// \brief Implements a tensor comp

#include <expr/compRwCl.hpp>

// #include <stdint.h>

// #include <resources/device.hpp>
// #include <metaprogramming/feature.hpp>
#include <metaprogramming/inline.hpp>
#include <metaprogramming/nonConstMethod.hpp>
// #include <metaprogramming/templateEnabler.hpp>
// //#include <metaprogramming/tuple.hpp>
#include <metaprogramming/arithmeticOperatorsViaCast.hpp>
#include <metaprogramming/typeConversion.hpp>
// #include <metaprogramming/unrolledFor.hpp>

namespace esnort
{
  /// Dynamic size
  constexpr int DYNAMIC=-1;
  
  template <typename T,
	    typename Index>
  struct Comp;
  
  template <RwCl _RC,
	    int _Which,
	    template <RwCl,int> typename _C,
	    typename _Index>
  struct Comp<_C<_RC,_Which>,_Index> :
    ArithmeticOperators<_Index,_C<_RC,_Which>>
  {
    /// Row or column
    static constexpr
    RwCl RC=
      _RC;
    
    /// Index of the component
    static constexpr
    int Which=
      _Which;
    
    /// Value type
    using Index=_Index;
    
    /// Component
    using C=_C<RC,Which>;
    
    /// Transposed component
    using Transp=_C<transpRwCl<RC>,Which>;
    
    /// Non column version
    using NonCln=
      _C<((RC==RwCl::ANY)?RwCl::ANY:RwCl::ROW),Which>;
    
    /// Value
    Index i;
    
    /// Define if this component is of matrix type (row or column)
    static constexpr bool isMatrix=
      RC!=RwCl::ANY;
    
    /// Check if the size is known at compile time
    static constexpr
    bool sizeIsKnownAtCompileTime=
      C::sizeAtCompileTime()!=DYNAMIC;
    
    /// Returns the size at compile time, with assert
    static constexpr Index sizeAtCompileTimeAssertingNotDynamic()
    {
      static_assert(sizeIsKnownAtCompileTime,"Size not known at compile time!");
      
      return C::sizeAtCompileTime();
    }
    
    /// Default constructor
    INLINE_FUNCTION HOST_DEVICE_ATTRIB constexpr
    Comp() : i(0)
    {
    }
    
    /// Init from value
    template <typename T=Index,
	      ENABLE_THIS_TEMPLATE_IF(isSafeNumericConversion<Index,T>)>
    INLINE_FUNCTION HOST_DEVICE_ATTRIB constexpr
    Comp(T&& i) : i(i)
    {
    }
    
    /// Assignment operator
    INLINE_FUNCTION HOST_DEVICE_ATTRIB constexpr
    Comp& operator=(const Index& oth) &
    {
      i=oth;
      
      return
	*this;
    }
    
    /// Assignment operator of a TensComp
    INLINE_FUNCTION HOST_DEVICE_ATTRIB constexpr
    Comp& operator=(const Comp& oth) &
    {
      return
	(*this)=oth.i;
    }
    
    /// Forbid assignement to a temporary
    Comp& operator=(const Comp& oth) && = delete;
    
    /// Convert to actual reference with const attribute
    INLINE_FUNCTION constexpr HOST_DEVICE_ATTRIB
    const Index& toPod() const
    {
      return i;
    }
    
    PROVIDE_ALSO_NON_CONST_METHOD_WITH_ATTRIB(toPod,HOST_DEVICE_ATTRIB);
    
#define PROVIDE_CAST_TO_VALUE(ATTRIB)					\
    /*! Convert to actual reference with or without const attribute */	\
    INLINE_FUNCTION HOST_DEVICE_ATTRIB constexpr			\
    explicit operator ATTRIB Index&() ATTRIB				\
    {									\
      return toPod();							\
    }
    
    PROVIDE_CAST_TO_VALUE(const);
    PROVIDE_CAST_TO_VALUE(/* non const */);
    
#undef PROVIDE_CAST_TO_VALUE
    
    /// Convert to actual reference with const attribute, to be removed
    INLINE_FUNCTION constexpr HOST_DEVICE_ATTRIB
    const Index& nastyConvert() const
    {
      return toPod();
    }
    
    PROVIDE_ALSO_NON_CONST_METHOD_WITH_ATTRIB(nastyConvert,HOST_DEVICE_ATTRIB);
    
    /// Transposed index
    INLINE_FUNCTION HOST_DEVICE_ATTRIB constexpr
    auto transp()
      const
    {
      return
	Transp{i};
    }
    
    /// Dagger index, alias for transposed
    INLINE_FUNCTION HOST_DEVICE_ATTRIB constexpr
    auto dag()
      const
    {
      return
	transp();
    }
    
  };

  #if 0
  
  /// Declare a component with no special feature
  ///
  /// The component has no row/column tag or index, so it can be
  /// included only once in a tensor
#define DECLARE_COMP(NAME,TYPE,SIZE)			\
  DECLARE_COMP_SIGNATURE(NAME,TYPE,SIZE);			\
  								\
  /*! NAME component */						\
  using NAME=							\
    TensorComp<NAME ## Signature,ANY,0>
  
  /// Declare a template component
  ///
  /// The component has no row/column tag or index, so it can be
  /// included only once in a tensor
#define DECLARE_TEMPLATED_COMP(NAME,SIZE)			\
  DECLARE_TEMPLATED_COMP_SIGNATURE(NAME,SIZE);		\
  								\
  /*! NAME component */						\
  template <typename T>						\
  using NAME=							\
    TensorComp<NAME ## Signature<T>,ANY,0>
  
  /// Declare a component which can be included more than once
  ///
  /// The component has a row/column tag, and an additional index, so
  /// it can be included twice in a tensor
#define DECLARE_ROW_OR_CLN_COMP(NAME,TYPE,SIZE)		\
  DECLARE_COMP_SIGNATURE(NAME,TYPE,SIZE);			\
  								\
  /*! NAME component */						\
  template <RwCl RC=ROW,					\
	    int Which=0>					\
  using NAME ## RC=TensorComp<NAME ## Signature,RC,Which>;	\
								\
  /*! Row kind of NAME component */				\
  using NAME ## Row=NAME ## RC<ROW,0>;				\
								\
  /*! Column kind of NAME component */				\
  using NAME ## Cln=NAME ## RC<CLN,0>;				\
  								\
  /*! Default NAME component is Row */				\
  using NAME=NAME ## Row/*;*/					\
  								\
    /*DECLARE_COMP_FACTORY(FACTORY ## Row,NAME ## Row);*/	\
								\
    /*DECLARE_COMP_FACTORY(FACTORY ## Cln,NAME ## Cln);*/	\
								\
    /*DECLARE_COMP_FACTORY(FACTORY,NAME)*/
  
  /////////////////////////////////////////////////////////////////
  
#define FOR_COMP_VALUES_IN_RANGE(TYPE,NAME,MIN,MAX)	\
  for(TYPE NAME=MIN;NAME<MAX;NAME++)
  
#define UNROLL_FOR_COMP_VALUES_IN_RANGE(TYPE,NAME,MIN,MAX)	\
  UNROLL_FOR(TYPE,NAME,MIN,MAX)
  
#define FOR_ALL_COMP_VALUES_STARTING_AT(TYPE,NAME,MIN)			\
  FOR_COMP_VALUES_IN_RANGE(TYPE,NAME,MIN,TYPE::sizeAtCompileTimeAssertingNotDynamic())
  
#define UNROLL_FOR_ALL_COMP_VALUES_STARTING_AT(TYPE,NAME,MIN)		\
  UNROLL_FOR_COMP_VALUES_IN_RANGE(TYPE,NAME,MIN,TYPE::sizeAtCompileTimeAssertingNotDynamic())
  
#define FOR_ALL_COMP_VALUES(TYPE,NAME)			\
  FOR_ALL_COMP_VALUES_STARTING_AT(TYPE,NAME,0)
  
#define UNROLL_FOR_ALL_COMP_VALUES(TYPE,NAME)		\
  UNROLL_FOR_ALL_COMP_VALUES_STARTING_AT(TYPE,NAME,0)
  
  /////////////////////////////////////////////////////////////////
  
  /// Collection of components
  template <typename...Tc>
  using TensorComps=
    std::tuple<Tc...>;
  
  /// Alias to make it easier to understand tensor instantiation
  template <typename...Tc>
  using OfComps=
    TensorComps<Tc...>;
  
  /// Returns the number of components of a tensComp
  template <typename T>
  constexpr int nOfComps=
    std::tuple_size<typename T::Comps>::value;
  
  namespace impl
  {
    /// Provides the result of filtering from a list of components the Row or Column
    ///
    /// Forward definition
    template <RwCl F,
	      typename TC>
    struct _TensorCompsFilterRwCl;
    
    /// Cannot use directly the TupleFilter, because of some template template limitation
    template <RwCl F,
	      typename...Tc>
    struct _TensorCompsFilterRwCl<F,TensorComps<Tc...>>
    {
      /// Predicate to filter
      ///
      /// Forward definition
      template <typename T>
      struct Filter;
      
      /// Predicate to filter
      template <typename S,
		RwCl RC,
		int Which>
      struct Filter<TensorComp<S,RC,Which>>
      {
	/// Predicate result, counting whether the type match
	static constexpr
	bool value=
	  (RC==F);
      };
      
      /// Returned type
      typedef TupleFilter<Filter,TensorComps<Tc...>> type;
    };
  }
  
  /// Filter all Row components
  template <typename TC>
  using TensorCompsFilterRow=
    typename impl::_TensorCompsFilterRwCl<RwCl::ROW,TC>::type;
  
  /// Filter all Col components
  template <typename TC>
  using TensorCompsFilterCln=
    typename impl::_TensorCompsFilterRwCl<RwCl::CLN,TC>::type;
  
  /// Filter all Any components
  template <typename TC>
  using TensorCompsFilterAny=
    typename impl::_TensorCompsFilterRwCl<RwCl::ANY,TC>::type;
  
  /// Gets the value of the dynamic components of a tensComps
  template <typename TC>
  constexpr decltype(auto) getDynamicCompsOfTensorComps(TC&& tc)
  {
    return
      tupleFilter<predicate::CompSizeIsKnownAtCompileTime<false>::t>(std::forward<TC>(tc));
  }
  
  /// Gets the dynamic component types of a TensorComps
  template <typename TC>
  using GetDynamicCompsOfTensorComps=
    decltype(getDynamicCompsOfTensorComps(TC{}));
  
  /// Gets the fixed size components of a tensComps
  template <typename TC>
  constexpr decltype(auto) getFixedSizeCompsOfTensorComps(TC&& tc)
  {
    return tupleFilter<predicate::CompSizeIsKnownAtCompileTime<true>::t>(std::forward<TC>(tc));
  }
  
  /// Gets the fixed size component types of a TensorComps
  template <typename TC>
  using GetFixedSizeCompsOfTensorComps=
    decltype(getFixedSizeCompsOfTensorComps(TC{}));
  
  /////////////////////////////////////////////////////////////////
  
  namespace impl
  {
    /// Transposes a list of components
    ///
    /// Actual implementation, forward declaration
    template <typename TC>
    struct _TransposeTensorComps;
    
    /// Transposes a list of components, considering the components as matrix
    ///
    /// Actual implementation
    template <typename...TC>
    struct _TransposeTensorComps<TensorComps<TC...>>
    {
      /// Resulting type
      using type=
	TensorComps<typename TC::Transp...>;
    };
  }
  
  /// Transposes a list of components
  template <typename TC>
  using TransposeTensorComps=
    typename impl::_TransposeTensorComps<TC>::type;
  
  /////////////////////////////////////////////////////////////////
  
  namespace impl
  {
    /// Transposes a list of components, considering the components as matrix
    ///
    /// Actual implementation, forward declaration
    template <typename TC>
    struct _TransposeMatrixTensorComps;
    
    /// Transposes a list of components, considering the components as matrix
    ///
    /// Actual implementation
    template <typename...TC>
    struct _TransposeMatrixTensorComps<TensorComps<TC...>>
    {
      /// Returns a given components, or its transposed if it is missing
      template <typename C,
		typename TranspC=typename C::Transp>
      using ConditionallyTransposeComp=
	std::conditional_t<tupleHasType<TensorComps<TC...>,TranspC,1>,C,TranspC>;
      
      /// Resulting type
      using type=
	TensorComps<ConditionallyTransposeComp<TC>...>;
    };
  }
  
  /// Transposes a list of components, considering the components as matrix
  ///
  /// - If a component is not of ROW/CLN case, it is left unchanged
  /// - If a ROW/CLN component is matched with a CLN/ROW one, it is left unchanged
  /// - If a ROW/CLN component is not matched, it is transposed
  ///
  /// \example
  ///
  /// using T=TensorComps<Complex,ColorRow,ColorCln,SpinRow>
  /// using U=TransposeTensorcomps<T>; //TensorComps<Complex,ColorRow,ColorCln,SpinCln>
  template <typename TC>
  using TransposeMatrixTensorComps=
    typename impl::_TransposeMatrixTensorComps<TC>::type;
  
  /////////////////////////////////////////////////////////////////
  
  namespace internal
  {
    /// Independent components of a set of TensorComponents
    ///
    /// Internal implementation, forward declararation
    template <typename Tc>
    struct _IndependentComps;
    
    /// Independent components of a set of TensorComps
    ///
    /// Internal implementation
    template <typename...Tc>
    struct _IndependentComps<TensorComps<Tc...>>
    {
      /// Returned type
      using type=
	UniqueTuple<typename Tc::NonCln...>;
    };
  }
  
  /// Independent components of a set of TensorComps
  ///
  /// Only Row version is returned, even if only Col is present
  ///
  /// \example
  ///
  /// IndependentComps<TensorComps<ColorRow,ColorCln>,TensorComps<SpinCln>>; /// TensorComps<ColorRow,SpinRow>
  ///
  template <typename...TP>
  using IndependentComps=
    typename internal::_IndependentComps<TupleCat<TP...>>::type;
  
#endif
}

#endif
