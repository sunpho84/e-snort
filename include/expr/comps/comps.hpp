#ifndef _COMPS_HPP
#define _COMPS_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file expr/comps/comps.hpp

#include <tuple>

#include <expr/comps/comp.hpp>
#include <metaprogramming/unrolledFor.hpp>
#include <resources/SIMD.hpp>
#include <tuples/tupleCommonTypes.hpp>
#include <tuples/tupleExecForAllTypes.hpp>
#include <tuples/tupleHasType.hpp>
#include <tuples/tupleReplaceType.hpp>

namespace esnort
{
  /// Collection of components
  template <typename...Tc>
  using CompsList=
    std::tuple<Tc...>;
  
  /// Alias to make it easier to understand tensor instantiation
  template <typename...Tc>
  using OfComps=
    CompsList<Tc...>;
  
  /////////////////////////////////////////////////////////////////
  
  namespace internal
  {
    template <typename Tp>
    struct _LastComp;
    
    template <typename...Tp>
    struct _LastComp<OfComps<Tp...>>
    {
      static constexpr int nComps=sizeof...(Tp);
      
      static constexpr auto _lastCompTypeProvider()
      {
	if constexpr(nComps>0)
	  return std::tuple_element_t<nComps-1,std::tuple<Tp...>>{};
      }
      
      using type=
	decltype(_lastCompTypeProvider());
    };
  }
  
  /// Provide last component of a tuple
  template <typename Tp>
  using LastComp=typename internal::_LastComp<Tp>::type;
  
  /////////////////////////////////////////////////////////////////
  
  template <typename Index,
	    int Size>
  struct NonSimdifiedComp :
    Comp<compFeat::IsTransposable::FALSE,
	 Index,
	 NonSimdifiedComp<Index,Size>>
  {
    using Base=Comp<compFeat::IsTransposable::FALSE,
		    Index,
		    NonSimdifiedComp<Index,Size>>;
    
    using Base::Base;
    
    /// Size at compile time
    static constexpr int sizeAtCompileTime=Size;
  };
  
  /// Returns whether the last component can simdify
  ///
  /// Forward declaration
  template <typename Tp,
	    typename F>
  struct CompsListSimdifiableTraits;
  
  /// Empty comps
  template <typename F>
  struct CompsListSimdifiableTraits<CompsList<>,F>
  {
    using LastComp=void;
  };
  
  /// Returns whether the last component can simdify
  template <typename Head,
	    typename...Tail,
	    typename F>
  struct CompsListSimdifiableTraits<CompsList<Head,Tail...>,F>
  {
    static constexpr int nComps=1+sizeof...(Tail);
    
    using LastComp=esnort::LastComp<CompsList<Head,Tail...>>;
    
    static constexpr int _lastCompSizeProvider()
    {
      if constexpr(nComps>0)
	return LastComp::sizeAtCompileTime;
      else
	return 0;
    }
    
    static constexpr int lastCompSize=
      _lastCompSizeProvider();
    
    using Traits=
      SimdOfTypeTraits<F,lastCompSize>;
    
    static constexpr bool canSimdify=
      Traits::canSimdify();
    
    using SimdFund=
      typename Traits::type;
    
    static constexpr int nNonSimdifiedElements=
      Traits::nNonSimdifiedElements();
    
    using Comps=
      TupleReplaceType<CompsList<Head,Tail...>,LastComp,NonSimdifiedComp<typename LastComp::Index,nNonSimdifiedElements>>;
  };
  
  /////////////////////////////////////////////////////////////////
  
  /// Determine whether the component list is transposible
  ///
  /// Default case
  template <typename Tp>
  constexpr bool compsAreTransposable=false;
  
  /// Determine whether the component list is transposible
  template <typename...C>
  inline
  constexpr bool compsAreTransposable<CompsList<C...>> =
    (C::isTransposable||...);
  
  /////////////////////////////////////////////////////////////////
  
  /////////////////////////////////////////////////////////////////
  
  namespace impl
  {
    /// Transposes a list of components, considering the components as matrix
    ///
    /// Actual implementation, forward declaration
    template <typename TC>
    struct _TranspMatrixTensorComps;
    
    /// Transposes a list of components, considering the components as matrix
    ///
    /// Actual implementation
    template <typename...TC>
    struct _TranspMatrixTensorComps<CompsList<TC...>>
    {
      /// Returns a given components, or its transposed if it is missing
      template <typename C,
		typename TranspC=typename C::Transp>
      using ConditionallyTranspComp=
	std::conditional_t<tupleHasType<CompsList<TC...>,TranspC,1>,C,TranspC>;
      
      /// Resulting type
      using type=
	CompsList<ConditionallyTranspComp<TC>...>;
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
  using TranspMatrixTensorComps=
    typename impl::_TranspMatrixTensorComps<TC>::type;
  
  /////////////////////////////////////////////////////////////////
  
  /// Combine the dynamic components of a tuple of dynamic comps, filling with each occurrence
  template <typename DcsOut,
	    typename..._DcsIn>
  HOST_DEVICE_ATTRIB INLINE_FUNCTION constexpr
  auto dynamicCompsCombiner(const std::tuple<_DcsIn...>& dcsIns)
  {
    using DcsIns=
      std::tuple<_DcsIn...>;
    
    /// Result
    DcsOut dcsOut;
    UNROLLED_FOR((i,0,2),
		 {
		   EXEC_FOR_ALL_TUPLE_IDS(IDcsIn,DcsIns,
					  
					  /// Input component on which we loop
					  using DcsIn=
					  std::tuple_element_t<IDcsIn,DcsIns>;
					  
					  /// List of dynamic components in common with result
					  using DcsCommonToOut=
					  TupleCommonTypes<DcsOut,DcsIn>;
					  
					  /// Value of all dynamic components
					  decltype(auto) dcsIn=
					  std::get<IDcsIn>(dcsIns);
					  
					  EXEC_FOR_ALL_TUPLE_IDS(IDcIn,DcsCommonToOut,
								 
								 const auto& dcIn=
								 std::get<IDcIn>(dcsIn);
								 
								 auto& dcOut=
								 std::get<IDcIn>(dcsOut);
								 
								 if(i==0)
								   dcOut=dcIn;
								 else
								   if(dcOut!=dcIn)
								     CRASH<<"unmatched dynamic comps among expressions";
								 ));
		     });
    
    return
      dcsOut;
  }
}

#endif
