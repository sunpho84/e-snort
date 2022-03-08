#ifndef _INDEX_COMPUTER_HPP
#define _INDEX_COMPUTER_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file indexComputer.hpp
///
/// \brief Compute index given components

#include <expr/comp.hpp>
#include <expr/comps.hpp>
#include <tuples/tupleDiscriminate.hpp>
#include <tuples/tupleSubset.hpp>
#include <tuples/uniqueTuple.hpp>

namespace esnort
{
  // DEFINE_CRTP_INHERITANCE_DISCRIMINER_FOR_TYPE(IndexComputer)
  
  // template <typename T>
  // struct IndexComputer :
  // Crtp<IndexComputer<T>,crtp::IndexComputerDiscriminer>
  // {
    // /// Type to be used for the index
    // using Index=
    //   std::common_type_t<int,typename TC::Index...>;
    
    // /// Set the dynamic sizes
    // template <typename...TD>
    // IndexComputer(const TD&&...td) :
    //   dynamicSizes(tupleGetSubset<DynamicComps>(std::make_tuple(td...)))
    // {
    //   static_assert(sizeof...(TD)==std::tuple_size_v<DynamicComps>,"Cannot allocate without knowledge of all the dynamic sizes");
    // }
    
    // /// Static component maximal value
    // static constexpr Index staticPartMaxValue=
    //   ((TC::sizeIsKnownAtCompileTime?
    // 	TC::sizeAtCompileTime():
    //    Index{1})*...*1);
    
    
    // /// Determine whether the components are all static, or not
    // static constexpr bool allCompsAreStatic=
    //   std::is_same<DynamicComps,std::tuple<>>::value;
    
    // /// Computes the maximal value size at compile time, if known
    // static constexpr Index maxValAtCompileTime=
    //   allCompsAreStatic?(Index)staticPartMaxValue:(Index)DYNAMIC;
    
    
    // The parsing of the variadic components is done left to right, so
    // to compute the nested bracket list we must proceed inward. Let's
    // say we are at component j. We define outer=(0*ni+i) the result
    // of inner step. We need to compute thisVal=outer*nj+j and pass it
    // to the inner step, which incorporate iteratively all the inner
    // components. The first step requires outer=0.
  // template <typename DynamicSizes,
  // 	    typename Index,
  // 	    typename Head,
  // 	    typename...Tail>
  // constexpr HOST_DEVICE_ATTRIB INLINE_FUNCTION
  // Index _index(const DynamicSizes& dynamicSizes, ///< Dynamic sizes
  // 	       const Index& outer,               ///< Value of all the outer components
  // 	       const Head& head,                 ///< Currently parsed component
  // 	       const Tail&...tail)               ///< Inner components
  // {
  //   // Calculate index iteratively
  //   // Given the components (i,j,k) we must compute ((0*ni+i)*nj+j)*nk+k
    
  //   /// Size of this component
  //   Index size;
    
  //   if constexpr(Head::sizeIsKnownAtCompileTime)
  //     size=Head::sizeAtCompileTime();
  //   else
  //     size=std::get<Head>(dynamicSizes);
    
  //   /// Value of the index when including this component
  //   const Index inner=
  //     outer*size+head();
    
  //   if constexpr(sizeof...(tail))
  //     return
  // 	_index(inner,tail...);
  //   else
  //     return inner;
  // }
    
    /// Dispatch the internal index calculation
    ///
    /// This works when the passed components are already well ordered
  template <typename DynamicComps,
	    typename...C,
	    typename...Index>
  HOST_DEVICE_ATTRIB INLINE_FUNCTION
  auto index(const DynamicComps& dynamicSizes,
	     const BaseComp<C,Index>&...comps)
  {
    using GlbIndex=
      std::common_type_t<int,Index...>;
    
    auto index=
      [&dynamicSizes](const auto& index,const GlbIndex& outer,const auto& head,const auto&...tail)INLINE_ATTRIBUTE
    {
      using Head=
	std::decay_t<decltype(head)>;
      
      GlbIndex size;
    
      if constexpr(Head::sizeIsKnownAtCompileTime)
	size=Head::sizeAtCompileTime;
      else
	size=std::get<Head>(dynamicSizes)();
      
      /// Value of the index when including this component
      const GlbIndex inner=
	outer*size+head();
      
      if constexpr(sizeof...(tail))
	return
	  index(index,inner,tail...);
      else
	return inner;
    };
    
    return index(index,0,comps.crtp()...);
  }

    /// Dispatch the internal index calculation
    ///
    /// This works when the passed components are already well ordered
  template <typename...C,
	    typename...Index>
  constexpr HOST_DEVICE_ATTRIB INLINE_FUNCTION
  auto index(const std::tuple<>&,
	     const BaseComp<C,Index>&...comps)
  {
    using GlbIndex=
      std::common_type_t<int,Index...>;
    
    constexpr auto index=
      [](const auto& index,const GlbIndex& outer,const auto& head,const auto&...tail)INLINE_ATTRIBUTE
    {
      using Head=std::remove_reference_t<decltype(head)>;
      
      constexpr GlbIndex size=Head::sizeAtCompileTime;
      
      /// Value of the index when including this component
      const GlbIndex inner=
	outer*size+head();
      
      if constexpr(sizeof...(tail))
	return
	  index(index,inner,tail...);
      else
	return inner;
    };
    
    return index(index,0,comps.crtp()...);
  }
}

#endif
