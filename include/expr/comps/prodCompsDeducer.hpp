#ifndef _PRODCOMPSDEDUCER_HPP
#define _PRODCOMPSDEDUCER_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file expr/comps/prodCompsDeducer.hpp

#include <expr/comps/comps.hpp>
#include <tuples/uniqueTupleFromTuple.hpp>

namespace esnort
{
  /// Product component deducer
  ///
  /// Takes as argument the components of the first factor, the
  /// components of the second factor, and puts in the output the
  /// visible and contracted components separately
  ///
  /// Forward declaration
  template <typename A,
	    typename B>
  struct ProdCompsDeducer;
  
  /// Product component deducer
  template <typename...TA,
	    typename...TB>
  struct ProdCompsDeducer<CompsList<TA...>,CompsList<TB...>>
  {
    /// Check if a certain component is contracted or visible
    template <RwCl RC,
	      typename C,
	      typename...O>
    struct CheckComp
    {
      static constexpr bool isContracted()
      {
	if constexpr(C::isTransposable)
	  return (C::RC==RC and (std::is_same_v<Transp<C>,O>||...));
	else
	  return false;
      }
      
      using Visible=
	std::conditional_t<isContracted(),std::tuple<>,std::tuple<C>>;
      
      using Contracted=
	std::conditional_t<not isContracted(),std::tuple<>,std::tuple<typename C::Transp>>;
    };
    
    template <typename A>
    using FirstCase=CheckComp<RwCl::CLN,A,TB...>;
    
    template <typename B>
    using SecondCase=CheckComp<RwCl::ROW,B,TA...>;
    
    using VisibleComps=
      UniqueTupleFromTuple<TupleCat<typename FirstCase<TA>::Visible...,
				    typename SecondCase<TB>::Visible...>>;
    
    using ContractedComps=
      TupleCat<typename FirstCase<TA>::Contracted...>;
  };
  
}

#endif
