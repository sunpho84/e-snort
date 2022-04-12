#ifndef _GRILLADE_HPP
#define _GRILLADE_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file grill/grillade.hpp

#include <grill/grill.hpp>

namespace esnort
{
  template <int NDims,
	    int...I>
  struct Universe<NDims,std::integer_sequence<int,I...>>::Grillade
  {
#define DECLARE_GRILL(SITE_INT_TYPE,NSITES,COMP_NAME,comp_name,HASHING)	\
    									\
    DECLARE_UNTRANSPOSABLE_COMP(COMP_NAME ## Coord,int,0,comp_name ## Coord); \
    									\
    DECLARE_UNTRANSPOSABLE_COMP(COMP_NAME ## Site,SITE_INT_TYPE,NSITES,comp_name ## Coord); \
    									\
    using COMP_NAME ## Coords=DirTens<COMP_NAME ## Coord>;		\
    									\
    Grill<COMP_NAME ## Coord,COMP_NAME ## Site,HASHING> comp_name ## Grill
    
    DECLARE_GRILL(int64_t,0,Glb,glb,false);
    
    DECLARE_GRILL(int64_t,0,Rank,rank,true);
    
    DECLARE_GRILL(int64_t,0,Loc,loc,true);
    
    DECLARE_GRILL(int,0,Par,par,true); /// This should be set to 2, but we hit the problem of allocating a non-allocatable
    
    DECLARE_GRILL(int64_t,0,LocEo,locEo,true);
    
#undef DECLARE_GRILL
    
    /// Initializes all directions as wrapping
    static constexpr DirTens<bool> allDirWraps() {return true;}
    
    /// Initialize a grillade
    Grillade(const GlbCoords& glbSides,
	     const RankCoords& rankSides,
	     const Dir& parityDir)
    {
      glbGrill.setSidesAndWrapping(glbSides,allDirWraps());
      
      const ParCoords parSides=
	ParCoords(1)+versor<ParCoord>(parityDir);
      
      parGrill.setSidesAndWrapping(parSides,allDirWraps());
    }
  };
}

#endif
