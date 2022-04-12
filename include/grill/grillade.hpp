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
    DECLARE_UNTRANSPOSABLE_COMP(COMP_NAME ## Coord,int,0,comp_name ## Coord); \
									\
    DECLARE_UNTRANSPOSABLE_COMP(COMP_NAME ## Site,SITE_INT_TYPE,NSITES,comp_name ## Coord); \
									\
									\
    Grill<COMP_NAME ## Coord,COMP_NAME ## Site,HASHING> comp_name ## Grill
    
    DECLARE_GRILL(int64_t,0,Glb,glb,false);
    
    DECLARE_GRILL(int64_t,0,Rank,rank,true);
    
    DECLARE_GRILL(int64_t,0,Loc,loc,true);
    
    DECLARE_GRILL(int,2,Par,par,true);
    
    DECLARE_GRILL(int64_t,0,LocEo,locEo,true);
    
#undef DECLARE_GRILL
    
    Grillade()
    {
      parGrill.setSidesAndWrapping({2,1,1,1},{1,1,1,1});
    }
  };
}

#endif
