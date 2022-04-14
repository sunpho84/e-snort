#ifndef _GRILLADE_HPP
#define _GRILLADE_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file grill/grillade.hpp

#include <expr/comps/compLoops.hpp>
#include <grill/grill.hpp>

namespace esnort
{
  template <int NDims,
	    int...I>
  struct Universe<NDims,std::integer_sequence<int,I...>>::Grillade
  {
#define DECLARE_GRILL(SITE_INT_TYPE,NSITES,COMP_NAME,comp_name,HASHING)	\
    									\
    static constexpr char _ ## COMP_NAME ## Name[]=			\
      TO_STRING(COMP_NAME);						\
    									\
    using COMP_NAME ## Grill=						\
      Grill<SITE_INT_TYPE,NSITES,HASHING,_ ## COMP_NAME ## Name>;	\
    									\
    using COMP_NAME ## Coord=						\
      typename COMP_NAME ## Grill::Coord;				\
    									\
    using COMP_NAME ## Site=						\
      typename COMP_NAME ## Grill::Site;				\
    									\
    using COMP_NAME ## Coords=						\
      typename COMP_NAME ## Grill::Coords;				\
									\
    COMP_NAME ## Grill comp_name ## Grill
    
    DECLARE_GRILL(int64_t,0,Glb,glb,false);
    
    DECLARE_GRILL(int64_t,0,Rank,rank,true);
    
    DECLARE_GRILL(int64_t,0,Loc,loc,true);
    
    DECLARE_GRILL(int,2,Par,par,true);
    
    DECLARE_GRILL(int64_t,0,LocEo,locEo,true);
    
#undef DECLARE_GRILL

    template <typename L,
	      typename T>
    static void printCoords(L&& l,const DirTens<T>& c)
    {
      COMP_LOOP(Dir,dir,
		{
		  l<<c(dir);
		  if(dir!=NDims-1) l<<" ";
		});
    }
    
    /// Initializes all directions as wrapping
    static constexpr DirTens<bool> allDirWraps()
    {
      return true;
    }
    
    /// Check that a coords set is partitionable by another one
    template <typename Dividend,
	      typename Divisor>
    static void assertIsPartitionable(const DirTens<Dividend>& dividend,
				      const char* dividendName,
				      const DirTens<Divisor>& divisor,
				      const char* divisorName)
    {
      const DirTens<bool> isPartitionable=
	(dividend%divisor==0);
      
      COMP_LOOP(Dir,dir,
		{
		  if(not isPartitionable(dir))
		    CRASH<<"in dir "<<dir<<" the "<<dividendName<<" grill side "<<dividend(dir)<<
		      " cannot be divided by the "<<divisorName<<" grill side "<<divisor(dir);
		});
    };
    
    /// Initialize a grillade
    Grillade(const GlbCoords& glbSides,
	     const RankCoords& rankSides,
	     const Dir& parityDir)
    {
      glbGrill.setSidesAndWrapping(glbSides,allDirWraps());
      
      rankGrill.setSidesAndWrapping(rankSides,allDirWraps());
      
      /////////////////////////////////////////////////////////////////
      
      const ParCoords parSides=
	1+versor<ParCoord>(parityDir);
      
      parGrill.setSidesAndWrapping(parSides,allDirWraps());
      
      /////////////////////////////////////////////////////////////////
      
      const LocCoords locSides=glbSides/rankSides;
      LOGGER<<"Local sides: ";
      printCoords(LOGGER,locSides);
      
      assertIsPartitionable(glbSides,"global",rankSides,"rank");
      
      const DirTens<bool> glbIsNotPartitioned=(rankSides==1);
      
      locGrill.setSidesAndWrapping(locSides,glbIsNotPartitioned);
      
      /////////////////////////////////////////////////////////////////
      
      assertIsPartitionable(locSides,"local",parSides,"parity");
      
      const LocEoCoords locEoSides=locSides/parSides;
      
      locEoGrill.setSidesAndWrapping(locEoSides,glbIsNotPartitioned);
      
      /////////////////////////////////////////////////////////////////
      
      LOGGER<<"Rank "<<Mpi::rank;
      
      LOGGER<<"Rank sides: ";
      printCoords(LOGGER,rankGrill.sides());
      
      const RankCoords rankCoords=rankGrill.coordsOfSite(Mpi::rank);
      LOGGER<<"Coordinates of the rank: ";
      printCoords(LOGGER,rankCoords);
      
      const GlbCoords originGlbCoords=rankCoords*locSides;
      
      LOGGER<<"Coordinates of the origin: ";
      printCoords(LOGGER,originGlbCoords);
      
      /////////////////////////////////////////////////////////////////
      
      const LocSite locVol=locGrill.vol();
      
      DynamicTens<CompsList<LocSite>,GlbCoords,ExecSpace::HOST> glbCoordsOfLocSite(std::make_tuple(locVol));
      locGrill.forAllSites([this,&glbCoordsOfLocSite,&originGlbCoords](const LocSite& locSite)
      {
	const LocCoords locCoords=locGrill.coordsOfSite(locSite);
	
	glbCoordsOfLocSite(locSite)=locCoords+originGlbCoords;
      });
      
      /////////////////////////////////////////////////////////////////
      
      DynamicTens<OfComps<LocSite>,ParSite,ExecSpace::HOST> parSiteOfLocSite(std::make_tuple(locVol));
      
      locGrill.forAllSites([this,&parSiteOfLocSite,&glbCoordsOfLocSite](const LocSite& locSite)
      {
	const GlbCoords glbCoords=glbCoordsOfLocSite(locSite);
	
	auto l=LOGGER;
	l<<" "<<locSite<<" ";
	printCoords(l,glbCoords);
	
	parSiteOfLocSite(locSite)=(glbCoords(Dir(I))+...)%2;
      });
      
      LocCoords r;
      r=0;
      for(r(Dir(2))=0;r(Dir(2))<locSides(Dir(2));r(Dir(2))++)
	{
	  auto l=LOGGER;
	  l<<" ";
	  for(r(Dir(3))=0;r(Dir(3))<locSides(Dir(3));r(Dir(3))++)
	    {
	      const LocSite locSite=locGrill.siteOfCoords(r);
	      l<<parSiteOfLocSite(locSite);
	    }
	}
      
      const LocEoSite locEoVol=locEoGrill.vol();
      
      /////////////////////////////////////////////////////////////////
      
      DynamicTens<OfComps<ParSite,LocEoSite>,LocSite,ExecSpace::HOST> locSiteOfParLocEoSite(std::make_tuple(locEoVol));
      
      
    }
  };
}

#endif
