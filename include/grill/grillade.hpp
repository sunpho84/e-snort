#ifndef _GRILLADE_HPP
#define _GRILLADE_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file grill/grillade.hpp

#include <expr/comps/compLoops.hpp>
#include <grill/universe.hpp>

namespace esnort
{
    /// Assert that a site is in the allowed range
#define PROVIDE_ASSERT_IS_SITE(NAME,name,MAX)				\
									\
    constexpr INLINE_FUNCTION HOST_DEVICE_ATTRIB			\
    void assertIs ## NAME ## Site(const NAME ## Site& name ## Site) const \
    {									\
      assertIsInRange(#name "Site",name ## Site,MAX);			\
    }
    
    /// Assert that a coordinate is in the valid range
#define PROVIDE_ASSERT_IS_COORD(NAME,name)				\
    INLINE_FUNCTION HOST_DEVICE_ATTRIB					\
    void assertIs ## NAME ## Coord(const NAME ## Coord& name ## Coord,	\
				   const Dir& dir) const		\
    {									\
      assertIsInRange(#name "Coord",name ## Coord,name ## Sides(dir));	\
    }
    
    /// Assert that coordinates are in the valid range
#define PROVIDE_ASSERT_ARE_COORDS(NAME,name)				\
    INLINE_FUNCTION HOST_DEVICE_ATTRIB					\
    void assertAre ## NAME ## Coords(const NAME ## Coords& name ## Coords) const \
    {									\
    for(Dir dir=0;dir<NDims;dir++)					\
      assertIs ## NAME ## Coord(name ## Coords(dir),dir);		\
    }
    
#define PROVIDE_ASSERT_SITE_COORD_COORDS(NAME,name,MAX)	\
    PROVIDE_ASSERT_IS_SITE(NAME, name, MAX);		\
    PROVIDE_ASSERT_IS_COORD(NAME, name);		\
    PROVIDE_ASSERT_ARE_COORDS(NAME, name)
  
  template <int NDims,
	    int...I>
  struct Universe<NDims,std::integer_sequence<int,I...>>::Grillade
  {
    DECLARE_UNTRANSPOSABLE_COMP(GlbCoord,int,0,glbCoord);
    
    using GlbCoords=DirTens<GlbCoord>;
    
    DECLARE_UNTRANSPOSABLE_COMP(GlbSite,int64_t,0,glbSite);
    
    GlbSite glbVol;
    
    GlbCoords glbSides;
    
    PROVIDE_ASSERT_SITE_COORD_COORDS(Glb,glb,glbVol);
    
    /////////////////////////////////////////////////////////////////
    
    DECLARE_UNTRANSPOSABLE_COMP(RankCoord,int,0,rankCoord);
    
    using RankCoords=DirTens<RankCoord>;
    
    DECLARE_UNTRANSPOSABLE_COMP(Rank,int,0,rank);
    
    using RankSite=Rank;
    
    RankCoords nRanksPerDir;
    
    RankCoords rankCoords;
    
    Rank nRanks;
    
    DirTens<bool> glbIsNotPartitioned;
    
    DirTens<bool> glbIsNotSimdPartitioned;
    
    PROVIDE_ASSERT_SITE_COORD_COORDS(Rank,rank,nRanks);
    
    /////////////////////////////////////////////////////////////////
    
    DECLARE_UNTRANSPOSABLE_COMP(SimdRankCoord,int,0,simdRankCoord);
    
    using SimdRankCoords=DirTens<SimdRankCoord>;
    
    DECLARE_UNTRANSPOSABLE_COMP(SimdRank,int,maxAvailableSimdSize,simdSite);
    
    using SimdRankSite=SimdRank;
    
    const SimdRank nSimdRanks=maxAvailableSimdSize;
    
    SimdRankCoords nSimdRanksPerDir;
    
    PROVIDE_ASSERT_SITE_COORD_COORDS(SimdRank,simdRank,nSimdRanks);
    
    /////////////////////////////////////////////////////////////////
    
    DECLARE_UNTRANSPOSABLE_COMP(LocCoord,int,0,locCoord);
    
    using LocCoords=DirTens<LocCoord>;
    
    DECLARE_UNTRANSPOSABLE_COMP(LocSite,int64_t,0,locSite);
    
    DECLARE_UNTRANSPOSABLE_COMP(HaloSite,int64_t,0,haloSite);
    
    LocCoords locSides;
    
    LocSite locVol;
    
    PROVIDE_ASSERT_SITE_COORD_COORDS(Loc,loc,locVol);
    
    /////////////////////////////////////////////////////////////////
    
    DECLARE_UNTRANSPOSABLE_COMP(SimdLocCoord,int,0,simdLocCoord);
    
    using SimdLocCoords=DirTens<SimdLocCoord>;
    
    DECLARE_UNTRANSPOSABLE_COMP(SimdLocSite,int64_t,0,simdLocSite);
    
    SimdLocCoords simdLocSides;
    
    SimdLocSite simdLocVol;
    
    PROVIDE_ASSERT_SITE_COORD_COORDS(SimdLoc,simdLoc,simdLocVol);
    
    /////////////////////////////////////////////////////////////////
    
    DECLARE_UNTRANSPOSABLE_COMP(ParityCoord,int,0,parityCoord);
    
    using ParityCoords=DirTens<ParityCoord>;
    
    DECLARE_UNTRANSPOSABLE_COMP(Parity,int,2,parity);
    
    ParityCoords paritySides;
    
    /// Assert that a parity is in the allowed range
    constexpr INLINE_FUNCTION HOST_DEVICE_ATTRIB
    void assertIsParity(const Parity& parity) const
    {
      assertIsInRange("parity",parity,2);
    }
    
    /////////////////////////////////////////////////////////////////
    
    DECLARE_UNTRANSPOSABLE_COMP(LocEoCoord,int,0,locEoCoord);
    
    using LocEoCoords=DirTens<LocEoCoord>;
    
    DECLARE_UNTRANSPOSABLE_COMP(LocEoSite,int64_t,0,locEoSite);
    
    LocEoCoords locEoSides;
    
    LocEoSite locEoVol;
    
    /// Assert that a local e/o site is in the allowed range
    constexpr INLINE_FUNCTION HOST_DEVICE_ATTRIB
    void assertIsLocEoSite(const LocEoSite& locEoSite) const
    {
      assertIsInRange("locEoSite",locEoSite,locEoVol);
    }
    
    /////////////////////////////////////////////////////////////////
    
    DECLARE_UNTRANSPOSABLE_COMP(SimdLocEoCoord,int,0,simdLocEoCoord);
    
    using SimdLocEoCoords=DirTens<SimdLocEoCoord>;
    
    DECLARE_UNTRANSPOSABLE_COMP(SimdLocEoSite,int64_t,0,simdLocEoSite);
    
    SimdLocEoCoords simdLocEoSides;
    
    SimdLocEoSite simdLocEoVol;
    
    /// Assert that a local e/o site is in the allowed range
    constexpr INLINE_FUNCTION HOST_DEVICE_ATTRIB
    void assertIsSimdLocEoSite(const SimdLocEoSite& simdLocEoSite) const
    {
      assertIsInRange("simdLocEoSite",simdLocEoSite,simdLocEoVol);
    }
    
    /////////////////////////////////////////////////////////////////
    
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
		    CRASH<<
		      "in dir "<<dir<<" the "<<dividendName<<" grill side "<<dividend(dir)<<
		      " cannot be divided by the "<<divisorName<<" grill side "<<divisor(dir);
		});
    };
    
    /// Initialize a grillade
    Grillade(const GlbCoords& glbSides,
	     const RankCoords& nRanksPerDir,
	     const SimdRankCoords& nSimdRanksPerDir,
	     const Dir& parityDir) :
      glbSides(glbSides),
      nRanksPerDir(nRanksPerDir),
      nSimdRanksPerDir(nSimdRanksPerDir)
    {
      // Set the global and rank grills
      
      glbVol=(glbSides(Dir(I))*...);
      
      /////////////////////////////////////////////////////////////////
      
      // Set the sides of the parity grill and initializes it
      
      paritySides=1+versor<ParityCoord>(parityDir);
      
      assertIsPartitionable(locSides,"local",paritySides,"parity");
      
      /////////////////////////////////////////////////////////////////
      
      // Set the local grill
      
      locSides=glbSides/nRanksPerDir;
      LOGGER<<"Local sides: ";
      printCoords(LOGGER,locSides);
      
      assertIsPartitionable(glbSides,"global",nRanksPerDir,"rank");
      
      locVol=(locSides(Dir(I))*...);
      
      glbIsNotPartitioned=(nRanksPerDir==1);
      
      /////////////////////////////////////////////////////////////////
      
      // Set the simd local grill
      
      simdLocSides=locSides/nSimdRanksPerDir;
      LOGGER<<"Simd local sides: ";
      printCoords(LOGGER,simdLocSides);
      
      assertIsPartitionable(locSides,"local",nSimdRanksPerDir,"simdRank");
      
      glbIsNotSimdPartitioned=glbIsNotPartitioned and (nRanksPerDir==1);
      
      /////////////////////////////////////////////////////////////////
      
      // Set rank coordinate and local origin global coordinates
      
      LOGGER<<"Rank "<<Mpi::rank;
      
      LOGGER<<"Rank sides: ";
      printCoords(LOGGER,nRanksPerDir);
      
      nRanks=(nRanksPerDir(Dir(I))*...);
      rankCoords=computeCoordsOfSiteInBoxOfSize(Mpi::rank,nRanksPerDir);
      LOGGER<<"Coordinates of the rank: ";
      printCoords(LOGGER,rankCoords);
      
      const GlbCoords originGlbCoords=rankCoords*locSides;
      
      LOGGER<<"Coordinates of the origin: ";
      printCoords(LOGGER,originGlbCoords);
      
      /////////////////////////////////////////////////////////////////
      
      // Set the simd grill
      
      const int nSimdRanksCheck=(nSimdRanksPerDir(Dir(I))*...);
      if(nSimdRanksCheck!=nSimdRanks)
	{
	  std::ostringstream os;
	  os<<"nSimdRanksPerDir ";
	  printCoords(os,nRanksPerDir);
	  os<<" amounts to total "<<nSimdRanksCheck<<" different from expected "<<nSimdRanks;
	  
	  CRASH<<os.str();
	}
      
      glbIsNotSimdPartitioned=glbIsNotPartitioned and (nRanksPerDir==1);
      
      /////////////////////////////////////////////////////////////////
      
      // Set the simd local e/o grill
      
      simdLocEoSides=simdLocSides/paritySides;
      LOGGER<<"Simd e/o local sides: ";
      printCoords(LOGGER,simdLocEoSides);
      
      assertIsPartitionable(locSides,"local",nSimdRanksPerDir,"simdRank");
      
      /////////////////////////////////////////////////////////////////
      
      // // Set the parity and even/odd of the local sites, and the local site of a given parity and locale e/o site
      
      // using ParLocEoSite=std::tuple<ParSite,LocEoSite>;
      
      // DynamicTens<OfComps<LocSite>,ParLocEoSite,ExecSpace::HOST> parLocEoSiteOfLocSite(std::make_tuple(locVol));
      // DynamicTens<OfComps<ParSite,LocEoSite>,LocSite,ExecSpace::HOST> locSiteOfParLocEoSite(std::make_tuple(locEoVol));
      
      // locGrill.forAllSites([this,&parLocEoSiteOfLocSite,&locSiteOfParLocEoSite,&glbCoordsOfLocSite,&parityDir](const LocSite& locSite)
      // {
      // 	const LocCoords locCoords=locGrill.coordsOfSite(locSite);
      // 	const LocEoCoords locEoCoords=locCoords/parSides;
      // 	const LocEoSite locEoSite=locEoGrill.siteOfCoords(locEoCoords);
	
      // 	const GlbCoords glbCoords=glbCoordsOfLocSite(locSite);
      // 	const ParSite parSite=(glbCoords(Dir(I))+...)%2;
	
      // 	parLocEoSiteOfLocSite(locSite)={parSite,locEoSite};
      // 	locSiteOfParLocEoSite(parSite,locEoSite)=locSite;
      // });
      
      // /////////////////////////////////////////////////////////////////
      
      // DynamicTens<OfComps<ParSite,LocEoSite,Ori,Dir>,LocEoSite,ExecSpace::HOST> neighsOfLoceo(std::make_tuple(locEoVol));
      
      // for(ParSite parSite=0;parSite<2;parSite++)
      // 	locEoGrill.forAllSites([this,&parSite,&neighsOfLoceo,&locSiteOfParLocEoSite](const LocEoSite& locEoSite)
      // 	{
      // 	  const LocSite locSite=locSiteOfParLocEoSite(parSite,locEoSite);
	  
      // 	  const GlbCoords glbCoords=glbCoordsOfLocSite(locSite);
	  
      // 	  for(Ori ori=0;ori<2;ori++)
      // 	    for(Dir dir=0;dir<NDims;dir++)
      // 	      {
      // 		const GlbCoords glbNeighCoords=glbCoords-moveOffset[ori]*versor<int>(dir);
      // 		const LocCoords locNeighCoords=glbNeighCoords-originGlbCoords;
      // 		const LocSite locNeigh=locGrill.neighOfSite(locSite,ori,dir);
      // 	      }
      // });
    }
  };
  
  
#undef PROVIDE_ASSERT_IS_SITE
#undef PROVIDE_ASSERT_IS_COORD
#undef PROVIDE_ASSERT_ARE_COORDS
#undef PROVIDE_ASSERT_SITE_COORD_COORDS
  
}

#endif
