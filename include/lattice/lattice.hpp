#ifndef _LATTICE_HPP
#define _LATTICE_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file lattice/lattice.hpp

#include <expr/comps/compLoops.hpp>
#include <lattice/universe.hpp>
#include <metaprogramming/forEachInTuple.hpp>
#include <resources/mathOperations.hpp>

namespace grill
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
  
#define PROVIDE_COORDS_OF_SITE_COMPUTER(NAME,name,SIDES)		\
									\
    /*! Computes the coordinates of the passed name ## Site */		\
    constexpr INLINE_FUNCTION HOST_DEVICE_ATTRIB			\
    NAME ## Coords compute ## NAME ## Coords(const NAME ## Site& name ## Site) const \
    {									\
      return U::computeCoordsOfSiteInBoxOfSides(name ## Site,SIDES);	\
    }									\
									\
    /*! Computes the global site given the global coordinates */	\
    constexpr INLINE_FUNCTION HOST_DEVICE_ATTRIB			\
    NAME ## Site compute ## NAME ## Site(const NAME ## Coords& name ## Coords) const \
    {									\
      return								\
	U::template computeSiteOfCoordsInBoxOfSides<NAME ## Site>(name ## Coords,SIDES); \
    }
  
  /// Declare lattice
  template <typename U>
  struct Lattice;
  
  /// Lattice
  template <int NDims>
  struct Lattice<Universe<NDims>>
  {
    /// Corresponding universe
    using U=
      Universe<NDims>;
    
    /// Directions in the space
    using Dir=
      typename U::Dir;
    
    /// A tensor with component direction
    template <typename F>
    using DirTens=typename U::template DirTens<F>;
    
    /// Global coordinate
    DECLARE_UNTRANSPOSABLE_COMP(GlbCoord,int,0,glbCoord);
    
    /// Global coordinates
    using GlbCoords=DirTens<GlbCoord>;
    
    /// Global site
    DECLARE_UNTRANSPOSABLE_COMP(GlbSite,int64_t,0,glbSite);
    
    /// Global volume
    GlbSite glbVol;
    
    /// Global sides
    GlbCoords glbSides;
    
    PROVIDE_ASSERT_SITE_COORD_COORDS(Glb,glb,glbVol);
    
    PROVIDE_COORDS_OF_SITE_COMPUTER(Glb,glb,glbSides);
    
    /// Returns the coordinates shifted in the asked direction
    ///
    /// Periodic boundary conditions are always assumed
    GlbCoords shiftedCoords(const GlbCoords& in,
			    const Ori& ori,
			    const Dir& dir,
			    const int amount=1)
      const
    {
      GlbCoords out=in;
      
      out(dir)=safeModulo(GlbCoord(in(dir)+moveOffset[ori]*amount),glbSides(dir));
      
      return
	out;
    }
    
    /////////////////////////////////////////////////////////////////
    
    /// Rank Coordinate
    DECLARE_UNTRANSPOSABLE_COMP(RankCoord,int,0,rankCoord);
    
    /// Rank coordinates
    using RankCoords=DirTens<RankCoord>;
    
    /// Rank
    DECLARE_UNTRANSPOSABLE_COMP(Rank,int,0,rank);
    
    using RankSite=Rank;
    
    RankCoords nRanksPerDir;
    
    DynamicTens<OfComps<Rank>,RankCoords,ExecSpace::HOST> rankCoords;
    
    /// Coordinates of the present rank
    RankCoords thisRankCoords;
    
    Rank nRanks;
    
    DirTens<bool> glbIsNotPartitioned;
    
    DirTens<bool> glbIsNotSimdPartitioned;
    
    DynamicTens<OfComps<Rank>,GlbCoords,ExecSpace::HOST> originGlbCoords;
    
    /// Global coordinates of the origin of present rank
    GlbCoords thisRankOriginGlbCoords;
    
    PROVIDE_ASSERT_SITE_COORD_COORDS(Rank,rank,nRanks);
    
    PROVIDE_COORDS_OF_SITE_COMPUTER(Rank,rank,nRanksPerDir);
    
    /////////////////////////////////////////////////////////////////
    
    DECLARE_UNTRANSPOSABLE_COMP(SimdRankCoord,int,0,simdRankCoord);
    
    using SimdRankCoords=DirTens<SimdRankCoord>;
    
    DECLARE_UNTRANSPOSABLE_COMP(SimdRank,int,maxAvailableSimdSize,simdSite);
    
    using SimdRankSite=SimdRank;
    
    const SimdRank nSimdRanks=maxAvailableSimdSize;
    
    SimdRankCoords nSimdRanksPerDir;
    
    PROVIDE_ASSERT_SITE_COORD_COORDS(SimdRank,simdRank,nSimdRanks);
    
    PROVIDE_COORDS_OF_SITE_COMPUTER(SimdRank,simdRank,nSimdRanksPerDir);
    
    StackTens<OfComps<SimdRank>,SimdRankCoords> simdRankCoords;
    
    /////////////////////////////////////////////////////////////////
    
    DECLARE_UNTRANSPOSABLE_COMP(LocCoord,int,0,locCoord);
    
    using LocCoords=DirTens<LocCoord>;
    
    DECLARE_UNTRANSPOSABLE_COMP(LocSite,int64_t,0,locSite);
    
    DECLARE_UNTRANSPOSABLE_COMP(HaloSite,int64_t,0,haloSite);
    
    LocCoords locSides;
    
    LocSite locVol;
    
    PROVIDE_ASSERT_SITE_COORD_COORDS(Loc,loc,locVol);
    
    PROVIDE_COORDS_OF_SITE_COMPUTER(Loc,loc,locSides);
    
    /// Returns the side in the bulk considering the wrapping
    LocCoords bulkSides;
    
    /// Total volume in the bulk
    LocSite bulkVol;
    
    // /// Computes the coordinates of the passed locSite
    // GlbCoords computeGlbCoords(const LocSite& locSite) const
    // {
    //   return originGlbCoords+computeLocCoords(locSite);
    // }
    
    /// Local origin of the simd rank
    StackTens<OfComps<SimdRank>,LocCoords> simdRankLocOrigins;
    
    /////////////////////////////////////////////////////////////////
    
    /// Surface sizes
    DirTens<LocSite> locSurfPerDir;
    
    /// Volume of the surface
    LocSite locSurf;
    
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
    
    /// Opposite parity
    constexpr INLINE_FUNCTION Parity oppositeParity(const Parity& parity)
    {
      return 1-parity();
    }
    
    ParityCoords paritySides;
    
    DynamicTens<OfComps<Rank>,Parity,ExecSpace::HOST> originParity;
    
    /// Parity of the origin of the present rank
    Parity thisRankOriginParity;
    
    StackTens<OfComps<Parity>,ParityCoords> parityCoords;
    
    /// Assert that a parity is in the allowed range
    constexpr INLINE_FUNCTION HOST_DEVICE_ATTRIB
    void assertIsParity(const Parity& parity) const
    {
      assertIsInRange("parity",parity,2);
    }
    
    /// Compute the parity of global coords
    constexpr INLINE_FUNCTION HOST_DEVICE_ATTRIB
    Parity glbCoordsParity(const GlbCoords& glbCoords) const
    {
      GlbCoord sum=0;
      for(Dir dir=0;dir<NDims;dir++)
	sum+=glbCoords(dir);
      
      return sum%2;
    }
    
    /// Parity split direction
    Dir parityDir;
    
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
    
    /// Surface e/o sizes
    StackTens<OfComps<Parity,Ori,Dir>,LocEoSite> locEoSurfPerDir;
    
    /// Volume of the e/o surface
    StackTens<OfComps<Parity>,LocEoSite> locEoSurf;
    
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
    
    PROVIDE_COORDS_OF_SITE_COMPUTER(SimdLocEo,simdLocEo,simdLocEoSides);
    
    /////////////////////////////////////////////////////////////////
    
    /// Components needed to identify a local site within a e/o simd representation
    using SimdEoRepOfLocSite=OfComps<Parity,SimdLocEoSite,SimdRank>;
    
    /// Computes the coordinate of the site, adding to the global
    /// origin the simdRank origin and adding the local simd eo
    /// coords inflated along the parity direction, and adjusting
    /// the resulting parity
    GlbCoords computeGlbCoordsOfSimdEoRepOfLocSite(const Parity& parity,
						   const SimdLocEoSite& simdLocEoSite,
						   const SimdRank& simdRank) const
    {
      const SimdLocEoCoords simdLocEoCoords=
	computeSimdLocEoCoords(simdLocEoSite);
      
      const SimdLocCoords simdLocCoords=
	simdLocEoCoords*paritySides;
      
      const LocCoords simdRankLocOrigin=
	simdRankLocOrigins(simdRank);
      
      const GlbCoords glbCoords=
	thisRankOriginGlbCoords+simdRankLocOrigin+simdLocCoords;
      
      /// Computes the parity of the resulting coordinate
      const Parity glbParity=
	glbCoordsParity(glbCoords);
      
      const Parity neededParity=
	(glbParity+parity)%2;
      
      return glbCoords+parityCoords(neededParity);
    }
    
    /// Components needed to identify a gloval site within a e/o simd representation
    using SimdEoRepOfGlbSite=OfComps<Rank,Parity,SimdLocEoSite,SimdRank>;
    
    SimdEoRepOfGlbSite computeSimdEoRepOfLocSiteOfGlbCoords(const GlbCoords& glbCoords) const
    {
      const RankCoords rankCoords=
	glbCoords/locSides;
      
      const Rank rank=
	computeRankSite(rankCoords);
      
      const LocCoords locCoords=
	glbCoords-originGlbCoords(rank);
      
      const Parity parity=
	glbCoordsParity(glbCoords);
      
      const SimdLocCoords simdLocCoords=
	locCoords%simdLocSides;
      
      const SimdRankCoords simdRankCoords=
	locCoords/simdLocSides;
      
      const SimdLocEoCoords simdLocEoCoords=
	simdLocCoords/paritySides;
      
      const SimdLocEoSite simdLocEoSite=
	computeSimdLocEoSite(simdLocEoCoords);
      
      const SimdRank simdRank=
	computeSimdRankSite(simdRankCoords);
      
      return {rank,parity,simdLocEoSite,simdRank};
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

#ifndef __CUDA_ARCH__
      COMP_LOOP(Dir,dir,
		{
		  if(not isPartitionable(dir))
		    CRASH<<
		      "in dir "<<dir<<" the "<<dividendName<<" lattice side "<<dividend(dir)<<
		      " cannot be divided by the "<<divisorName<<" lattice side "<<divisor(dir);
		});
#endif
    };
    
    /// Initialize a lattice
    Lattice(const GlbCoords& glbSides,
	    const RankCoords& nRanksPerDir,
	    const SimdRankCoords& nSimdRanksPerDir,
	    const Dir& parityDir) :
      glbSides(glbSides),
      nRanksPerDir(nRanksPerDir),
      nSimdRanksPerDir(nSimdRanksPerDir),
      parityDir(parityDir)
    {
      // Set the global and rank lattices
      
      glbVol=1;
      for(Dir dir=0;dir<NDims;dir++)
	glbVol*=glbSides(dir);
      
      /////////////////////////////////////////////////////////////////
      
      // Set the sides of the parity lattice and initializes it
      
      parityCoords(parity(0))=0;
      parityCoords(parity(1))=U::template versor<ParityCoord>(parityDir);
      
      paritySides=1+parityCoords(parity(1));
      
      LOGGER<<"Parity dir: "<<parityDir;
      
      LOGGER<<"Parity sides: ";
      printCoords(LOGGER,paritySides);
      
#ifndef __CUDA_ARCH__
      COMP_LOOP(Parity,parity,
		{
		  auto l=LOGGER;
		  l<<"parity "<<parity<<" coords: ";
		  printCoords(l,parityCoords(parity));
		});
#endif
      
      assertIsPartitionable(locSides,"local",paritySides,"parity");
      
      /////////////////////////////////////////////////////////////////
      
      // Set the local lattice
      
      locSides=glbSides/nRanksPerDir;
      LOGGER<<"Local sides: ";
      printCoords(LOGGER,locSides);
      
      assertIsPartitionable(glbSides,"global",nRanksPerDir,"rank");
      
      locVol=1;
      for(Dir dir=0;dir<NDims;dir++)
	locVol*=locSides(dir);
      
      glbIsNotPartitioned=(nRanksPerDir==1);
      
      /////////////////////////////////////////////////////////////////
      
      bulkVol=1;
      COMP_LOOP(Dir,dir,
		{
		  bulkSides(dir)=
		    (nRanksPerDir(dir)==1)?
		    locSides(dir):
		    ((locSides(dir)>2)?
		     (locSides(dir)-2):
		     0);
		  
		  bulkVol*=bulkSides(dir);
		});
      
      locSurf=locVol-bulkVol;
      
      /////////////////////////////////////////////////////////////////
      
      // Set the local surface
      
      COMP_LOOP(Dir,dir,
		{
		  locSurfPerDir(dir)=
		    (nRanksPerDir(dir)>1)?
		    (locVol/locSides(dir)):
		    0;
		  
		  locSurf+=locSurfPerDir(dir);
		});
      
      /////////////////////////////////////////////////////////////////
      
      // Set the local e/o sides
      
      locEoSides=locSides/paritySides;
      LOGGER<<"Local e/o sides: ";
      printCoords(LOGGER,locEoSides);
      
      assertIsPartitionable(locSides,"local",paritySides,"parity");
      
      locEoVol=locVol/2;
      
      /////////////////////////////////////////////////////////////////
      
      // Set the local e/o surface
      
      locEoSurf=locSurf/2;
      
      locEoSurfPerDir=locSurfPerDir/2;
      if(locSurfPerDir(parityDir)%2)
	{
	  locEoSurfPerDir(BW,thisRankOriginParity,parityDir)++;
	  locEoSurfPerDir(FW,oppositeParity(thisRankOriginParity),parityDir)++;
	}
      
      /////////////////////////////////////////////////////////////////
      
      // Set the simd local lattice
      
      simdLocSides=locSides/nSimdRanksPerDir;
      simdLocVol=locVol/nSimdRanks;
      
      LOGGER<<"Simd local sides: ";
      printCoords(LOGGER,simdLocSides);
      
      assertIsPartitionable(locSides,"local",nSimdRanksPerDir,"simdRank");
      
      glbIsNotSimdPartitioned=glbIsNotPartitioned and (nRanksPerDir==1);
      
      /////////////////////////////////////////////////////////////////
      
      // Set rank coordinate and local origin global coordinates
      
      LOGGER<<"Rank "<<Mpi::rank;
      
      LOGGER<<"nRanksPerDir: ";
      printCoords(LOGGER,nRanksPerDir);
      
      nRanks=1;
      for(Dir dir=0;dir<NDims;dir++)
	nRanks*=nRanksPerDir(dir);
      forEachInTuple(std::make_tuple(&originGlbCoords,&rankCoords,&originParity),[this](auto* p)
      {
	p->allocate(std::make_tuple(nRanks));
      });
      
      for(Rank rank=0;rank<nRanks;rank++)
	{
	  rankCoords(rank)=computeRankCoords(rank);
	  originGlbCoords(rank)=rankCoords(rank)*locSides;
	  originParity(rank)=glbCoordsParity(originGlbCoords(rank));
	}
      
      thisRankCoords=rankCoords(Rank(Mpi::rank));
      thisRankOriginGlbCoords=originGlbCoords(Rank(Mpi::rank));
      thisRankOriginParity=originParity(Rank(Mpi::rank));
      
      LOGGER<<"Coordinates of the rank: ";
      printCoords(LOGGER,thisRankCoords);
      
      LOGGER<<"Coordinates of the origin: ";
      printCoords(LOGGER,thisRankOriginGlbCoords);
      
      /////////////////////////////////////////////////////////////////
      
      // Set the simd lattice
      
      int nSimdRanksCheck=1;
      for(Dir dir=0;dir<NDims;dir++)
	nSimdRanksCheck*=nSimdRanksPerDir(dir);
      if(nSimdRanksCheck!=nSimdRanks)
	{
	  std::ostringstream os;
	  os<<"nSimdRanksPerDir ";
	  printCoords(os,nRanksPerDir);
	  os<<" amounts to total "<<nSimdRanksCheck<<" different from expected "<<nSimdRanks;
	  
	  CRASH<<os.str();
	}
      
      glbIsNotSimdPartitioned=glbIsNotPartitioned and (nRanksPerDir==1);
      
      COMP_LOOP(SimdRank,simdRank,
		{
		  simdRankCoords(simdRank)=computeSimdRankCoords(simdRank);
		  simdRankLocOrigins(simdRank)=simdRankCoords(simdRank)*simdLocSides;
		});
      
#ifndef __CUDA_ARCH__
      LOGGER<<"SimdRanksLocOrigin: ";
      COMP_LOOP(SimdRank,simdRank,
		{
		  auto l=LOGGER;
		  l<<"simdRank "<<simdRank<<": ";
      		  printCoords(l,simdRankLocOrigins(simdRank));
		});
      
      LOGGER<<"SimdRankCoords: ";
      COMP_LOOP(SimdRank,simdRank,
		{
		  auto l=LOGGER;
		  l<<"simdRank "<<simdRank<<": ";
      		  printCoords(l,simdRankCoords(simdRank));
		});
#endif
      
      /////////////////////////////////////////////////////////////////
      
      // Set the simd local e/o lattice
      
      simdLocEoSides=simdLocSides/paritySides;
      simdLocEoVol=simdLocVol/2;
      LOGGER<<"Simd e/o local sides: ";
      printCoords(LOGGER,simdLocEoSides);
      
      assertIsPartitionable(simdLocSides,"simdLocal",paritySides,"simdRank");
      
      /////////////////////////////////////////////////////////////////
      
      // DynamicTens<SimdEoRepOfLocSite,GlbCoords,ExecSpace::HOST> glbCoordsOfParSimdLocEoSimdRank(simdLocEoVol);
      
      
      
      
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
