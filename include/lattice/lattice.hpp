#ifndef _LATTICE_HPP
#define _LATTICE_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file lattice/lattice.hpp

#include <expr/comps/compLoops.hpp>
#include <expr/nodes/dynamicTens.hpp>
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
  constexpr INLINE_FUNCTION HOST_DEVICE_ATTRIB				\
  NAME ## Coords compute ## NAME ## Coords(const NAME ## Site& name ## Site) const \
  {									\
    return U::computeCoordsOfSiteInBoxOfSides(name ## Site,SIDES);	\
  }									\
  									\
  /*! Computes the global site given the global coordinates */		\
  constexpr INLINE_FUNCTION HOST_DEVICE_ATTRIB				\
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
    
    /// A tensor with component direction
    template <typename F>
    using DirTens=typename U::template DirTens<F>;
    
    /// Directions in the space
    using Dir=
      typename U::Dir;
    
    /// Oriented direction
    using OriDir=
      std::tuple<Ori,Dir>;
    
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
      
      return out;
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
    
    /////////////////////////////////////////////////////////////////
    
    /// Rank Coordinate
    DECLARE_UNTRANSPOSABLE_COMP(RankCoord,int,0,rankCoord);
    
    /// Rank coordinates
    using RankCoords=DirTens<RankCoord>;
    
    /// Rank
    DECLARE_UNTRANSPOSABLE_COMP(Rank,int,0,rank);
    
    using RankSite=Rank;
    
    RankCoords nRanksPerDir;
    
    /// Coordinates of all ranks
    DynamicTens<OfComps<Rank>,RankCoords,ExecSpace::HOST> rankCoords;
    
    /// Coordinates of the present rank
    RankCoords thisRankCoords;
    
    /// Neighboring ranks
    StackTens<OfComps<Ori,Dir>,Rank> rankNeighbours;
    
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
    
    DECLARE_UNTRANSPOSABLE_COMP(NonLocSimdRank,int,0,nonLocSimdRank);
    
    using SimdRankSite=SimdRank;
    
    const SimdRank nSimdRanks=maxAvailableSimdSize;
    
    SimdRankCoords nSimdRanksPerDir;
    
    PROVIDE_ASSERT_SITE_COORD_COORDS(SimdRank,simdRank,nSimdRanks);
    
    PROVIDE_COORDS_OF_SITE_COMPUTER(SimdRank,simdRank,nSimdRanksPerDir);
    
    /// Coordinates of all simd ranks
    StackTens<OfComps<SimdRank>,SimdRankCoords> simdRankCoords;
    
    /// Neighbouring simd ranks
    StackTens<OfComps<SimdRank,Ori,Dir>,SimdRank> simdRankNeighbours;
    
    /// Non local simd ranks for each direction
    StackTens<OfComps<Ori,Dir>,std::vector<SimdRank>> nonLocSimdRanks;
    
    /// Number of non local simd ranks for each direction
    StackTens<OfComps<Dir>,NonLocSimdRank> nNonLocSimdRanks;
    
    /////////////////////////////////////////////////////////////////
    
    // // DECLARE_UNTRANSPOSABLE_COMP(HaloSite,int64_t,0,haloSite);
    
    // /// Computes the coordinates of the passed locSite
    // GlbCoords computeGlbCoords(const LocSite& locSite) const
    // {
    //   return originGlbCoords+computeLocCoords(locSite);
    // }
    
    /////////////////////////////////////////////////////////////////
    
    DECLARE_UNTRANSPOSABLE_COMP(ParityCoord,int,0,parityCoord);
    
    using ParityCoords=DirTens<ParityCoord>;
    
    DECLARE_UNTRANSPOSABLE_COMP(Parity,int,2,parity);
    
    /// Compute the parity of coords
    template <typename C>
    constexpr INLINE_FUNCTION HOST_DEVICE_ATTRIB
    static Parity coordsParity(const DirTens<C>& coords)
    {
      C sum=0;
      for(Dir dir=0;dir<NDims;dir++)
	sum+=coords(dir);
      
      return (~sum)%2;
    }
    
    /// Opposite parity
    static constexpr INLINE_FUNCTION Parity oppositeParity(const Parity& parity)
    {
      return 1-(~parity);
    }
    
    /// Sides of the parity grid
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
    
    /// Parity split direction
    Dir parityDir;
    
    /////////////////////////////////////////////////////////////////
    
    /// Different kind of sublattices
    enum class SubLatticeType{LOC,SIMD_LOC};
    
    /// Defines a sublattice
    template <SubLatticeType SL>
    struct SubLattice
    {
      /// Returns the sublattice type tag
      static constexpr const char* tag()
      {
	switch(SL)
	  {
	  case SubLatticeType::LOC:
	    return "loc";
	    break;
	  case SubLatticeType::SIMD_LOC:
	    return "simdLoc";
	    break;
	  }
      }
      
      DECLARE_UNTRANSPOSABLE_COMP(Coord,int,0,coord);
      
      using Coords=DirTens<Coord>;
      
      DECLARE_UNTRANSPOSABLE_COMP(Site,int64_t,0,site);
      
      Coords sides;
      
      Site vol;
      
      /// Set lattice sides and volume
      template <typename R>
      void setSidesAndVol(const GlbCoords& dividendSides,const DirTens<R>& nPerDir)
      {
	sides=~dividendSides/~nPerDir;
	LOGGER<<tag()<<" sides: ";
	printCoords(LOGGER,sides);
	
	assertIsPartitionable(dividendSides,"dividend",nPerDir,"divisor");
	
	vol=1;
	for(Dir dir=0;dir<NDims;dir++)
	  vol*=sides(dir);
      }
      
      /////////////////////////////////////////////////////////////////
      
      /// Returns the side in the bulk considering the wrapping
      Coords bulkSides;
      
      /// Total volume in the bulk
      Site bulkVol;
      
      /// Sets the bulk
      template <typename R>
      void setBulk(const DirTens<R>& nPerDir)
      {
	bulkSides=sides;
	
	bulkVol=1;
	
	COMP_LOOP(Dir,dir,
		  {
		    auto& b=bulkSides(dir);
		    
		    if(nPerDir(dir)>1)
		      b-=2;
		    
		    if(b<0)
		      b=0;
		    
		    bulkVol*=b;
		  });
	
	LOGGER<<tag()<<" bulk sides: ";
	printCoords(LOGGER,bulkSides);
	
	LOGGER<<tag()<<" bulk vol: "<<bulkVol;
      }
      
      /////////////////////////////////////////////////////////////////
      
      /// Surface sizes
      DirTens<Site> surfPerDir;
      
      /// Volume of the surface
      Site surf;
      
      /// Set the surfaces
      template <typename R>
      void setSurfaces(const DirTens<R>& nPerDir)
      {
	surf=vol-bulkVol;
      
	LOGGER<<tag()<<" Surf: "<<surf;
	
	COMP_LOOP(Dir,dir,
		  {
		    surfPerDir(dir)=
		      (nPerDir(dir)>1)?
		      (vol/sides(dir)):
		      Site(0);
		  });
	
	LOGGER<<tag()<<" Surf per dir: ";
	printCoords(LOGGER,surfPerDir);
      }
      
      /////////////////////////////////////////////////////////////////
      
      DECLARE_UNTRANSPOSABLE_COMP(EoCoord,int,0,eoCoord);
      
      using EoCoords=DirTens<EoCoord>;
      
      DECLARE_UNTRANSPOSABLE_COMP(EoSite,int64_t,0,eoSite);
      
      EoCoords eoSides;
      
      EoSite eoVol;
      
      /// Assert that an e/o site is in the allowed range
      constexpr INLINE_FUNCTION HOST_DEVICE_ATTRIB
      void assertIsEoSite(const EoSite& eoSite) const
      {
	assertIsInRange("eoSite",eoSite,eoVol);
      }
      
      PROVIDE_COORDS_OF_SITE_COMPUTER(Eo,eo,eoSides);
      
      /// Sets the e/o volume
      void setEoSidesAndVol(const Lattice& lattice)
      {
	// Set the e/o sides
	
	eoSides=~sides/~lattice.paritySides;
	LOGGER<<tag()<<" e/o sides: ";
	printCoords(LOGGER,eoSides);
	
	assertIsPartitionable(sides,tag(),lattice.paritySides,"parity");
	
	eoVol=
	  ~vol/2;
      }
      
      /////////////////////////////////////////////////////////////////
      
      /// Simd surface e/o sizes
      StackTens<OfComps<Parity,Ori,Dir>,EoSite> eoSurfPerDir;
      
      /// Volume of the simd e/o surface
      EoSite eoSurf;
      
      /// Set the e/o surface
      void setEoSurfaces(const Lattice& lattice)
      {
	eoSurf=
	  ~surf/2;
	
	eoSurfPerDir=
	  ~surfPerDir/2;
	
	if(surfPerDir(lattice.parityDir)%2)
	  {
	    eoSurfPerDir(BW,lattice.thisRankOriginParity,lattice.parityDir)++;
	    eoSurfPerDir(FW,oppositeParity(lattice.thisRankOriginParity),lattice.parityDir)++;
	  }
	
	LOGGER<<tag()<<" e/o surface: "<<eoSurf;
	for(Parity parity=0;parity<2;parity++)
	  {
	    LOGGER<<"Parity: "<<parity;
	    LOGGER<<tag()<<" e/o surface per ori dir: ";
	    for(Ori ori=0;ori<2;ori++)
	      for(Dir dir=0;dir<NDims;dir++)
		LOGGER<<" ori: "<<ori<<" dir: "<<dir<<" "<<eoSurfPerDir(parity,ori,dir);
	  }
      }
      
      /////////////////////////////////////////////////////////////////
      
      /// Halo size for each direction
      DirTens<Site> haloPerDir;
      
      /// Total halo
      Site halo;
      
      /// Offset where the halo for each orientation and direction starts, w.r.t locVol
      StackTens<OfComps<Ori,Dir>,Site> haloOffsets;
      
      /// Initialize the halo
      template <typename R>
      void setHalo(const DirTens<R>& nPerDir)
      {
	haloPerDir=surfPerDir*(nPerDir!=1);
	
	halo=0;
	for(Dir dir=0;dir<NDims;dir++)
	  halo+=haloPerDir(dir);
	halo*=2;
	
	/////////////////////////////////////////////////////////////////
	
	Site offset=0;
	
	for(Ori ori=0;ori<2;ori++)
	  for(Dir dir=0;dir<NDims;dir++)
	    {
	      haloOffsets(ori,dir)=offset;
	      offset+=haloPerDir(dir);
	    }
	
	if(offset!=halo)
	  CRASH<<"Not reached the total halo volume, "<<offset<<" while "<<halo<<" expected";
      }
      
    /// We loop on orientation, then direction, to scan all sites that
    /// need to be copied elsewhere. The computed halo site points to
    /// where the the site should reside if we wanted to fill a
    /// fictitious halo for current rank alone
    template <typename T>
    void precomputeHaloFiller(T& tab)
    {
      tab.allocate(eoHalo);
      
      for(Parity parity=0;parity<2;parity++)
	{
	  std::vector<std::tuple<EoSite,EoSite,Ori,Dir>> list;
	  
	  for(EoSite eoSite=0;eoSite<eoVol;eoSite++)
	    for(Ori ori=0;ori<2;ori++)
	      for(Dir dir=0;dir<NDims;dir++)
		{
		  const EoSite neighEoSite=
		    eoNeighbours(parity,eoSite,ori,dir);
		  
		  const EoSite excess=
		    (neighEoSite-eoVol);
		  
		  if(excess>=0)
		    {
		      const EoSite flippingOriOffset=
			eoHaloOffsets(parity,oppositeOri(ori),dir)-eoHaloOffsets(parity,ori,dir);
		      list.emplace_back(eoSite,excess+flippingOriOffset,ori,dir);
		      
		      LOGGER<<"parity "<<parity<<" eoSite "<<eoSite<<" ori "<<ori<<" dir "<<dir<<" must be copied in "<<excess;
		    }
		}
	  
	  std::sort(list.begin(),list.end());
	  
	  if((int64_t)list.size()!=eoHalo)
	    CRASH<<"list of sites that must be copied to other ranks e/o halo has size"<<list.size()<<" not agreeing with e/o halo size "<<eoHalo;
	  
	  for(int64_t i=0;i<(int64_t)list.size();i++)
	    tab(parity,EoSite(i))=list[i];
	}
    }
    
    /////////////////////////////////////////////////////////////////
    
    // /// Compute the site of given coords in the given ori,dir halo
    //   Site haloSiteOfCoords(const Coords& cs,const Ori& ori,const Dir& dir) const
    //   {
    // 	/// Returned site
    // 	Site out=0;
	
    // 	COMP_LOOP(Dir,mu,
    // 		  {
    // 		    if(mu!=dir)
    // 		      out=out*sides()(mu)+cs(mu);
    // 		  });
	
    // 	return haloOffset(ori,dir)+out;
    //   }
      
    //   /// Returns the orientation and direction of a point in the halo
    //   OriDir oriDirOfHaloSite(const Site& haloSite) const
    //   {
    // 	assertIsHaloSite(haloSite);
	
    // 	const Ori ori=(Ori::Index)(haloSite/(halo()/2));
	
    // 	assertIsOri(ori);
	
    // 	Dir dir=0;
	
    // 	while(dir<NDims-1 and haloOffset(ori,dir+1)<=haloSite)
    // 	  dir++;
	
    // 	assertIsDir(dir);
	
    // 	return {ori,dir};
    //   }
      
      
      /// E/o halo for each direction
      StackTens<OfComps<Ori,Parity,Dir>,EoSite> eoHaloPerDir;
      
      /// Total e/o halo
      EoSite eoHalo;
      
      /// Offset where the e/o halo for each orientation and direction starts, w.r.t eoLocVol
      StackTens<OfComps<Parity,Ori,Dir>,EoSite> eoHaloOffsets;
      
      /// Set the e/o halo
      void setEoHalo(const Lattice& lattice)
      {
	eoHalo=
	  ~halo/2;
	
	eoHaloPerDir=~haloPerDir/2;
	if(haloPerDir(lattice.parityDir)%2)
	  {
	    eoHaloPerDir(BW,oppositeParity(lattice.thisRankOriginParity),lattice.parityDir)++;
	    eoHaloPerDir(FW,lattice.thisRankOriginParity,lattice.parityDir)++;
	  }
	
	for(Parity parity=0;parity<2;parity++)
	  {
	    EoSite offset=0;
	    
	    for(Ori ori=0;ori<2;ori++)
	      for(Dir dir=0;dir<NDims;dir++)
		{
		  eoHaloOffsets(parity,ori,dir)=offset;
		  offset+=eoHaloPerDir(ori,parity,dir);
		}
	    
	    if(offset!=eoHalo)
	      CRASH<<"Not reached the total e/o halo volume, "<<offset<<" while "<<eoHalo<<" expected";
	  }
	
	LOGGER<<tag()<<" e/o halo: "<<eoHalo;
	for(Parity parity=0;parity<2;parity++)
	  {
	    LOGGER<<"Parity: "<<parity;
	    LOGGER<<tag()<<" e/o halo per ori dir: ";
	    for(Ori ori=0;ori<2;ori++)
	      for(Dir dir=0;dir<NDims;dir++)
		LOGGER<<" ori: "<<ori<<" dir: "<<dir<<" "<<eoHaloPerDir(parity,ori,dir);
	  }
      }
      
      /////////////////////////////////////////////////////////////////
      
      /// Eo neighbour of a site of a given parity, pointing to an opposite-parity site
      DynamicTens<OfComps<Parity,EoSite,Ori,Dir>,EoSite,ExecSpace::HOST> eoNeighbours;
      
      /// Sets the neighbours
      template <typename R>
      void setEoNeighbours(const Lattice& lattice,const DirTens<R>& nPerDir)
      {
	auto allocateSize=std::make_tuple(eoVol);
	
	eoNeighbours.allocate(allocateSize);
	
	loopOnAllComps<CompsList<Parity,EoSite>>(allocateSize,[this,&lattice,&nPerDir](const Parity& parity,const EoSite& eoSite)
	{
	  /// Parity of neighbors
	  const Parity oppoParity=
	    oppositeParity(parity);
	  
	  const EoCoords eoCoords=
	    computeEoCoords(eoSite);
	  
	  const Coords coordsWithUnfixedParity=
	    ~eoCoords*~lattice.paritySides;
	  
	  // LOGGER<<"Setting neigh of site of parity "<<parity<<" eoSite "<<eoSite<<": ";
	  
	  const Parity resultingParity=
	    (coordsParity(coordsWithUnfixedParity)+lattice.thisRankOriginParity)%2;
	  
	  /// Check if parity needs to be adjusted
	  const Parity neededParity=
	    (resultingParity+parity)%2;
	  
	  const Coords coords=
	    coordsWithUnfixedParity+lattice.parityCoords(neededParity);
	  
	  // {
	  //   auto l=LOGGER;
	  //   l<<" coords: ";
	  //   printCoords(l,coords);
	  // }
	  
	  for(Ori ori=0;ori<2;ori++)
	    for(Dir dir=0;dir<NDims;dir++)
	      {
		// LOGGER<<"  ori "<<ori<<" dir "<<dir;
		
		/// Trivial shift of current site coord in the direction dir and orientation ori
		const Coords triviallyShiftedCoords
		  ([extDir=dir,ori,&coords,this,&nPerDir](const Dir& dir)
		  {
		    Coord c=coords(dir);
		    
		    if(extDir==dir)
		      {
			c+=moveOffset[ori];
			if(nPerDir(extDir)==1)
			  c=safeModulo(c,sides(extDir));
		      }
		    
		    return c;
		  });
		
		const Coord& triviallyShiftedCoord=
		  triviallyShiftedCoords(dir);
		
		// {
		//   auto l=LOGGER;
		//   l<<"  triviallyShiftedCoords: ";
		//   printCoords(l,triviallyShiftedCoords);
		// }
		
		EoSite& n=eoNeighbours(parity,eoSite,ori,dir);
		
		if(triviallyShiftedCoord<0 or triviallyShiftedCoord>=sides(dir))
		  {
		    // LOGGER<<"  exceeding local sides";
		    
		    /// Sets to zero the shifted coordinate in the excess direction
		    const Coords shiftedCoords
		      ([extDir=dir,triviallyShiftedCoords](const Dir& dir)
		      {
			return (dir==extDir)?Coord(0):triviallyShiftedCoords(dir);
		      });
		    
		    // {
		    //   auto l=LOGGER;
		    //   l<<"  shiftedCoords: ";
		    //   printCoords(l,shiftedCoords);
		    // }
		    
		    const Coords haloSides
		      ([this,extDir=dir](const Dir& dir)
		      {
			return (dir==extDir)?Coord(1):sides(dir);
		      });
		    const EoSite posInEoHalo=
		      ~U::template computeSiteOfCoordsInBoxOfSides<Site>(shiftedCoords,haloSides)/2;
		    
		    // {
		    //   auto l=LOGGER;
		    //   l<<"  computing neigh in halo of sides: ";
		    //   printCoords(l,haloSides);
		    //   l<<" result: "<<eoHaloOffsets(oppoParity,ori,dir)+posInEoHalo;
		    // }
		    
		    if(posInEoHalo>=eoHaloPerDir(oppoParity,ori,dir))
		      CRASH<<"Neighbour for site of parity "<<parity<<" in the orientation "<<ori<<" direction "<<dir<<
			" is "<<posInEoHalo<<" larger than maximal size "<<eoHaloPerDir(oppoParity,ori,dir);
		    
		    n=eoVol+
		      eoHaloOffsets(oppoParity,ori,dir)+
		      posInEoHalo;
		    
		    // LOGGER<<" pointing to "<<n;
		  }
		else
		  n=computeEoSite(~triviallyShiftedCoords/~lattice.paritySides);
	      }
	});
      }
      
      /////////////////////////////////////////////////////////////////
      
      template <typename R>
      void init(const Lattice& lattice,const DirTens<R>& nPerDir)
      {
	setSidesAndVol(lattice.glbSides,nPerDir);
	
	setBulk(nPerDir);
	
	setSurfaces(nPerDir);
	
	setHalo(nPerDir);
	
	setEoSidesAndVol(lattice);
	
	setEoSurfaces(lattice);
	
	setEoHalo(lattice);
	
	setEoNeighbours(lattice,nPerDir);
      }
    };
    
    /////////////////////////////////////////////////////////////////
    
    using Loc=
      SubLattice<SubLatticeType::LOC>;
    
    using SimdLoc=
      SubLattice<SubLatticeType::SIMD_LOC>;
    
    /////////////////////////////////////////////////////////////////
    
    Loc loc;
    
    SimdLoc simdLoc;
    
    /////////////////////////////////////////////////////////////////
    
#define IMPORT_SUBLATTICE_TYPES(SUB)		\
						\
    using SUB ## Coord=				\
      typename SUB::Coord;			\
						\
    using SUB ## Coords=			\
      typename SUB::Coords;			\
						\
    using SUB ## EoSite=			\
      typename SUB::EoSite;			\
						\
    using SUB ## Site=				\
      typename SUB::Site;			\
						\
    using SUB ## EoCoords=			\
      typename SUB::EoCoords;			\
						\
    using SUB ## EoCoord=			\
      typename SUB::EoCoord
    
    IMPORT_SUBLATTICE_TYPES(Loc);
    
    IMPORT_SUBLATTICE_TYPES(SimdLoc);
    
#undef IMPORT_SUBLATTICE_TYPES
    
    /////////////////////////////////////////////////////////////////
    
    DynamicTens<OfComps<Parity,LocEoSite>,std::tuple<LocEoSite,LocEoSite,Ori,Dir>,ExecSpace::HOST> eoHaloFillerTable;
    
    DynamicTens<OfComps<Parity,SimdLocEoSite>,std::tuple<SimdLocEoSite,SimdLocEoSite,Ori,Dir>,ExecSpace::HOST> simdEoHaloFillerTable;
    
    //DynamicTens<OfComps<Parity,LocEoSite>,std::tuple<SimdLocEoSite,SimdRank,LocEoSite>,ExecSpace::HOST> simdEoHaloNonLocFillerTable;
    
    /////////////////////////////////////////////////////////////////
    
    /// Local origin of the simd rank
    StackTens<OfComps<SimdRank>,LocCoords> simdRankLocOrigins;
    
    // /// Surface e/o sizes
    // StackTens<OfComps<Parity,Ori,Dir>,LocEoSite> locEoSurfPerDir;
    
    // /// Volume of the e/o surface
    // StackTens<OfComps<Parity>,LocEoSite> locEoSurf;
    
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
      /// Computes the local coordinates in the simd e/o lattice
      const SimdLocEoCoords simdLocEoCoords=
	simdLoc.computeEoCoords(simdLocEoSite);
      
      /// Coordinates of the site, not having yet fixed the parity
      const SimdLocCoords simdLocCoordsWithUnfixedParity=
	~simdLocEoCoords*~paritySides;
      
      /// Local coordinates of the simdRank origin
      const LocCoords simdRankLocOrigin=
	simdRankLocOrigins(simdRank);
      
      /// Global coordinates of the site, not having yet fixed the parity
      const GlbCoords glbCoordsWithUnfixedParity=
	thisRankOriginGlbCoords+simdRankLocOrigin+simdLocCoordsWithUnfixedParity;
      
      /// Computes the parity of the resulting coordinates
      const Parity glbParity=
	coordsParity(glbCoordsWithUnfixedParity);
      
      /// Check if parity needs to be adjusted
      const Parity neededParity=
	(glbParity+parity)%2;
      
      return glbCoordsWithUnfixedParity+parityCoords(neededParity);
    }
    
    /// Returns the global coordinates of a given loceo
    GlbCoords glbCoordsOfLoceo(const Parity& parity,const LocEoSite& locEoSite)
    {
      const LocEoCoords locEoCoords=
	loc.computeEoCoords(locEoSite);
      
      const GlbCoords coordsWithUnfixedParity=
	thisRankOriginGlbCoords+locEoCoords*paritySides;
      
      const Parity resultingParity=
	coordsParity(coordsWithUnfixedParity)%2;
      
      /// Check if parity needs to be adjusted
      const Parity neededParity=
	(resultingParity+parity)%2;
      
      const GlbCoords glbCoords=
	coordsWithUnfixedParity+parityCoords(neededParity);
      
      return glbCoords;
    }
    
    /// Components needed to identify a gloval site within a e/o simd representation
    using SimdEoRepOfGlbSite=OfComps<Rank,Parity,typename SimdLoc::EoSite,SimdRank>;
    
    SimdEoRepOfGlbSite computeSimdEoRepOfGlbCoords(const GlbCoords& glbCoords) const
    {
      const RankCoords rankCoords=
	~glbCoords/~loc.sides;
      
      const Rank rank=
	computeRankSite(rankCoords);
      
      const LocCoords locCoords=
	(~glbCoords)-(~originGlbCoords(rank));
      
      const Parity parity=
	coordsParity(glbCoords);
      
      const SimdLocCoords simdLocCoords=
	~locCoords%simdLoc.sides;
      
      const SimdRankCoords simdRankCoords=
	~locCoords/~simdLoc.sides;
      
      const SimdLocEoCoords simdLocEoCoords=
	~simdLocCoords/~paritySides;
      
      const SimdLocEoSite simdLocEoSite=
	simdLoc.computeEoSite(simdLocEoCoords);
      
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
    
    // /// Initializes all directions as wrapping
    // static constexpr DirTens<bool> allDirWraps()
    // {
    //   return true;
    // }
    
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
      DirTens<SimdRank> nGlbSimdRanks=
	~nRanksPerDir*
	~nSimdRanksPerDir;
      
      // Set the global and rank lattices
      
      LOGGER<<"glbSides: ";
      printCoords(LOGGER,glbSides);
      
      glbVol=1;
      for(Dir dir=0;dir<NDims;dir++)
	glbVol*=glbSides(dir);
      
      LOGGER<<"glbVol: "<<glbVol;
      
      glbIsNotPartitioned=(nRanksPerDir==1);
      
      glbIsNotSimdPartitioned=glbIsNotPartitioned and (nSimdRanksPerDir==1);
      
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
	  originGlbCoords(rank)=~rankCoords(rank)*glbSides/~nRanksPerDir; // Don't use loc.sides since not yet initialized here
	  originParity(rank)=coordsParity(originGlbCoords(rank));
	}
      
      thisRankCoords=rankCoords(Rank(Mpi::rank));
      thisRankOriginGlbCoords=originGlbCoords(Rank(Mpi::rank));
      thisRankOriginParity=originParity(Rank(Mpi::rank));
      
      LOGGER<<"Neighbours rank:";
      for(Ori ori=0;ori<2;ori++)
	for(Dir dir=0;dir<NDims;dir++)
	  {
	    const RankCoords neighRankCoords=
	      (thisRankCoords+moveOffset[ori]*U::template versor<RankCoord>(dir)+nRanksPerDir)%nRanksPerDir;
	    
	    const Rank neighRank=
	      U::template computeSiteOfCoordsInBoxOfSides<Rank>(neighRankCoords,nRanksPerDir);
	    
	    LOGGER<<" ori "<<ori<<" dir "<<dir<<": "<<neighRank;
	    
	    rankNeighbours(ori,dir)=neighRank;
	  }
      
      LOGGER<<"Coordinates of the rank: ";
      printCoords(LOGGER,thisRankCoords);
      
      LOGGER<<"Coordinates of the origin: ";
      printCoords(LOGGER,thisRankOriginGlbCoords);
      
      /////////////////////////////////////////////////////////////////
      
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
      
      COMP_LOOP(SimdRank,simdRank,
		{
		  simdRankCoords(simdRank)=computeSimdRankCoords(simdRank);
		  simdRankLocOrigins(simdRank)=
		    ~simdRankCoords(simdRank)*~glbSides/~nGlbSimdRanks; // Don't use simdLoc.sides since not yet initialized here
		  
		  const Parity par=coordsParity(simdRankLocOrigins(simdRank));
		  if(par!=0)
		    CRASH<<"Simd Rank origins do not have the same parity, please set the simd local sides to an even number";
		});
      
      LOGGER<<"Neighbours simd rank:";
      for(Ori ori=0;ori<2;ori++)
	for(Dir dir=0;dir<NDims;dir++)
	  {
	    // Computed twice, so reset twice
	    nNonLocSimdRanks(dir)=0;
	    
	    LOGGER<<" ori "<<ori<<" dir "<<dir;
	    
	    for(SimdRank simdRank=0;simdRank<nSimdRanks;simdRank++)
	      {
		const SimdRankCoords triviallyShiftedCoords=
		  simdRankCoords(simdRank)+moveOffset[ori]*U::template versor<SimdRankCoord>(dir);
		
		const SimdRankCoords simdNeighRankCoords=
		  (triviallyShiftedCoords+nSimdRanksPerDir)%nSimdRanksPerDir;
		
		const SimdRank simdNeighRank=
		  U::template computeSiteOfCoordsInBoxOfSides<SimdRank>(simdNeighRankCoords,nSimdRanksPerDir);
		
		LOGGER<<"  simd rank "<<simdRank<<": "<<simdNeighRank;
		
		if(triviallyShiftedCoords(dir)!=simdNeighRankCoords(dir) and nRanksPerDir(dir)>1)
		  {
		    nonLocSimdRanks(ori,dir).push_back(simdRank);
		    LOGGER<<" simdRank "<<simdRank<<" is non local, stored as #"<<nNonLocSimdRanks(dir);
		    nNonLocSimdRanks(dir)++;
		  }
		
		simdRankNeighbours(simdRank,ori,dir)=simdNeighRank;
	      }
	}
      
      for(Dir dir=0;dir<NDims;dir++)
	LOGGER<<" nSimdRanksPerDir: "<<nSimdRanksPerDir(dir);
      
      for(Dir dir=0;dir<NDims;dir++)
	{
	  LOGGER<<" nonlocSimdRanks(dir="<<dir<<")="<<nNonLocSimdRanks(dir);
	  for(Ori ori=0;ori<2;ori++)
	    {
	      LOGGER<<" orientation "<<ori;
	      for(const SimdRank& sr :nonLocSimdRanks(ori,dir))
		LOGGER<<"  "<<sr;
	    }
	}
      
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
      
      assertIsPartitionable(loc.sides,"local",paritySides,"parity");
      
      /////////////////////////////////////////////////////////////////
      
      loc.init(*this,nRanksPerDir);
      
      loc.precomputeHaloFiller(eoHaloFillerTable);
      
      simdLoc.init(*this,nGlbSimdRanks);
      
      simdLoc.precomputeHaloFiller(simdEoHaloFillerTable);
      
      // LOGGER<<"Scanning surf of loc, converting to simd rep";
      
      //simdEoHaloNonLocFillerTable.allocate(loc.eoHalo);
      
      // for(Parity parity=0;parity<2;parity++)
      // 	{
      // 	  // LOGGER<<" Parity "<<parity;
      // 	  std::vector<std::tuple<SimdLocEoSite,SimdRank,LocEoSite>> list;
	  
      // 	  loopOnAllComps<CompsList<LocEoSite>>(std::make_tuple(loc.eoHalo),
      // 					       [&list,parity,this](const LocEoSite& siteRemappingId) MUTABLE_INLINE_ATTRIBUTE
      // 					       {
      // 						 const auto& r=eoHaloFillerTable(parity,siteRemappingId);
      // 						 const LocEoSite source=std::get<0>(r);
      // 						 const LocEoSite dest=std::get<1>(r);
      // 						 // const Ori ori=std::get<2>(r);
      // 						 // const Dir dir=std::get<3>(r);
						 
      // 						 const GlbCoords glbCoords=glbCoordsOfLoceo(parity,source);
      // 						 const auto [rankCheck,parityCheck,simdLocEoSite,simdRankCheck]=
      // 						   computeSimdEoRepOfGlbCoords(glbCoords);
						 
      // 						 // LOGGER<<" ori: "<<ori;
      // 						 // LOGGER<<" dir: "<<dir;
      // 						 // LOGGER<<" rank: "<<rankCheck<<" expected: "<<Mpi::rank;
      // 						 // LOGGER<<" parity: "<<parityCheck<<" expected: "<<parity;
      // 						 // LOGGER<<" simdLocEoSite: "<<simdLocEoSite;
      // 						 // LOGGER<<" simdRank: "<<simdRankCheck;
      // 						 // LOGGER<<" dest CCC: "<<dest;
						 
      // 						 list.emplace_back(simdLocEoSite,simdRankCheck,dest);
      // 					       });
	  
      // 	  // std::sort(list.begin(),list.end());
	  
      // 	  if((int64_t)list.size()!=loc.eoHalo)
      // 	    CRASH<<"list of sites that must be copied to other ranks e/o halo has size "<<list.size()<<" not agreeing with e/o halo size "<<loc.eoHalo;
	  
      // 	  // for(int64_t i=0;i<(int64_t)list.size();i++)
      // 	  //   simdEoHaloNonLocFillerTable(parity,LocEoSite(i))=list[i];
      // 	}
      
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
