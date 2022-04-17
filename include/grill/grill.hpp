#ifndef _GRILL_HPP
#define _GRILL_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file grill/grill.hpp

#include <expr/nodes/stackTens.hpp>
#include <grill/universe.hpp>
#include <resources/mathOperations.hpp>

namespace esnort
{
  /// Hashable properties of a \c Grill
  ///
  /// Hashing version
  template <int NDims,
	    int...I>
  template <typename GT>       // Type of coordinate values
  struct Universe<NDims,std::integer_sequence<int,I...>>::
  HashableTableProvider<GT,
			true>
  {
    using Site=typename GT::Site;
    
    using HaloSite=typename GT::HaloSite;
    
    using Coord=typename GT::Coord;
    
    /// Hashed coords of all sites
    DynamicTens<OfComps<Site,Dir>,Coord,ExecSpace::HOST> coordsOfSitesHashTable;
    
    /// Hashed neighbors
    DynamicTens<OfComps<Site,Ori,Dir>,Site,ExecSpace::HOST> neighsOfSitesHashTable;
    
    /// Wrapping surface site corresponding to a given halo
    DynamicTens<OfComps<HaloSite>,Site,ExecSpace::HOST> surfSiteOfHaloSitesHashTable;
  };
  
  /// Hashable properties of a \c Grill
  ///
  /// Not-Hashing version
  template <int NDims,
	    int...I>
  template <typename GT>          // Type of coordinate values
  struct Universe<NDims,std::integer_sequence<int,I...>>::
  HashableTableProvider<GT,
			false>
  {
  };
  
  /// A grill
  template <int NDims,
	    int...I>
  template <typename SiteIntType,
	    int NSites,
	    bool Hashing,
	    const char* Name>
  struct Universe<NDims,std::integer_sequence<int,I...>>::Grill :
  GrillTypes<SiteIntType,NSites,Name>,
    HashableTableProvider<GrillTypes<SiteIntType,NSites,Name>,Hashing>
  {
    /////////////////////////////////////////////////////////////////
    
    /// Surface sizes
    DirTens<Site> _surfSizes;
    
    /// Surface size, constant access
    INLINE_FUNCTION HOST_DEVICE_ATTRIB
    const Site& surfSize(const Dir& dir) const
    {
      return _surfSizes(dir);
    }
    
    /////////////////////////////////////////////////////////////////
    
    /// Determine if the lattice wraps in a given direction
    DirTens<bool> _wrapping;
    
    /// Gets the wrapping in a given direction, constant access
    INLINE_FUNCTION HOST_DEVICE_ATTRIB
    const bool& wrapping(const Dir& dir) const
    {
      return _wrapping(dir);
    }
    
    // /// Set the wrapping to a given direction
    // INLINE_FUNCTION HOST_DEVICE_ATTRIB
    // void setWrappingTo(const Dir& dir,const bool& b)
    // {
    // 	_wrapping(dir)=b;
    // }
    
    // /// Set the wrapping for all directions
    // INLINE_FUNCTION HOST_DEVICE_ATTRIB
    // void setWrappingAllDirsTo(const bool& b)
    // {
    // 	compLoop<Dir>([this,&b](const Dir& dir) INLINE_ATTRIBUTE
    // 	{
    // 	  setWrappingTo(dir,b);
    // 	});
    // }
    
    /// Returns the side in the bulk considering the wrapping
    Coord bulkSide(const Dir& dir) const
    {
      Coord bulkSide=side(dir);
      
      if(wrapping(dir))
	return bulkSide;
      else
	if(bulkSide<=2)
	  return 0;
	else
	  return bulkSide-2;
    }
    
    /////////////////////////////////////////////////////////////////
    
    /// Halo sizes
    DirTens<HaloSite> _haloSizes;
    
    /// Halo size in a given direction, constant access
    INLINE_FUNCTION HOST_DEVICE_ATTRIB
    const HaloSite& haloSize(const Dir& dir) const
    {
      return _haloSizes(dir);
    }
    
    /// Total halo volume
    HaloSite _totHaloVol;
    
    /// Total halo volume, constant access
    INLINE_FUNCTION HOST_DEVICE_ATTRIB
    const HaloSite& totHaloVol() const
    {
      return _totHaloVol;
    }
    
    /// Initialize halo sizes
    void initHaloSizes()
    {
      _haloSizes=_surfSizes-_wrapping*_surfSizes;
      
      _totHaloVol=2*(_haloSizes(Dir(I))+...);
    }
    
    /// Assert that a halo site
    INLINE_FUNCTION HOST_DEVICE_ATTRIB
    void assertIsHaloSite(const HaloSite& halo) const
    {
      assertIsInRange("coord",halo,totHaloVol());
    }
    
    /////////////////////////////////////////////////////////////////
    
    /// Volume in the bulk
    Site _bulkVol;
    
    /// Volume in the bulk, const access
    const Site bulkVol() const
    {
      return _bulkVol;
    }
    
    /// Initialize the bulk volume
    void initBulkVol()
    {
      _bulkVol=(((Site)bulkSide(I)())*...);
    }
    
    /////////////////////////////////////////////////////////////////
    
    /// Volume not in the bulk
    Site _surfVol;
    
    /// Volume not in the bulk, const access
    const Site surfVol() const
    {
      return _surfVol;
    }
    
    /// Initialize the volume of the lattice part not in the bulk
    void initSurfVol()
    {
      _surfVol=_vol-_bulkVol;
    }
    
    /////////////////////////////////////////////////////////////////
    
    /// Offset where the halo for each orientation and direction starts
    StackTens<OfComps<Ori,Dir>,HaloSite> _haloOffsets;
    
    /// Constant access the offset where the halo for each
    /// orientation and direction starts. First offset is 0
    const HaloSite& haloOffset(const Ori& ori,const Dir& dir) const
    {
      return _haloOffsets(ori,dir);
    }
    
    /// Initializes the site where the halo of each direction starts
    void initHaloOffsets()
    {
      HaloSite offset=0;
      
      for(Ori ori=0;ori<2;ori++)
	for(Dir dir=0;dir<NDims;dir++)
	  {
	    _haloOffsets(ori,dir)=offset;
	    offset+=_haloSizes(dir);
	  }
      
      if(offset!=totHaloVol())
	CRASH<<"Not reached the total halo volume, "<<offset<<" while "<<totHaloVol()<<" expected";
    }
    
    /////////////////////////////////////////////////////////////////
    
    /// Set the sides and derived quantities
    void setSidesAndWrapping(const Coords& sides,
			     const DirTens<bool>& wrapping,
			     const bool& fillHashTables=true)
    {
      _vol=(sides(Dir(I))*...);
      _sides=sides;
      _wrapping=wrapping;
      COMP_LOOP(Dir,dir,
		{
		  _surfSizes(dir)=_vol/sides(dir);
		});
      
      initHaloSizes();
      initBulkVol();
      initSurfVol();
      initHaloOffsets();
      
      if constexpr(Hashing)
	if(fillHashTables)
	  this->fillHashTables(vol());
    }
    
    /////////////////////////////////////////////////////////////////
    
    /// Compute the site of given coords in the given ori,dir halo
    HaloSite haloSiteOfCoords(const Coords& cs,const Ori& ori,const Dir& dir) const
    {
      /// Returned site
      HaloSite out=0;
      
      COMP_LOOP(Dir,mu,
		{
		  if(mu!=dir)
		    out=out*sides()(mu)+cs(mu);
		});
      
      return haloOffset(ori,dir)+out;
    }
    
    /// Returns the orientation and direction of a point in the halo
    auto oriDirOfHaloSite(const HaloSite& haloSite) const
    {
      assertIsHaloSite(haloSite);
      
      const Ori ori=(Ori::Index)(haloSite/(totHaloVol()/2));
      
      assertIsOri(ori);
      
      Dir dir=0;
      
      while(dir<NDims-1 and haloOffset(ori,dir+1)<=haloSite)
	dir++;
      
      assertIsDir(dir);
      
      return std::make_tuple(ori,dir);
    }
    
    /// Computes the (wrapping) surface site corresponding to a given halo
    Site computeSurfSiteOfHaloSite(HaloSite haloSite /* don't make const */)
    {
      assertIsHaloSite(haloSite);
      
      const auto [ori,dir]=oriDirOfHaloSite(haloSite);
      
      haloSite-=haloOffset(ori,dir);
      
      Coords haloSides=sides();
      haloSides(dir)=1;
      
      Coords c=computeCoordsOfSiteInBoxOfSize(haloSite,haloSides);
      
      c(dir)=safeModulo(c(dir)+moveOffset[ori],side(dir)());
      
      return siteOfCoords(c);
    }
    
    /// Wrapping surface site corresponding to a given halo
    Site surfSiteOfHaloSite(const HaloSite& haloSite)
    {
      assertIsHaloSite(haloSite);
      
      if constexpr(Hashing)
	return
	  (this->surfSiteOfHaloSitesHashTable(haloSite));
      else
	return
	  computeSurfSiteOfHaloSite(haloSite);
    }
    
    /////////////////////////////////////////////////////////////////
    
    /// Default constructor
    Grill()
    {
    }
    
    /// Create from sides
    Grill(const Coords& _sides,
	  const DirTens<bool>& wrapping,
	  const bool& fillHashTables=true)
    {
      setSidesAndWrapping(_sides,
			  wrapping,
			  fillHashTables);
    }
    
    /// Fill all the HashTables
    void fillHashTables(const Site& size)
    {
      static_assert(Hashing,"Cannot initialize the hash table if not present");
      
      if constexpr(Hashing)
	{
	  this->coordsOfSitesHashTable.allocate(std::make_tuple(size));
	  this->neighsOfSitesHashTable.allocate(std::make_tuple(size));
	  this->surfSiteOfHaloSitesHashTable.allocate(std::make_tuple(totHaloVol()));
	  
	  // Set the hash table of coordinates of all sites
	  forAllSites([&](const Site& site)
	  {
	    this->coordsOfSitesHashTable(site)=
	      computeCoordsOfSite(site);
	  });
	  
	  // Set the hash table of surface site of each halo site
	  forAllHaloSites([&](const HaloSite& haloSite)
	  {
	    this->surfSiteOfHaloSitesHashTable(haloSite)=
	      computeSurfSiteOfHaloSite(haloSite);
	  });
	  
	  loopOnAllComps<CompsList<Site,Ori,Dir>>(std::make_tuple(vol()),
						  [&](const Site& site,
						      const Ori& ori,
						      const Dir& dir)
						  {
						    this->neighsOfSitesHashTable(site,ori,dir)=
						      computeNeighOfSite(site,ori,dir);
						  });
	}
    }
    
    /////////////////////////////////////////////////////////////////
    
    /// Get the coords of given site
    constexpr INLINE_FUNCTION HOST_DEVICE_ATTRIB
    decltype(auto) coordsOfSite(const Site& site) const
    {
      assertIsSite(site);
      
      if constexpr(Hashing)
	return
	  (this->coordsOfSitesHashTable(site));
      else
	return
	  computeCoordsOfSite(site);
    }
    
    /// Return the neighbor in the given oriented dir
    constexpr INLINE_FUNCTION HOST_DEVICE_ATTRIB
    decltype(auto) neighOfSite(const Site& site,
			       const Ori& ori,
			       const Dir& dir)
      const
    {
      assertIsSite(site);
      assertIsOri(ori);
      assertIsDir(dir);
      
      if constexpr(Hashing)
	return
	  (this->neighsOfSitesHashTable(site,ori,dir));
      else
	return
	  computeNeighOfSite(site,ori,dir);
    }
    
    /// Loop on all sites calling the passed function
    template <typename F>
    constexpr INLINE_FUNCTION HOST_DEVICE_ATTRIB
    void forAllSites(F f) const
    {
      loopOnAllComps<CompsList<Site>>(std::make_tuple(vol()),f);
    }
    
    /// Loop on all halo sites calling the passed function
    template <typename F>
    constexpr INLINE_FUNCTION HOST_DEVICE_ATTRIB
    void forAllHaloSites(F f) const
    {
      loopOnAllComps<CompsList<HaloSite>>(std::make_tuple(totHaloVol()),f);
    }
    
    /// Returns the coordinates shifted in the asked direction
    ///
    /// Periodic boundary conditions are always assumed
    template <typename CoordsProvider>
    Coords shiftedCoords(const CoordsProvider& in,
			 const Ori& ori,
			 const Dir& dir,
			 const Coord amount=1)
      const
    {
      /// Returned coordinates
      Coords out;
      
      /// Offset to add
      const Coord offset=
	moveOffset[ori]*amount;
      
      /// Destination not considering wrap
      const Coord rawDest=
	in(dir)+offset;
      
      const Coord& side=
	sides()(dir);
      
      /// Actual destintion
      const Coord dest=
	safeModulo(rawDest,side);
      
      COMP_LOOP(Dir,mu,
		{
		  if(mu!=dir)
		    out(mu)=
		      in(mu);
		  else
		    out(mu)=
		      dest;
		});
      
      return
	out;
    }
    
    /// Compute the site of given coords
    Site siteOfCoords(const Coords& cs) const ///< Coordinates of the site
    {
      assertAreCoords(cs);
      
      /// Returned site
      Site out=0;
      
      COMP_LOOP(Dir,mu,
		{
		  out=out*sides()(mu)+cs(mu);
		});
      
      return out;
    }
    
    /// Compute the coordinate of site
    DirTens<Coord> computeCoordsOfSite(const Site& site) const
    {
      assertIsSite(site);
      
      return computeCoordsOfSiteInBoxOfSize(site,sides());
    }
    
    /// Compute the neighbor in the oriented direction oriDir of site i
    Site computeNeighOfSite(const Site& site,   ///< Site
			    const Ori& ori,     ///< Orientation
			    const Dir& dir)     ///< Direction
      const
    {
      assertIsSite(site);
      assertIsOri(ori);
      assertIsDir(dir);
      
      /// Current site coords
      decltype(auto) siteCoords=this->coordsOfSite(site);
      
      /// Trivial shift of current site coord in the direction dir and orientation ori
      const Coord triviallyShftedCoord=siteCoords(dir)+moveOffset[ori];
      
      /// Shifted coords including wrap
      Coords neighCoords=shiftedCoords(siteCoords,ori,dir);
      
      const bool neighIsNotOnHalo=
	wrapping(dir) or neighCoords(dir)==triviallyShftedCoord;
      
      // LOGGER<<site<<" "<<ori<<" "<<dir<<" ";
      // COMP_LOOP(Dir,dir,{LOGGER<<siteCoords(dir);});
      // LOGGER<<"Neigh: ";
      // COMP_LOOP(Dir,dir,{LOGGER<<neighCoords(dir);});
      // LOGGER<<" "<<wrapping(dir)<<" or "<<neighCoords(dir)<<"=="<<triviallyShftedCoord<<" : "<<neighIsNotOnHalo;
      
      if(neighIsNotOnHalo)
	return
	  siteOfCoords(neighCoords);
      else
	return
	  vol()+haloSiteOfCoords(neighCoords,ori,dir);
    }
    
    /// Determines if a site is on halo
    bool siteIsOnHalo(const Site& site) const
    {
      return site>=vol();
    }
  };
};

#endif
