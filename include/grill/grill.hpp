#ifndef _GRILL_HPP
#define _GRILL_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file grill/grill.hpp

#include <expr/comps/comp.hpp>
#include <expr/nodes/stackTens.hpp>
#include <resources/mathOperations.hpp>

namespace esnort
{
  DECLARE_UNTRANSPOSABLE_COMP(Ori,int,2,ori);
  
  /// Offset to move \c BW or \c FW
  constexpr int moveOffset[2]=
    {-1,+1};
  
  template <int NDims,
	    typename I=std::make_integer_sequence<int,NDims>>
  struct Universe;
  
  template <int NDims,
	    int...I>
  struct Universe<NDims,std::integer_sequence<int,I...>>
  {
    DECLARE_TRANSPOSABLE_COMP(Dir,int,NDims,dir);
    
    /// Tensor with Dir component and F type
    template <typename F>
    using DirTens=StackTens<OfComps<Dir>,F>;
    
    DECLARE_TRANSPOSABLE_COMP(Surf,int,NDims,dir);
    
    /// Hashable properties of a \c Grill
    ///
    /// Forward implementation
    template <typename Coord,       // Type of coordinate values
	      typename Site,        // Type of index of points
	      bool Hashing>         // Store or not tables
    struct HashableTableProvider;
    
    /// Hashable properties of a \c Grill
    ///
    /// Hashing version
    template <typename Coord,       // Type of coordinate values
	      typename Site>        // Type of index of points
    struct HashableTableProvider<Coord,
				 Site,
				 true>
    {
      /// Hashed coords of all points
      DynamicTens<OfComps<Site,Dir>,Coord,ExecSpace::HOST> coordsOfPointsHashTable;
      
      /// Hashed neighbors
      DynamicTens<OfComps<Site,Ori,Dir>,Site,ExecSpace::HOST> neighsOfPointsHashTable;
    };
    
    /// Hashable properties of a \c Grill
    ///
    /// Not-Hashing version
    template <typename Coords,      // Type of coordinate values
	      typename Site>        // Type of index of points
    struct HashableTableProvider<Coords,
				 Site,
				 false>
    {
    };
    
    /// A grill
    template <typename Coord,
	      typename Site,
	      bool Hashing>
    struct Grill :
      HashableTableProvider<Coord,Site,Hashing>
    {
      /// Type needed to store all coords
      using Coords=
	DirTens<Coord>;
      
      /////////////////////////////////////////////////////////////////
      
      /// Volume
      Site _vol;
      
      /// Volume, const access
      INLINE_FUNCTION HOST_DEVICE_ATTRIB
      const Site& vol() const
      {
	return _vol;
      }
      
      /////////////////////////////////////////////////////////////////
      
      /// Sides
      Coords _sides;
      
      /// Sides, constant access
      INLINE_FUNCTION HOST_DEVICE_ATTRIB
      const Coords& sides() const
      {
	return _sides;
      }
      
      /// Side in a given direction, constant access
      INLINE_FUNCTION HOST_DEVICE_ATTRIB
      const Coord& side(const Dir& dir) const
      {
	return _sides(dir);
      }
      
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
	Coord side=side(dir);
	
	if(wrapping(dir))
	  return side;
	else
	  if(side<=2)
	    return 0;
	  else
	    side-2;
      }
      
      /////////////////////////////////////////////////////////////////
      
      /// Halo sizes
      DirTens<Site> _haloSizes;
      
      /// Halo size in a given direction, constant access
      INLINE_FUNCTION HOST_DEVICE_ATTRIB
      const Site& haloSize(const Dir& dir) const
      {
	return _haloSizes(dir);
      }
      
      /// Total halo volume
      Site _totHaloVol;
      
      /// Total halo volume, constant access
      INLINE_FUNCTION HOST_DEVICE_ATTRIB
      const Site& totHaloVol() const
      {
	return _totHaloVol;
      }
      
      /// Initialize border sizes
      void initBordSizes()
      {
	_haloSizes=_surfSizes-_wrapping*_surfSizes;
	
	_totHaloVol=(_haloSizes(Dir(I))+...);
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
	_bulkVol=(((Site)bulkSide(I))*...);
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
      
      /// Set the sides and derived quantities
      void setSidesAndWrapping(const Coords& sides,
			       const DirTens<bool>& wrapping,
			       const bool& fillHashTables=true)
      {
	_vol=(sides(Dir(I))*...);
	_sides=sides;
	_wrapping=wrapping;
	compLoop<Dir>([this,&sides](const Dir& dir) INLINE_ATTRIBUTE
	{
	  _surfSizes(dir)=_vol/sides(dir);
	});
	
	initBordSizes();
	initBulkVol();
	initSurfVol();
	
	if constexpr(Hashing)
	  if(fillHashTables)
	    this->fillHashTables(vol());
      }
      
      /////////////////////////////////////////////////////////////////
      
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
	    this->coordsOfPointsHashTable.allocate(std::make_tuple(size));
	    this->neighsOfPointsHashTable.allocate(std::make_tuple(size));
	    
	    /// Set the hash table of coordinates of all points
	    forAllSites([&](const Site& site)
	    {
	      this->coordsOfPointsHashTable(site)=
		computeCoordsOfPoint(site);
	    });
	    
	    loopOnAllComps<CompsList<Site,Ori,Dir>>(std::make_tuple(vol()),
						    [&](const Site& site,
							const Ori& ori,
							const Dir& dir)
						    {
						      this->neighsOfPointsHashTable(site,ori,dir)=
							computeNeighOfPoint(site,ori,dir);
						    });
	  }
      }
      
      /////////////////////////////////////////////////////////////////
      
      /// Get the coords of given point
      constexpr INLINE_FUNCTION HOST_DEVICE_ATTRIB
      decltype(auto) coordsOfPoint(const Site& site) const
      {
	// DE_CRTPFY(T,this).assertPointIsInRange(site);
	
	if constexpr(Hashing)
	  return
	    (this->coordsOfPointsHashTable(site));
	else
	  return
	    computeCoordsOfPoint(site);
      }
      
      /// Return the neighbor in the given oriented dir
      constexpr INLINE_FUNCTION HOST_DEVICE_ATTRIB
      decltype(auto) neighOfPoint(const Site& site,
				  const Ori& ori,
				  const Dir& dir)
	const
      {
	// CRTP_THIS.assertPointIsInRange(i);
	// CRTP_THIS.assertOriDirIsInRange(oriDir);
	
	if constexpr(Hashing)
	  return
	    (this->neighsOfPointsHashTable(site,ori,dir));
	else
	  return
	    computeNeighOfPoint(site,ori,dir);
      }
      
      /// Loop on all points calling the passed function
      template <typename F>
      void forAllSites(F f) const
      {
	loopOnAllComps<CompsList<Site>>(std::make_tuple(vol()),f);
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
	
	compLoop<Dir>([&](const Dir& mu)
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
      
      /// Compute the point of given coords
      Site pointOfCoords(const Coords& cs) const ///< Coordinates of the point
      {
	// assertCoordsAreInRange(cs);
	
	/// Returned point
	Site out=0;
	
	compLoop<Dir>([&](const Dir& mu)
	{
	  /// Grill side
	  const Coord& s=
	    sides()(mu);
	  
	  // Increment the coordinate
	  out=out*s+cs(mu);
	});
	
	return out;
      }
      
      /// Compute the coordinate of point i
      DirTens<Coord> computeCoordsOfPoint(Site site /*don't make const*/) const
      {
	// assertPointIsInRange(i);
	
	/// Result
	DirTens<Coord> c;
	
	for(Dir mu=NDims-1;mu>=0;mu--)
	  {
	    /// Dividend, corresponding to the \c mu side length
	    const Coord& d=sides()(mu);
	    
	    /// Quotient, corresponding to the index of the remaining \c nDims-1 components
	    const Site q=site/d;
	    
	    /// Remainder, corresponding to the coordinate
	    const Coord r=(typename Coord::Index)(site-d*q);
	    
	    // Store the component
	    c(mu)=r;
	    
	    // Iterate on the remaining nDims-1 components
	    site=q;
	  }
	
	return c;
      }
      
      /// Compute the neighbor in the oriented direction oriDir of point i
      Site computeNeighOfPoint(const Site& site,      ///< Point
			       const Ori& ori,     ///< Orientation
			       const Dir& dir)     ///< Direction
	const
      {
	// assertPointIsInRange(i);
	// assertOriDirIsInRange(oriDir);
	
	return
	  pointOfCoords(shiftedCoords(this->coordsOfPoint(site),ori,dir));
      }
    };
  };
};

#endif
