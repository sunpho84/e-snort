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
    
    /// Hashable properties of a \c Grill
    ///
    /// Forward implementation
    template <typename T,           // Derived class
	      typename Coord,       // Type of coordinate values
	      typename Site,        // Type of index of points
	      bool Hashing>         // Store or not tables
    struct HashableTableProvider;
    
    /// Hashable properties of a \c Grill
    ///
    /// Hashing version
    template <typename T,           // Derived type
	      typename Coord,       // Type of coordinate values
	      typename Site>        // Type of index of points
    struct HashableTableProvider<T,
				 Coord,
				 Site,
				 true>
    {
      /// Hashed coords of all points
      DynamicTens<OfComps<Site,Dir>,Coord,ExecSpace::HOST> coordsOfPointsHashTable;
      
      /// Hashed neighbors
      DynamicTens<OfComps<Site,Ori,Dir>,Site,ExecSpace::HOST> neighsOfPointsHashTable;
      
      /// Set the hash table of coordinates of all points
      void fillCoordsOfPointsHashTables()
      {
	DE_CRTPFY(T,this).forAllSites([&](const Site& site)
	{
	  coordsOfPointsHashTable(site)=
	    DE_CRTPFY(T,this).computeCoordsOfPoint(site);
	});
      }
      
      /// Set the hash table of neighbors
      void fillNeighsOfPointsHashTables()
      {
	loopOnAllComps<CompsList<Site,Ori,Dir>>(std::make_tuple(DE_CRTPFY(T,this).volume),
					   [&](const Site& site,
					       const Ori& ori,
					       const Dir& dir)
					   {
					     neighsOfPointsHashTable(site,ori,dir)=
					       DE_CRTPFY(T,this).computeNeighOfPoint(site,ori,dir);
					   });
      }
    };
    
    /// Hashable properties of a \c Grill
    ///
    /// Not-Hashing version
    template <typename T,           // Derived type
	      typename Coords,      // Type of coordinate values
	      typename Site>        // Type of index of points
    struct HashableTableProvider<T,
				 Coords,
				 Site,
				 false>
    {
    };
    
    /// Hashable properties of a \c Grill
    ///
    /// Forward implementation
    template <typename T,           // Derived class
	      typename Coords,      // Type of coordinate values
	      typename Site,        // Type of index of points
	      bool Hashing>         // Store or not tables
    struct HashableProvider :
      HashableTableProvider<T,Coords,Site,Hashing>
    {
      /// Fill all the HashTables
      void fillHashTables(const Site& size)
      {
	static_assert(Hashing,"Cannot initialize the hash table if not present");
	
	if constexpr(Hashing)
	  {
	    this->coordsOfPointsHashTable.allocate(std::make_tuple(size));
	    this->neighsOfPointsHashTable.allocate(std::make_tuple(size));
	    
	    this->fillCoordsOfPointsHashTables();
	    this->fillNeighsOfPointsHashTables();
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
	    this->coordsOfPointsHashTable(site);
	else
	  return DE_CRTPFY(T,this).computeCoordsOfPoint(site);
      }
      
      /// Return the neighbor in the given oriented dir
      constexpr INLINE_FUNCTION HOST_DEVICE_ATTRIB
      const Site& neighOfPoint(const Site& site,
			       const Ori& ori,
			       const Dir& dir)
	const
      {
	// CRTP_THIS.assertPointIsInRange(i);
	// CRTP_THIS.assertOriDirIsInRange(oriDir);
	
	if constexpr(Hashing)
	  return
	    this->neighsOfPointsHashTable(site,ori,dir);
	else
	  return
	    computeNeighOfPoint(site,ori,dir);
      }
    };
    
    /// A grill
    template <typename Coord,
	      typename Site,
	      bool Hashing>
    struct Grill :
      HashableProvider<Grill<Coord,Site,Hashing>,Coord,Site,Hashing>
    {
      /// Type needed to store all coords
      using Coords=
	StackTens<OfComps<Dir>,Coord>;
      
      /// Volume
      Site volume;
      
      /// Grid sides
      Coords sides;
      
      /// Create from sides
      Grill(const Coords& sides) :
	volume((sides(Dir(I))*...)),
	sides(sides)
      {
	if constexpr(Hashing)
	  this->fillHashTables(volume);
      }
      
      /// Loop on all points calling the passed function
      template <typename F>
      void forAllSites(F f) const
      {
	loopOnAllComps<CompsList<Site>>(std::make_tuple(volume),f);
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
	
	/// Destintion not considering wrap
	const Coord rawDest=
	  in(dir)+offset;
	
	const Coord& side=
	  sides(dir);
	
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
	    sides(mu);
	  
	  // Increment the coordinate
	  out=out*s+cs(mu);
	});
	
	return out;
      }
      
      /// Compute the coordinate of point i
      StackTens<OfComps<Dir>,Coord> computeCoordsOfPoint(Site site /*don't make const*/) const
      {
	// assertPointIsInRange(i);
	
	/// Result
	StackTens<OfComps<Dir>,Coord> c;
	
	for(Dir mu=NDims-1;mu>=0;mu--)
	  {
	    /// Dividend, corresponding to the \c mu side length
	    const Coord& d=sides(mu);
	    
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
