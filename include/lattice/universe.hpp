#ifndef _UNIVERSE_HPP
#define _UNIVERSE_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file lattice/universe.hpp

#include <expr/comps/comps.hpp>
#include <expr/nodes/stackTens.hpp>

namespace grill
{
  /// Assert that a quantity is in the given range
  template <typename T>
  constexpr INLINE_FUNCTION HOST_DEVICE_ATTRIB
  void assertIsInRange(const char* name,const T& val,const T& max)
  {
#ifdef ENABLE_GRILL_DEBUG
      if(val<0 or val>=max)
	CRASH<<name<<" value "<<val<<" not valid, maximal value: "<<max;
#endif
  }
  
  DECLARE_UNTRANSPOSABLE_COMP(Ori,int,2,ori);
  
#define BW Ori(0)
  
#define FW Ori(1)
  
  /// Opposite orientation
  constexpr INLINE_FUNCTION HOST_DEVICE_ATTRIB
  Ori oppositeOri(const Ori& ori)
  {
    return Ori(1)-ori;
  }
  
  /// Assert that ori is an orientation
  constexpr INLINE_FUNCTION HOST_DEVICE_ATTRIB
  void assertIsOri(const Ori& ori)
  {
    assertIsInRange("ori",ori,Ori(2));
  }
  
  /// Offset to move \c BW or \c FW
  constexpr int moveOffset[2]=
    {-1,+1};
  
  template <int NDims>
  struct Universe
  {
    DECLARE_TRANSPOSABLE_COMP(Dir,int,NDims,dir);
    
    /// Assert that dir is a direction
    constexpr INLINE_FUNCTION HOST_DEVICE_ATTRIB
    static void assertIsDir(const Dir& dir)
    {
      assertIsInRange("dir",dir,Dir(NDims));
    }
    
    /// Tensor with Dir component and F type
    template <typename F>
    using DirTens=StackTens<OfComps<Dir>,F>;
    
    DECLARE_TRANSPOSABLE_COMP(Surf,int,NDims,dir);
    
    /// Returns a vector containing 1 in a certain direction
    template <typename F>
    static constexpr INLINE_FUNCTION
    DirTens<F> versor(const Dir& dir)
    {
      DirTens<F> res=F(0);
      
      res(dir)=1;
      
      return res;
    }
    
    /// Compute the coordinate of site i
    template <typename Site,
	      typename Coord>
    INLINE_FUNCTION constexpr HOST_DEVICE_ATTRIB
    static DirTens<Coord> computeCoordsOfSiteInBoxOfSides(Site site /*don't make const*/,const DirTens<Coord>& sides)
      {
	/// Result
	DirTens<Coord> c;
	
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
    
    /// Compute the site of given coords in the given ori,dir halo
    template <typename Site,
	      typename Coord>
    INLINE_FUNCTION constexpr HOST_DEVICE_ATTRIB
    static Site computeSiteOfCoordsInBoxOfSides(const DirTens<Coord>& coords,const DirTens<Coord>& sides)
    {
      /// Returned site
      Site out=0;
      
      COMP_LOOP(Dir,mu,
		{
		    out=out*sides(mu)+coords(mu);
		});
      
      return out;
    }
  };
}

#endif
