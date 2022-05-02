#ifndef _FIELD_HPP
#define _FIELD_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file grill/field.hpp

#include <mpi.h>

#include <expr/assign/executionSpace.hpp>
#include <expr/comps/comps.hpp>
#include <expr/nodes/tensRef.hpp>
#include <lattice/fieldCompsProvider.hpp>
#include <lattice/lattice.hpp>

namespace grill
{
  PROVIDE_DETECTABLE_AS(Field);
  
#define UNIVERSE Universe<NDims>
  
#define LATTICE Lattice<UNIVERSE>
  
#define FIELD_COMPS typename FieldCompsProvider<CompsList<C...>,_Fund,LATTICE,LC,FL>::Comps
  
#define THIS					\
  Field<CompsList<C...>,_Fund,LATTICE,LC,FL,ES,IsRef>
  
#define BASE					\
    Node<THIS>
  
  /// Defines a field
  template <typename...C,
	    typename _Fund,
	    int NDims,
	    LatticeCoverage LC,
	    FieldLayout FL,
	    ExecSpace ES,
	    bool IsRef>
  struct THIS :
    DynamicCompsProvider<FIELD_COMPS>,
    DetectableAsField,
    // SubNodes<_E...>,
    BASE
  {
    /// Import the base expression
    using Base=BASE;
    
    using This=THIS;
    
#undef BASE
    
#undef THIS
    
    /// Importing assignment operator from BaseTens
    using Base::operator=;
    
    /// Copy assign
    INLINE_FUNCTION
    Field& operator=(const Field& oth)
    {
      Base::operator=(oth);
      
      return *this;
    }
    
    /// Move assign
    INLINE_FUNCTION
    Field& operator=(Field&& oth)
    {
      std::swap(data,oth.data);
      
      return *this;
    }
    
    /// Referred lattice
    using L=LATTICE;
    
    /// Components
    using Comps=
      FIELD_COMPS;
    
    /// Import dynamic comps
    using DynamicComps=
      typename DynamicCompsProvider<FIELD_COMPS>::DynamicComps;
    
#undef FIELD_COMPS
    
#undef LATTICE
    
#undef UNIVERSE
    
    /// Fundamental tye
    using Fund=
      typename FieldCompsProvider<CompsList<C...>,_Fund,L,LC,FL>::Fund;
    
    /// Internal storage type
    using Data=
      std::conditional_t<IsRef,
      TensRef<Comps,Fund,ES>,
      DynamicTens<Comps,Fund,ES>>;
    
    /// Executes where allocated
    static constexpr ExecSpace execSpace=ES;
    
    /// Returns the size needed to allocate
    INLINE_FUNCTION HOST_DEVICE_ATTRIB
    decltype(auto) getInitVol(const bool flagVol) const
    {
      auto res=[flagVol](const auto& vol,const auto& halo)->std::decay_t<decltype(vol)>
      {
	if(flagVol)
	  return vol+halo;
	else
	  return vol;
      };
      
      if constexpr(FL==FieldLayout::SIMDIFIABLE or
		   FL==FieldLayout::SIMDIFIED)
	return res(lattice->simdLoc.eoVol,lattice->simdLoc.eoHalo);
      else
	return res(lattice->loc.eoVol,lattice->loc.eoHalo);
    }
    
    /// Returns the dynamic sizes
    INLINE_FUNCTION HOST_DEVICE_ATTRIB
    auto getDynamicSizes() const
    {
      return std::make_tuple(getInitVol(haloFlag));
    }
    
#define PROVIDE_EVAL(ATTRIB)					\
    template <typename...U>					\
    constexpr INLINE_FUNCTION					\
    ATTRIB Fund& eval(const U&...cs) ATTRIB			\
    {								\
      return data(cs...);					\
    }
    
    PROVIDE_EVAL(const);
    
    PROVIDE_EVAL(/* non const */);
    
#undef PROVIDE_EVAL
    
    /// Return whether can be assigned at compile time
    static constexpr bool canAssignAtCompileTime=
      not std::is_const_v<Fund>;
    
    /////////////////////////////////////////////////////////////////
    
#define PROVIDE_GET_REF(ATTRIB)						\
  auto getRef() ATTRIB							\
  {									\
    return Field<CompsList<C...>,ATTRIB _Fund,L,LC,FL,ES,true>		\
      (*lattice,haloFlag,data.storage,data.nElements,data.getDynamicSizes()); \
  }
  
  PROVIDE_GET_REF(const);
  
  PROVIDE_GET_REF(/* non const */);
  
  /////////////////////////////////////////////////////////////////
  
#undef PROVIDE_GET_REF
    
#define PROVIDE_SIMDIFY(ATTRIB)						\
    INLINE_FUNCTION							\
  auto simdify() ATTRIB							\
    {									\
      if constexpr(FL==FieldLayout::SIMDIFIABLE)			\
	return Field<CompsList<C...>,ATTRIB _Fund,L,LC,FieldLayout::SIMDIFIED,ES,true> \
	  (*lattice,haloFlag,(ATTRIB void*)data.storage,data.nElements,data.getDynamicSizes()); \
      else								\
	{								\
	  using Traits=CompsListSimdifiableTraits<CompsList<C...>,_Fund>; \
	  								\
	  using SimdFund=typename Traits::SimdFund;			\
	  								\
	  return Field<typename Traits::Comps,ATTRIB SimdFund,L,LC,FieldLayout::SERIAL,ES,true> \
	    (*lattice,haloFlag,(ATTRIB void*)data.storage,data.nElements,data.getDynamicSizes()); \
	}								\
    }
  
  PROVIDE_SIMDIFY(const);
  
  PROVIDE_SIMDIFY(/* non const */);
  
#undef PROVIDE_SIMDIFY
  
    /// States whether the field can be simdified
    static constexpr bool canSimdify=
      FL!=FieldLayout::SIMDIFIED and Data::canSimdify;
    
    /// Simdifying component
    using SimdifyingComp=
      typename Data::SimdifyingComp;
    
    /// We keep referring to the original object
    static constexpr bool storeByRef=not IsRef;
    
    /// Returns that can assign
    constexpr INLINE_FUNCTION
    bool canAssign()
    {
      return canAssignAtCompileTime;
    }
    
    /// Underneath lattice
    const L* lattice;
    
    /// Storage data
    Data data;
    
    /// Determine whether the halos are allocated
    const bool haloFlag;
    
    void updateHalo()
    {
      if(not haloFlag)
	CRASH<<"Trying to synchronize the halo but they are not allocated";
      
      LOGGER<<"Updating the halo";
      
      using LocEoSite=typename L::Loc::EoSite;
      using Parity=typename L::Parity;
      using Dir=typename L::U::Dir;
      
      MPI_Request requests[4*NDims];
      
      if constexpr(FL==FieldLayout::SERIAL)
	{
	  if constexpr(LC==LatticeCoverage::EVEN_ODD)
	    {
	      const LocEoSite& locEoHalo=lattice->loc.eoHalo;
	      
	      DynamicTens<OfComps<Parity,LocEoSite,C...>,Fund,ES> bufferOut(locEoHalo);
	      DynamicTens<OfComps<Parity,LocEoSite,C...>,Fund,ES> bufferIn(locEoHalo);
	      
	      int iRequest=0;
	      for(typename L::Parity parity=0;parity<2;parity++)
		{
		  loopOnAllComps<CompsList<LocEoSite>>(std::make_tuple(locEoHalo),
						       [&bufferOut,parity,this](const LocEoSite& siteRemappingId)
						       {
							 const auto& r=lattice->eoHaloSiteOfEoSurfSite(parity,siteRemappingId);
							 bufferOut(parity,r.second)=data(parity,r.first);
						       });
		  
		  for(Ori ori=0;ori<2;ori++)
		    for(Dir dir=0;dir<NDims;dir++)
		      if(lattice->nRanksPerDir(dir)>1)
			{
			  const LocEoSite& sendOffset=lattice->loc.eoHaloOffsets(parity,ori,dir);
			  const LocEoSite& recvOffset=lattice->loc.eoHaloOffsets(parity,ori,dir);
			  const LocEoSite& nSites=lattice->loc.eoHaloPerDir(ori,parity,dir);
			  
			  void* sendbuf=&bufferOut(parity,sendOffset,C(0)...);
			  int sendcount=nSites*(C::sizeAtCompileTime*...);
			  void* recvbuf=&bufferIn(parity,recvOffset,C(0)...);
			  int recvcount=sendcount;
			  int sendtag=index({},parity,ori,dir);
			  int recvtag=index({},parity,oppositeOri(ori),dir); ///Here we switch the orientation...
			  int dest=lattice->rankNeighbours(ori,dir)();
			  int source=dest;
			  
			  MPI_Isend(sendbuf,sendcount,MPI_CHAR,dest,sendtag,MPI_COMM_WORLD,&requests[iRequest++]);
			  MPI_Irecv(recvbuf,recvcount,MPI_CHAR,source,recvtag,MPI_COMM_WORLD,&requests[iRequest++]);
			}
		}
	      
	      MPI_Waitall(iRequest,requests,MPI_STATUS_IGNORE);
	      
	      for(typename L::Parity parity=0;parity<2;parity++)
		loopOnAllComps<CompsList<LocEoSite>>(std::make_tuple(locEoHalo),
					  [this,parity,&bufferIn](const LocEoSite& haloSite)
					  {
					    const LocEoSite dest=haloSite+lattice->loc.eoVol;
					    data(parity,dest)=bufferIn(parity,haloSite);
					  });
	    }
	}
    }
    
    /// Create a field
    Field(const L& lattice,
	  const bool& haloFlag=false) :
      lattice(&lattice),
      haloFlag(haloFlag)
    {
      static_assert(not IsRef,"Can allocate only if not a reference");
      
      data.allocate(std::make_tuple(getInitVol(haloFlag)));
    }
    
    /// Create a refence to a field
    Field(const L& lattice,
	  const bool& haloFlag,
	  void* storage,
	  const int64_t& nElements,
	  const DynamicComps& dynamicSizes) :
      lattice(&lattice),
      data((Fund*)storage,nElements,dynamicSizes),
      haloFlag(haloFlag)
    {
      static_assert(IsRef,"Can initialize as reference only if declared as a reference");
    }
    
    /// Copy constructor
    INLINE_FUNCTION HOST_DEVICE_ATTRIB
    Field(const Field& oth) :
      lattice(oth.lattice),
      data(oth.data),
      haloFlag(oth.haloFlag)
    {
    }
  };
}

#endif
