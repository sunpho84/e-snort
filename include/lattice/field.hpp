#ifndef _FIELD_HPP
#define _FIELD_HPP

#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file grill/field.hpp

#include <expr/assign/executionSpace.hpp>
#include <expr/comps/comps.hpp>
#include <expr/nodes/tensRef.hpp>
#include <lattice/fieldCompsProvider.hpp>
#include <lattice/fieldParity.hpp>
#include <lattice/lattice.hpp>
#include <resources/Mpi.hpp>

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
    FieldParity<LATTICE,LC>,
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
    
    /// Updates the halo when the layout is simdifiable
    void updateSimdifiableHalo()
    {
      LOGGER<<" updateSimdifiableHalo";
      
      using SimdLocEoSite=typename L::SimdLoc::EoSite;
      using SimdRank=typename L::SimdRank;
      using Parity=typename L::Parity;
      
      const SimdLocEoSite& simdLocEoHalo=lattice->simdLoc.eoHalo;
      
      for(Ori ori=0;ori<2;ori++)
	for(typename L::Dir dir=0;dir<NDims;dir++)
	  {
	    LOGGER<<"Ori "<<ori<<" dir "<<dir;
	    
	    for(SimdRank simdRank=0;simdRank<lattice->nSimdRanks;simdRank++)
	      LOGGER<<"simdRank "<<lattice->simdRankNeighbours(simdRank,ori,dir)<<" will be copied to "<<simdRank;
	  }
      
      auto fillBufferParity=
	[this,simdLocEoHalo](auto&& data,const Parity& parity)
	{
	  // for(Ori ori=0;ori<2;ori++)
	    // for(typename L::Dir dir=0;dir<NDims;dir++)
	      {
		// const SimdLocEoSite& eoHaloPerDir=lattice->simdLoc.eoHaloPerDir(ori,parity,dir);
		// const SimdLocEoSite& eoHaloOffset=lattice->simdLoc.eoHaloOffsets(ori,parity,dir);
		
		// LOGGER<<"Ori "<<ori<<" dir "<<dir<<" "<<eoHaloPerDir;
		
		// if(eoHaloPerDir)
		  loopOnAllComps<CompsList<SimdLocEoSite>>(std::make_tuple(lattice->simdLoc.eoHalo),
							   [data=data.getRef(),// eoHaloOffset,
							    parity,this//,&ori// ,&dir
							    ](const SimdLocEoSite& eoHaloSite) MUTABLE_INLINE_ATTRIBUTE
							   {
							     // const SimdLocEoSite siteRemappingId=eoHaloSiteInOriDir+eoHaloOffset;
							     const auto& r=lattice->simdEoHaloFillerTable(parity,eoHaloSite// siteRemappingId
													  );
							     const auto source=std::get<0>(r);
							     const SimdLocEoSite dest=std::get<1>(r)+lattice->simdLoc.eoVol;
							     const Ori ori=std::get<2>(r);
							     const typename L::Dir dir=std::get<3>(r);
							     LOGGER<<"Filling halo, id "<<eoHaloSite<<" parity "<<parity<<" ori "<<ori<<" dir "<<dir<<" dest "<<std::get<0>(r)<<" lattice->simdLoc.eoVol "<<lattice->simdLoc.eoVol<<"full dest "<<dest<<" with source "<<source;
							     
							     for(SimdRank simdRank=0;simdRank<lattice->nSimdRanks;simdRank++)
							       LOGGER<<"simdRank "<<lattice->simdRankNeighbours(simdRank,ori,dir)<<" will be copied to "<<simdRank;
							     
							     loopOnAllComps<CompsList<C...>>(this->getDynamicSizes(),[this,&data,&dest,&source,&r,&ori,&dir](const auto&...cs)
							     {
							       for(SimdRank simdRank=0;simdRank<lattice->nSimdRanks;simdRank++)
								 {
								   const SimdRank& sourceSimdRank=simdRank;
								   const SimdRank destSimdRank=lattice->simdRankNeighbours(simdRank,ori,dir);
								   data(dest,cs...,destSimdRank)=data(source,cs...,sourceSimdRank);
								   LOGGER<<" Copying "<<sourceSimdRank<<" simd rank into "<<destSimdRank;
								 }
							     });
						   });
	      }
	};
      
      if constexpr(LC==LatticeCoverage::EVEN_ODD)
	for(typename L::Parity parity=0;parity<2;parity++)
	  fillBufferParity(data(parity),parity);
      else
	fillBufferParity(data,this->parity);
    }
    
    /// Updates the halo when the layout is serial or gpu
    void updateNonSimdifiableHalo()
    {
      MPI_Request requests[2*4*NDims];
      
      using LocEoSite=typename L::Loc::EoSite;
      using Parity=typename L::Parity;
      using Dir=typename L::U::Dir;
      
      const LocEoSite& locEoHalo=lattice->loc.eoHalo;
      
      auto fillBufferParity=
	[this,locEoHalo](auto&& out,const auto& in,const Parity& parity)
	{
	  loopOnAllComps<CompsList<LocEoSite>>(std::make_tuple(locEoHalo),
					       [out=out.getRef(),in=in.getRef(),parity,this](const LocEoSite& siteRemappingId) MUTABLE_INLINE_ATTRIBUTE
					       {
						 const auto& r=lattice->eoHaloFillerTable(parity,siteRemappingId);
						 // const auto source=r.first;
						 // const auto dest=r.second;
						 // const _Fund& p=bufferOut(parity,dest,C(0)...);
						 // LOGGER<<"Filling buffer, parity "<<parity<<" dest "<<dest<<" with source "<<source<<" before: "<<p;
						 out(std::get<1>(r))=in(std::get<0>(r));
						 // LOGGER<<" "<<p;
					       });
	};
      
      auto startCommunicationsParity=
	[this](auto&& in,auto&& out,MPI_Request* requests,int& iRequest,const Parity& parity)
	{
	  for(Ori ori=0;ori<2;ori++)
	    for(Dir dir=0;dir<NDims;dir++)
	      if(lattice->nRanksPerDir(dir)>1)
		{
		  const LocEoSite& sendOffset=lattice->loc.eoHaloOffsets(parity,ori,dir);
		  const LocEoSite& recvOffset=lattice->loc.eoHaloOffsets(parity,ori,dir);
		  const LocEoSite& nSites=lattice->loc.eoHaloPerDir(ori,parity,dir);
		  
		  void* sendbuf=&out(sendOffset,C(0)...);
		  int sendcount=nSites*(C::sizeAtCompileTime*...)*sizeof(_Fund);
		  void* recvbuf=&in(recvOffset,C(0)...);
		  int recvcount=sendcount;
		  int sendtag=index({},parity,ori,dir);
		  int recvtag=sendtag;//index({},parity,oppositeOri(ori),dir); ///We don't have to switch the orientation as the offset is already computed w.r.t dest
		  int dest=lattice->rankNeighbours(ori,dir)();
		  int source=dest;
		  
		  // LOGGER<<"Rank "<<Mpi::rank<<" parity "<<parity<<" ori "<<ori<<" dir "<<dir<<" sending to rank "<<dest<<" with tag "<<sendtag<<" receiving from "<<source<<" with tag "<<recvtag<<" sending data "<<((_Fund*)sendbuf-bufferOut.storage)<<" receiving into "<<((_Fund*)recvbuf-bufferIn.storage)<<" nbytes: "<<sendcount;
		  
		  MPI_Isend(sendbuf,sendcount,MPI_CHAR,dest,sendtag,MPI_COMM_WORLD,&requests[iRequest++]);
		  MPI_Irecv(recvbuf,recvcount,MPI_CHAR,source,recvtag,MPI_COMM_WORLD,&requests[iRequest++]);
		}
	    };
      
      auto fillHaloParity=
	[this,locEoHalo](auto&& out,const auto& in,const Parity& parity)
	{
	  loopOnAllComps<CompsList<LocEoSite>>(std::make_tuple(locEoHalo),
					       [this,out=out.getRef(),in=in.getRef(),parity](const LocEoSite& haloSite) MUTABLE_INLINE_ATTRIBUTE
					       {
						 const LocEoSite dest=haloSite+lattice->loc.eoVol;
						 // const _Fund& p=data(parity,dest,C(0)...);
						 // LOGGER<<"Filling halo, parity "<<parity<<" site "<<dest<<" before: "<<p;
						 out(dest)=in(haloSite);
						 // LOGGER<<" "<<p;
						 
						 // LOGGER<<"Retry";
						 // auto lhs=data(parity,dest).simdify();
						 //auto rhs=scalar(2);
						 
						 // const auto& q=lhs(grill::NonSimdifiedComp<int, 1>(0));
						 // LOGGER<<" once simdify points to "<<&q;
						 // lhs=2;
						 // LOGGER<<" After reassigning: "<<p;
					       });
	};
      
      if constexpr(LC==LatticeCoverage::EVEN_ODD)
	{
	  DynamicTens<OfComps<Parity,LocEoSite,C...>,Fund,ES> bufferOut(locEoHalo);
	  DynamicTens<OfComps<Parity,LocEoSite,C...>,Fund,ES> bufferIn(locEoHalo);
	  
	  // bufferIn=-7;
	  // bufferOut=-6;
	  
	  // ALLOW_ALL_RANKS_TO_PRINT_FOR_THIS_SCOPE;
	  
	  int iRequest=0;
	  for(typename L::Parity parity=0;parity<2;parity++)
	    {
	      fillBufferParity(bufferOut(parity),data(parity),parity);
	      
	      startCommunicationsParity(bufferIn(parity),bufferOut(parity),requests,iRequest,parity);
	    }
	  
	  MPI_Waitall(iRequest,requests,MPI_STATUS_IGNORE);
	  
	  for(typename L::Parity parity=0;parity<2;parity++)
	    fillHaloParity(data(parity),bufferIn(parity),parity);
	}
      else
	{
	  DynamicTens<OfComps<LocEoSite,C...>,Fund,ES> bufferOut(locEoHalo);
	  DynamicTens<OfComps<LocEoSite,C...>,Fund,ES> bufferIn(locEoHalo);
	  
	  // bufferIn=-7;
	  // bufferOut=-6;
	  
	  // ALLOW_ALL_RANKS_TO_PRINT_FOR_THIS_SCOPE;
	  
	  int iRequest=0;
	  fillBufferParity(bufferOut,data,this->parity);
	  
	  startCommunicationsParity(bufferIn,bufferOut,requests,iRequest,this->parity);
	  
	  MPI_Waitall(iRequest,requests,MPI_STATUS_IGNORE);
	  
	  fillHaloParity(data,bufferIn,this->parity);
	}
    }
    
    /// Updates the halo
    void updateHalo()
    {
      if(not haloFlag)
	CRASH<<"Trying to synchronize the halo but they it is not allocated";
      
      // LOGGER<<"Updating the halo. Output buffer is pre-set to -6, Input buffer is preset to -7. Halo is preset to -8";
      
      static_assert(FL!=FieldLayout::SIMDIFIED,"Cannot communicate the halo of simdified layout");
      
      if constexpr(FL==FieldLayout::SIMDIFIABLE)
	updateSimdifiableHalo();
      else
	updateNonSimdifiableHalo();
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
