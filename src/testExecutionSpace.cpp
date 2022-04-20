#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

#include <algorithm>

#include <cstdarg>
#include <cstdio>
#include <unistd.h>
#include <iostream>

#include <grill.hpp>

using namespace grill;

/////////////////////////////////////////////////////////////////

DECLARE_TRANSPOSABLE_COMP(Spin,int,4,spin);
DECLARE_TRANSPOSABLE_COMP(Span,int,4,span);
DECLARE_UNTRANSPOSABLE_COMP(SpaceTime,int64_t,0,spaceTime);

int j;

// template <typename E>
// struct SizeIsKnownAtCompileTime;

// struct S :
//   IndexComputer<S>
// {
  
// };

// struct InvalidExpr :
//   public Node<InvalidExpr>
// {
// };

// InvalidExpr e;

using Comps=CompsList<ComplId,SpinRow,SpinCln>;
StackTens<Comps,double> a;
StackTens<CompsList<ComplId,SpinRow,SpinCln>,double> b;

StackTens<CompsList<ComplId>,double> one;

void testDag()
{
  
  loopOnAllComps<Comps>({},[](const ComplId& reIm,const SpinRow& spinRow,const SpinCln& spinCln)
  {
    a(reIm,spinRow,spinCln)=spinCln+4*(spinRow+4*reIm);
    b(reIm,spinRow,spinCln)=0.0;
  });
  
  // static constexpr auto aa=std::bool_constant<isTransposableComp<SpinCln>>{};
  // auto ccc=Transp<SpinCln>{};
  // auto tb=transp(b);
  // auto tbc=decltype(tb)::SimdifyingComp{};
  // auto r=tb.simdify();
  // REORDER_BARRIER();
  // ASM_BOOKMARK_BEGIN("bEqDagA");
  // b=dag(a);
  // ASM_BOOKMARK_END("bEqDagA");
  // REORDER_BARRIER();
  
  // loopOnAllComps<Comps>({},[](const ComplId& reIm,const SpinRow& spinRow,const SpinCln& spinCln)
  // {
  //   //LOGGER<<b(reIm,spinRow,spinCln)<<" "<<(reIm?-1:+1)*(reIm+2*(spinCln+4*spinRow));
  //   LOGGER<<b(reIm,spinRow,spinCln)<<" "<<(reIm?-1:+1)*(spinRow+4*(spinCln+4*reIm));
  // });
  
  // auto c=(a*b).fillDynamicTens();
  
  // loopOnAllComps<Comps>({},[&c](const ComplId& reIm,const SpinRow& spinRow,const SpinCln& spinCln)
  // {
  //   //LOGGER<<b(reIm,spinRow,spinCln)<<" "<<(reIm?-1:+1)*(reIm+2*(spinCln+4*spinRow));
  //   LOGGER<<c(reIm,spinRow,spinCln);
  // });
  
  {
    real(one)=0.0;
    imag(one)=1.0;

    // LOGGER<<"/////////////////////////////////////////////////////////////////";
    
    // using A=decltype(conj(a));
    // LOGGER<<"conj(a) will return: "<<demangle(typeid(A).name())<<(std::is_lvalue_reference_v<A>?"&":(std::is_rvalue_reference_v<A>?"&&":""));
    
    // LOGGER<<"/////////////////////////////////////////////////////////////////";
    
    // auto aa=dag(a);
    // using B=decltype(aa);
    // LOGGER<<"aa is: "<<demangle(typeid(B).name())<<(std::is_lvalue_reference_v<B>?"&":(std::is_rvalue_reference_v<B>?"&&":""));
    // auto b=aa.fillDynamicTens();
    
    // auto t=dag(a);
    // LOGGER<<a.storage<<" "<<t.conjExpr.storage;
    
    auto b=(one*dag(one)).fillDynamicTens();
    
    
    ASM_BOOKMARK_BEGIN("sum");
    auto c=(one+dag(one)).fillDynamicTens();
    ASM_BOOKMARK_END("sum");
    
    ASM_BOOKMARK_BEGIN("sumTheProd");
    auto d=(one+dag(one)*one).fillDynamicTens();
    ASM_BOOKMARK_END("sumTheProd");
    
    //LOGGER<<b.factNode<0>().storage<<" "<<b.factNode<1>().conjExpr.storage<<" "<<a.storage;
    
    // .fillDynamicTens();
    
    LOGGER<<real(b)<<" "<<imag(b);
    LOGGER<<real(c)<<" "<<imag(c);
    LOGGER<<real(d)<<" "<<imag(d);
  }
}

void testProd()
{
  using Comps=CompsList<SpanRow,SpinRow>;
  using Comps2=CompsList<SpinRow,SpanRow>;
  
  StackTens<Comps,double> a,c;
  StackTens<Comps2,double> b;
  
  loopOnAllComps<Comps>({},[&a,&b](const SpanRow& spanRow,const SpinRow& spinRow)
  {
    a(spanRow,spinRow)=spinRow+4*spanRow;
    b(spanRow,spinRow)=-spinRow+4*spanRow;
  });

  ASM_BOOKMARK_BEGIN("prod");
  c=a*b;
  ASM_BOOKMARK_END("prod");
  
  loopOnAllComps<Comps>({},[&c](const SpanRow& spanRow,const SpinRow& spinRow)
  {
    LOGGER<<" "<<c(spanRow,spinRow)<<" "<<(spinRow+4*spanRow)*(-spinRow+4*spanRow);
  });
}

namespace Tests
{
  DECLARE_UNTRANSPOSABLE_COMP(Sim,int,4,sim);
  
  using Comps=CompsList<SpinRow,ComplId,Sim>;
  
  StackTens<Comps,double> a,b,c;
  
  void testS()
  {
    loopOnAllComps<Comps>({},[](const SpinRow& sr,const ComplId& reIm,const Sim& sim)
    {
      b(sr,reIm,sim)=c(sr,reIm,sim)=(reIm==0);
    });
    
    ASM_BOOKMARK_BEGIN("TESTS");
    a=b*c;
    ASM_BOOKMARK_END("TESTS");
    
    loopOnAllComps<Comps>({},[](const SpinRow& sr,const ComplId& reIm,const Sim& sim)
    {
      LOGGER<<a(sr,reIm,sim);
    });
  }
}

DECLARE_UNTRANSPOSABLE_COMP(GlbCoord,int,0,glbCoord);
DECLARE_UNTRANSPOSABLE_COMP(GlbSite,int64_t,0,glbSite);

void testGrill()
{
  using U=Universe<4>;
  using Dir=U::Dir;
  
  using Lattice=U::Lattice;
  //using GlbGrill=U::GlbGrill;
  
  using Parity=Lattice::Parity;
  using SimdLocEoSite=Lattice::SimdLocEoSite;
  using SimdRank=Lattice::SimdRank;
  
  using GlbCoords=Lattice::GlbCoords;
  // using LocCoords=Lattice::LocCoords;
  using RankCoords=Lattice::RankCoords;
  using SimdRankCoords=Lattice::SimdRankCoords;
  
  GlbCoords glbSides(6,6,6,12);
  RankCoords rankSides(2,1,1,1);
  SimdRankCoords simdRankSides(1,1,2,4);
  
  U::Lattice lattice(glbSides,rankSides,simdRankSides,1);
  
  [[maybe_unused]]
  auto printCoords=[](auto& l,const auto& c)
  {
    l<<"("<<c(Dir(0))<<","<<c(Dir(1))<<","<<c(Dir(2))<<","<<c(Dir(3))<<")";
  };
  
  static_assert(not std::is_same_v<U::Lattice::GlbCoord,U::Lattice::LocCoord>,"");
  
  LOGGER<<"Now doing the computeGlbCoordsOfsimdEoRepOfLocSite test";
  loopOnAllComps<CompsList<Parity,SimdLocEoSite,SimdRank>>(std::make_tuple(lattice.simdLocEoVol),
							   [&printCoords,
							    &lattice](const Parity& parity,
								       const SimdLocEoSite& simdLocEoSite,
								       const SimdRank& simdRank)
  {
    const GlbCoords glbCoords=
      lattice.computeGlbCoordsOfSimdEoRepOfLocSite(parity,simdLocEoSite,simdRank);
    
    const auto [rankP,parityP,simdLocEoSiteP,simdRankP]=lattice.computeSimdEoRepOfLocSiteOfGlbCoords(glbCoords);
    
    auto l=LOGGER;
    l<<"parity "<<parity<<" simdLocEoSite "<<simdLocEoSite<<" simdRank "<<simdRank<<" glbCoords: ";
    printCoords(l,glbCoords);
    l<<" rank "<<rankP<<" parity "<<parityP<<" simdLocEoSite "<<simdLocEoSiteP<<" simdRank "<<simdRankP<<" "<<(0!=rankP or parity!=parityP or simdLocEoSite!=simdLocEoSiteP or simdRank!=simdRankP);
  });
  LOGGER<<"Finished";
  
  int nNonLoc=0,nLoc=0;
  LOGGER<<"Now doing the neigh test";
  loopOnAllComps<CompsList<Parity,SimdLocEoSite>>(std::make_tuple(lattice.simdLocEoVol),
						  [&nLoc,&nNonLoc,
						   &printCoords,
						   &lattice](const Parity& parity,
							      const SimdLocEoSite& simdLocEoSite)
						  {
    for(Ori ori=0;ori<2;ori++)
      {
	for(Dir dir=0;dir<4;dir++)
	  {
	    bool isLoc=true;
	    
 	    for(SimdRank simdRank=0;simdRank<SimdRank::sizeAtCompileTime;simdRank++)
	      {
		const GlbCoords glbCoords=
		  lattice.computeGlbCoordsOfSimdEoRepOfLocSite(parity,simdLocEoSite,simdRank);
		
		GlbCoords neighCoords=lattice.shiftedCoords(glbCoords,ori,dir);
		
		const auto [rankP,parityP,simdLocEoSiteP,simdRankP]=lattice.computeSimdEoRepOfLocSiteOfGlbCoords(neighCoords);
		
		auto l=LOGGER;
		l<<"   ";
		l<<"parity "<<parity<<" simdLocEoSite "<<simdLocEoSite<<" simdRank "<<simdRank<<" glbCoords: ";
		printCoords(l,glbCoords);
		l<<" ori "<<ori<<" dir "<<dir<<" neighCoords: ";
		printCoords(l,neighCoords);
		
		l<<" rank "<<rankP<<" parity "<<parityP<<" simdLocEoSite "<<simdLocEoSiteP<<" simdRank "<<simdRankP;
		if(Mpi::rank!=rankP)
		  {
		    isLoc=false;
		    l<<" (rank) ";
		  }
		if(simdRank!=simdRankP)
		  {
		    isLoc=false;
		    l<<" (simdRank) ";
		  }
	      }
	    LOGGER<<"isLoc: "<<isLoc;
	    LOGGER<<"";
	    
	    if(not isLoc)
	      nNonLoc++;
	    else
	      nLoc++;
	  }
      }
  });
  
  LOGGER<<"Summary, loc: "<<nLoc<<" nonLoc: "<<nNonLoc;
  
  // GlbGrill glbGrill(sides,wrapping);
  
  // U::DirTens<bool> wrapping(1,1,0,1);
  // {
  //   auto l=LOGGER;
  //   l<<"wrapping: ";
  //   printCoords(l,glbGrill._wrapping);
  // }
  
  // glbGrill.forAllSites([&glbGrill,&printCoords](const GlbSite& site)
  // {
  //   loopOnAllComps<CompsList<Ori,Dir>>({},[printCoords,&glbGrill](const GlbSite& site,const Ori& ori,const Dir& dir)
  //   {
  //     const auto coords=glbGrill.coordsOfSite(site);
  //     const GlbSite neigh=glbGrill.neighOfSite(site,ori,dir);
      
  //     auto l=LOGGER;
  //     l<<"Site "<<site;
  //     printCoords(l,coords);
  //     l<<" ori "<<ori<<" dir "<<dir<<" neigh "<<neigh<<" ";
  //     if(glbGrill.siteIsOnHalo(neigh))
  // 	{
  // 	  l<<"on halo, wraps to: ";
  // 	  const GlbSite wrapSite=glbGrill.surfSiteOfHaloSite(neigh-glbGrill.vol());
  // 	  l<<wrapSite<<" ";
  // 	  const GlbCoords wrapCoords=glbGrill.coordsOfSite(wrapSite);
  // 	  printCoords(l,wrapCoords);
  // 	}
  //     else
  // 	{
  // 	  const GlbCoords neighCoords=glbGrill.coordsOfSite(neigh);
  // 	  printCoords(l,neighCoords);
  // 	}
  //   },site);
  // });
  
  // compLoop<Dir>([&glbGrill](const Dir& dir)
  // {
  //   LOGGER<<"Dir: "<<dir<<" surf size: "<<glbGrill.surfSize(dir);
  // });
  
  // LOGGER<<"Volume:            "<<glbGrill.vol();
  // LOGGER<<"Surf volume:       "<<glbGrill.surfVol();
  // LOGGER<<"Bulk volume:       "<<glbGrill.bulkVol();
  // LOGGER<<"Total halo volume: "<<glbGrill.totHaloVol();
  // for(Ori ori=0;ori<2;ori++)
  //   for(Dir dir=0;dir<4;dir++)
  //     LOGGER<<" halo offset("<<ori<<","<<dir<<"): "<<glbGrill.haloOffset(ori,dir);
}

int in_main(int narg,char** arg)
{
  testGrill();
  
  Tests::testS();
  
  LOGGER<<"/////////////////////////////////////////////////////////////////";
  
  testProd();
  
  LOGGER<<"/////////////////////////////////////////////////////////////////";
  
  // ASM_BOOKMARK_BEGIN("TEST_INDEX");
  // j=index(CompsList<SpaceTime>{5},SpinRow{3},SpaceTime{1});
  // LOGGER<<j;
  // ASM_BOOKMARK_END("TEST_INDEX");
  // LOGGER<<j;
  
  // LOGGER<<"Begin assign";
  testDag();
  
  return 0;
  
  
  StackTens<OfComps<SpinRow>,double> hostTens;
  for(SpinRow st=0;st<4;st++)
    hostTens(st)=st();
  
  for(SpinRow st=0;st<4;st++)
    LOGGER<<st()<<" "<<hostTens(st);
  
  /// Create
  const auto deviceTens=hostTens.getCopyOnExecSpace<ExecSpace::DEVICE>();
  
  StackTens<OfComps<SpinRow>,double> backHostTens=deviceTens;
  //dtd=dtg.getCopyOnExecSpace<ExecSpace::HOST>();
  
  for(SpinRow st=0;st<4;st++)
    LOGGER<<st()<<" "<<backHostTens(st);
  
  /////////////////////////////////////////////////////////////////
  
  DynamicTens<OfComps<SpinRow,SpaceTime,SpanRow>,double,ExecSpace::DEVICE> s(SpaceTime{4});
  DynamicTens<OfComps<SpanRow,SpaceTime,SpinRow>,double,ExecSpace::DEVICE> t(SpaceTime{4});
  
  auto rrr=s(SpanRow{0}).getRef();
  
  //using T=internal::_NodeRefOrVal<decltype(s.getRef())>;
  
  static_assert(not std::is_lvalue_reference_v<decltype(rrr)::SubNode<0>>,"");
  //auto ee=(decltype(rrr)::BoundExpr)(SpaceTime{0});
  // ExprRefOrVal<typename _E>
  
  t(SpanRow{0})=s(SpanRow{0});
  
  /////////////////////////////////////////////////////////////////
  
  // {
  //   DynamicTens<CompsList<SpaceTime,ComplId,SpinRow>,double,ExecSpace::HOST> a(SpaceTime{10});
  //   auto b=conj(a);
  //   auto c=conj(b);
  // }
  
  // {
  //   DynamicTens<CompsList<ComplId,SpaceTime,SpinRow>,double,ExecSpace::HOST> a(SpaceTime{10});
  //   StackTens<CompsList<SpinRow>,double> b;
  //   using t=grill::ExprRefOrVal<grill::DynamicTens<std::tuple<grill::ComplId, SpaceTime, Spin<grill::RwCl::ROW, 0> >, double, grill::ExecSpace::HOST>&>;
    

  //   auto ca=conj(a);
  //   real(ca);
  //   //real(conj(a));
  //   bindComps(conj(a),std::make_tuple(ComplId{0}));
  //   //auto e=real(conj(a));
  //   //b=real(conj(a))(SpaceTime{2});
  // }
  
  // {
  //   DynamicTens<CompsList<SpaceTime,ComplId,SpinRow>,double,ExecSpace::HOST> a;
  //   auto b=dag(a);
  //   (void)b;
  // }
  
  //   {
//     DynamicTens<OfComps<SpinRow,SpaceTime>,double,ExecSpace::HOST> dt(SpaceTime{3});
//     for(SpinRow s=0;s<4;s++)
//       for(SpaceTime st=0;st<3;st++)
// 	dt(st,s)=st()+3*s;
    
//     DynamicTens<OfComps<SpaceTime,SpinRow>,double,ExecSpace::HOST> td(SpaceTime{3});
    
//     td=dt;
    
//     for(SpinRow s=0;s<4;s++)
//       for(SpaceTime st=0;st<3;st++)
// 	LOGGER<<td(st,s)<<" "<<st()+3*s;
//   }
  
//   {
//     DynamicTens<OfComps<SpinRow,SpaceTime>,double,ExecSpace::HOST> t(SpaceTime{3});
//     for(SpinRow s=0;s<4;s++)
//       for(SpaceTime st=0;st<3;st++)
// 	t(st,s)=st()+3*s;
//     DynamicTens<OfComps<SpinRow,SpaceTime>,double,ExecSpace::DEVICE> dt(SpaceTime{3});
//     dt=t;
    
//     DynamicTens<OfComps<SpaceTime,SpinRow>,double,ExecSpace::DEVICE> td(SpaceTime{3});
    
//     td=dt;
    
//     DynamicTens<OfComps<SpaceTime,SpinRow>,double,ExecSpace::HOST> d(SpaceTime{3});
    
//     d=td;
//     for(SpinRow s=0;s<4;s++)
//       for(SpaceTime st=0;st<3;st++)
// 	LOGGER<<d(st,s)<<" "<<st()+3*s;
//   }
  
//   return 0;
  
//   StackTens<OfComps<SpinRow>,double> st;
//   DynamicTens<OfComps<SpinRow>,double,ExecSpace::HOST> dst;
//   st=dst;
  
//   dtg=dt;
  
//   {
//     auto rdt=dt.getRef();
//     rdt(SpinRow{3})=0;
//   }
//   //dt(SpinRow(0))=9;
//   st(SpinRow(2));
//   LOGGER<<"AAAA";
//   auto r=
//       TupleDiscriminate<SizeIsKnownAtCompileTime, C>::Valid{};
  
//   Spin<RwCl::ROW,0> s(1);
//   s=1;
//   // auto rr=Transp<Spin<RwCl::ROW>>::type{};
  
//   // decltype(s)::Transp t;
//   auto t=s.transp();
  
//   // SpinRow st;
//   // auto _st=st.transp();
  
//   s.sizeAtCompileTimeAssertingNotDynamic();
  
// #if not COMPILING_FOR_DEVICE
//   static_assert(StackedVariable<int>::execSpace==grill::ExecSpace::HOST,"We are issuing A on the host");
// #endif
  
  // ASM_BOOKMARK_BEGIN("TEST_ASSIGN");
  
  // StackedVariable<int> a;
  // a()=1;
  
  // DynamicVariable<int,ExecSpace::DEVICE> b;
  // b=a;
  
  // DynamicVariable<int,ExecSpace::DEVICE> c;
  // c=b;
  
  // StackedVariable<int> d;
  // d=c;
  // logger()<<"Result: "<<a()<<" -> "<<d();
  
  // ASM_BOOKMARK_END("TEST_ASSIGN");
  
  // ASM_BOOKMARK_BEGIN("TEST_UNROLL");
  // j=0;
  // UNROLLED_FOR((I,0,10),
  // 	       {
  // 		 j+=I;
  // 	       });
  // LOGGER<<j;
  // ASM_BOOKMARK_END("TEST_UNROLL");
  
  return 0;
}

int main(int narg,char** arg)
{
  grill::runProgram(narg,arg,in_main);
  
  return 0;
}
