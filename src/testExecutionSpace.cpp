#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

#include <algorithm>

#include <cstdarg>
#include <cstdio>
#include <unistd.h>
#include <iostream>

#include <esnort.hpp>

using namespace esnort;

/////////////////////////////////////////////////////////////////

DEFINE_TRANSPOSABLE_COMP(Spin,int,4,spin);
DEFINE_TRANSPOSABLE_COMP(Span,int,4,span);
DEFINE_UNTRANSPOSABLE_COMP(SpaceTime,int64_t,0,spaceTime);

int j;

// template <typename E>
// struct SizeIsKnownAtCompileTime;

// struct S :
//   IndexComputer<S>
// {
  
// };

  using SpinRow=Spin<RwCl::ROW,0>;
  using SpanRow=Span<RwCl::ROW,0>;

// struct InvalidExpr :
//   public Node<InvalidExpr>
// {
// };

// InvalidExpr e;

using Comps=CompsList<ComplId,SpinRow,SpinCln>;
StackTens<Comps,double> a;
StackTens<CompsList<ComplId,SpinRow,SpinCln>,double> b;

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
    StackTens<CompsList<ComplId>,double> a;
    real(a)=0.0;
    imag(a)=1.0;

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
    
    auto b=(a*dag(a));
    
    //LOGGER<<b.factNode<0>().storage<<" "<<b.factNode<1>().conjExpr.storage<<" "<<a.storage;
    
    // .fillDynamicTens();
    
    LOGGER<<real(b)<<" "<<imag(b);
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

int in_main(int narg,char** arg)
{
  testProd();
  
  return 0;
  
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
  
  using T=internal::_NodeRefOrVal<decltype(s.getRef())>;
  
  static_assert(not std::is_lvalue_reference_v<decltype(rrr)::BoundExpr>,"");
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
  //   using t=esnort::ExprRefOrVal<esnort::DynamicTens<std::tuple<esnort::ComplId, SpaceTime, Spin<esnort::RwCl::ROW, 0> >, double, esnort::ExecSpace::HOST>&>;
    

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
//   static_assert(StackedVariable<int>::execSpace==esnort::ExecSpace::HOST,"We are issuing A on the host");
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
  esnort::runProgram(narg,arg,in_main);
  
  return 0;
}
