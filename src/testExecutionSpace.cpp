#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

#include <algorithm>

#include <cstdarg>
#include <cstdio>
#include <unistd.h>
#include <iostream>

#include <resources/device.hpp>

#include <expr/executionSpace.hpp>
#include <expr/expr.hpp>
#include <resources/valueWithExtreme.hpp>
#include <resources/environmentFlags.hpp>
#include <tensor/stackedVariable.hpp>

#include <esnort.hpp>

using namespace esnort;

/////////////////////////////////////////////////////////////////

DEFINE_TRANSPOSABLE_COMP(Spin,int,4);
DEFINE_TRANSPOSABLE_COMP(Span,int,4);
DEFINE_UNTRANSPOSABLE_COMP(SpaceTime,int64_t,0);

int j;

// template <typename E>
// struct SizeIsKnownAtCompileTime;

// struct S :
//   IndexComputer<S>
// {
  
// };

  using SpinRow=Spin<RwCl::ROW,0>;
  using SpanRow=Span<RwCl::ROW,0>;

int in_main(int narg,char** arg)
{
  // ASM_BOOKMARK_BEGIN("TEST_INDEX");
  // j=index(CompsList<SpaceTime>{5},SpinRow{3},SpaceTime{1});
  // LOGGER<<j;
  // ASM_BOOKMARK_END("TEST_INDEX");
  // LOGGER<<j;
  
  // LOGGER<<"Begin assign";
  
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
  
  
  t(SpanRow{0})=s(SpanRow{0});

  
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
