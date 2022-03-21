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

template <RwCl _RC=RwCl::ROW,
	  int _Which=0>
struct Spin :
  Comp<compFeat::IsTransposable::TRUE,
       int,
       Spin<_RC,_Which>>
{
  using Base=  Comp<compFeat::IsTransposable::TRUE,
       int,
		    Spin<_RC,_Which>>;
  
  using Base::Base;
  
  /// Size at compile time
  static constexpr int sizeAtCompileTime=4;
};

template <RwCl _RC=RwCl::ROW,
	  int _Which=0>
struct Span :
  Comp<compFeat::IsTransposable::TRUE,
       int,
       Spin<_RC,_Which>>
{
  using Base=  Comp<compFeat::IsTransposable::TRUE,
       int,
		    Spin<_RC,_Which>>;
  
  using Base::Base;
  
  /// Size at compile time
  static constexpr int sizeAtCompileTime=4;
};

struct SpaceTime :
  Comp<compFeat::IsTransposable::FALSE,
       int64_t,
       SpaceTime>
{
  using Base=Comp<compFeat::IsTransposable::FALSE,
		  int64_t,
		  SpaceTime>;
  
  using Base::Base;
  
  /// Size at compile time
  static constexpr int sizeAtCompileTime=0;
};

int j;

// template <typename E>
// struct SizeIsKnownAtCompileTime;

// struct S :
//   IndexComputer<S>
// {
  
// };

  using SpinRow=Spin<RwCl::ROW,0>;
  using SpanRow=Span<RwCl::ROW,0>;

StackTens<OfComps<SpinRow>,double> dt;
StackTens<OfComps<SpinRow>,double> pt;


void testSimdifiedAssign()
{
  ASM_BOOKMARK_BEGIN("TEST_SIMDIFIED_ASSIGN");
  dt.simdify()=pt.simdify();
  ASM_BOOKMARK_END("TEST_SIMDIFIED_ASSIGN");
}

int in_main(int narg,char** arg)
{
  device::initialize(Mpi::rank);
  
    ASM_BOOKMARK_BEGIN("TEST_INDEX");
  __asm volatile ("");
  
  using C=std::tuple<SpinRow,SpaceTime>;
  
  
  ASM_BOOKMARK_BEGIN("TEST_INDEX");
  j=index(CompsList<SpaceTime>{5},SpinRow{3},SpaceTime{1});
  LOGGER<<j;
  ASM_BOOKMARK_END("TEST_INDEX");
  LOGGER<<j;
  
  // using I=IndexComputer<C>;
  
  // I i(SpaceTime(10));

  LOGGER<<"Begin assign";
  
  for(SpinRow st=0;st<4;st++)
    pt(st)=st();
  
  //testSimdifiedAssign();
  
  for(SpinRow st=0;st<4;st++)
    LOGGER<<st()<<" "<<dt(st);
  
  // {
  //   DynamicTens<OfComps<SpinRow>,double,ExecutionSpace::HOST> dt;
  //   StackTens<OfComps<SpinRow>,double> st;
  //   st=dt;
  //   st.fillDynamicTens();
  // }
  
  DynamicTens<OfComps<SpinRow>,double,ExecutionSpace::DEVICE> dtg;
  dtg=dt.getRef();
  
  DynamicTens<OfComps<SpinRow>,double,ExecutionSpace::HOST> dtd;
  dtd=dtg;
  
  for(SpinRow st=0;st<4;st++)
    LOGGER<<st()<<" "<<dtd(st);
  
  {
    DynamicTens<OfComps<SpinRow,SpaceTime>,double,ExecutionSpace::HOST> dt(SpaceTime{3});
    for(SpinRow s=0;s<4;s++)
      for(SpaceTime st=0;st<3;st++)
	dt(st,s)=st()+3*s;
    
    DynamicTens<OfComps<SpaceTime,SpinRow>,double,ExecutionSpace::HOST> td(SpaceTime{3});
    
    td=dt;
    
    for(SpinRow s=0;s<4;s++)
      for(SpaceTime st=0;st<3;st++)
	LOGGER<<td(st,s)<<" "<<st()+3*s;
  }
  
  {
    DynamicTens<OfComps<SpinRow,SpaceTime>,double,ExecutionSpace::HOST> t(SpaceTime{3});
    for(SpinRow s=0;s<4;s++)
      for(SpaceTime st=0;st<3;st++)
	t(st,s)=st()+3*s;
    DynamicTens<OfComps<SpinRow,SpaceTime>,double,ExecutionSpace::DEVICE> dt(SpaceTime{3});
    dt=t;
    
    DynamicTens<OfComps<SpaceTime,SpinRow>,double,ExecutionSpace::DEVICE> td(SpaceTime{3});
    
    td=dt;
    
    DynamicTens<OfComps<SpaceTime,SpinRow>,double,ExecutionSpace::HOST> d(SpaceTime{3});
    
    d=td;
    for(SpinRow s=0;s<4;s++)
      for(SpaceTime st=0;st<3;st++)
	LOGGER<<d(st,s)<<" "<<st()+3*s;
  }
  
  return 0;
  
  StackTens<OfComps<SpinRow>,double> st;
  DynamicTens<OfComps<SpinRow>,double,ExecutionSpace::HOST> dst;
  st=dst;
  
  dtg=dt;
  
  {
    auto rdt=dt.getRef();
    rdt(SpinRow{3})=0;
  }
  //dt(SpinRow(0))=9;
  st(SpinRow(2));
  LOGGER<<"AAAA";
  auto r=
      TupleDiscriminate<SizeIsKnownAtCompileTime, C>::Valid{};
  
  Spin<RwCl::ROW,0> s(1);
  s=1;
  // auto rr=Transp<Spin<RwCl::ROW>>::type{};
  
  // decltype(s)::Transp t;
  auto t=s.transp();
  
  // SpinRow st;
  // auto _st=st.transp();
  
  s.sizeAtCompileTimeAssertingNotDynamic();
  
#if not COMPILING_FOR_DEVICE
  static_assert(StackedVariable<int>::execSpace==esnort::ExecutionSpace::HOST,"We are issuing A on the host");
#endif
  
  // ASM_BOOKMARK_BEGIN("TEST_ASSIGN");
  
  // StackedVariable<int> a;
  // a()=1;
  
  // DynamicVariable<int,ExecutionSpace::DEVICE> b;
  // b=a;
  
  // DynamicVariable<int,ExecutionSpace::DEVICE> c;
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
