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

int main(int narg,char** arg)
{
  ASM_BOOKMARK_BEGIN("TEST_INDEX");
  __asm volatile ("");
  
  using C=std::tuple<SpinRow,SpaceTime>;

  std::integral_constant<int,index(CompsList<>{},SpinRow{3})>{};
  
  ASM_BOOKMARK_BEGIN("TEST_INDEX");
  j=index(CompsList<SpaceTime>{5},SpinRow{3},SpaceTime{1});
  LOGGER<<j;
  ASM_BOOKMARK_END("TEST_INDEX");
  LOGGER<<j;
  
  // using I=IndexComputer<C>;
  
  // I i(SpaceTime(10));
  
  DynamicTens<OfComps<SpaceTime>,double,ExecutionSpace::HOST> dt(CompsList<SpaceTime>{5});
  DynamicTens<OfComps<SpaceTime>,double,ExecutionSpace::DEVICE> dtg(CompsList<SpaceTime>{5});
  for(SpaceTime st=0;st<SpaceTime(5);st=st+SpaceTime(1))
    dt(st)=st();
  
  dtg=dt;
  for(SpaceTime st=0;st<SpaceTime(5);st=st+SpaceTime(1))
    LOGGER<<st()<<" "<<dtg(st);
  
  StackTens<OfComps<SpinRow>,double> st;
  dtg=dt;
  
  {
    auto rdt=dt.getRef();
    rdt(SpaceTime{3})=0;
  }
  //dt(SpaceTime(0))=9;
  st(SpinRow(2));
  LOGGER<<"AAAA";
  auto r=
      TupleDiscriminate<SizeIsKnownAtCompileTime, C>::Valid{};
  
  Spin<RwCl::ROW,0> s(1);
  s=1;
  // auto rr=Transp<Spin<RwCl::ROW>>::type{};
  
  // decltype(s)::Transp t;
  auto t=s.transp();
  
  // SpaceTime st;
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
