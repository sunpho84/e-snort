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

namespace compFeat
{
  enum class IsTransposable{FALSE,TRUE};
  
  template <IsTransposable _IsT,
	    typename _C>
  struct Transposable;
  
  template <RwCl _RC,
	    int _Which,
	    template <RwCl,int> typename _C>
  struct Transposable<IsTransposable::TRUE,_C<_RC,_Which>>
  {
    static constexpr bool isTransposable=true;
    
    static constexpr RwCl RC=_RC;
    
    static constexpr RwCl transpRc=transpRwCl<RC>;
    
    static constexpr int Which=_Which;
    
    using Transp=_C<transpRc,_Which>;
    
    static constexpr bool isMatr=true;
  };
  
  template <typename _C>
  struct Transposable<IsTransposable::FALSE,_C>
  {
    static constexpr bool isTransposable=false;
    
    using Transp=_C;
    
    static constexpr bool isMatr=false;
  };
}

/////////////////////////////////////////////////////////////////


template <compFeat::IsTransposable IsTransposable,
	  typename Index,
	  typename Derived>
struct Comp  :
  compFeat::Transposable<IsTransposable,Derived>,
  BaseComp<Derived,Index>
{
  using Base=BaseComp<Derived,Index>;
    
  using Base::Base;
};


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
};

int j;

int main(int narg,char** arg)
{
  Spin<RwCl::ROW,0> s(1);
  s=1;
  // auto rr=Transp<Spin<RwCl::ROW>>::type{};
  
  // decltype(s)::Transp t;
  auto t=s.transp();
  
  SpaceTime st;
  auto _st=st.transp();
  
  s.sizeAtCompileTimeAssertingNotDynamic();
  
#if not COMPILING_FOR_DEVICE
  static_assert(StackedVariable<int>::execSpace()==esnort::ExecutionSpace::HOST,"We are issuing A on the host");
#endif
  
  ASM_BOOKMARK_BEGIN("TEST_ASSIGN");
  
  StackedVariable<int> a;
  a()=1;
  
  DynamicVariable<int,ExecutionSpace::DEVICE> b;
  b=a;
  
  DynamicVariable<int,ExecutionSpace::DEVICE> c;
  c=b;
  
  StackedVariable<int> d;
  d=c;
  logger()<<"Result: "<<a()<<" -> "<<d();
  
  ASM_BOOKMARK_END("TEST_ASSIGN");
  
  ASM_BOOKMARK_BEGIN("TEST_UNROLL");
  j=0;
  UNROLLED_FOR((I,0,10),
	       {
		 j+=I;
	       });
  LOGGER<<j;
  ASM_BOOKMARK_END("TEST_UNROLL");
  
  return 0;
}
