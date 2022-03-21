#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

#include <esnort.hpp>

using namespace esnort;


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

  using SpinRow=Spin<RwCl::ROW,0>;

StackTens<OfComps<SpinRow>,double> dt;
StackTens<OfComps<SpinRow>,double> pt;


void testSimdifiedAssign()
{
  ASM_BOOKMARK_BEGIN("TEST_SIMDIFIED_ASSIGN");
  //std::bool_constant<dt.canSimdify>& a=1;
  dt=pt;
  ASM_BOOKMARK_END("TEST_SIMDIFIED_ASSIGN");
}

int main(int narg,char**arg)
{
  esnort::runProgram(narg,arg,[](int narg,char** arg){testSimdifiedAssign();});
  
  return 0;
}