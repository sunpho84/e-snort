#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

#include <esnort.hpp>

using namespace esnort;


DEFINE_TRANSPOSABLE_COMP(Spin,int,4);

StackTens<OfComps<SpinRow,SpinCln>,double> dt;

StackTens<OfComps<SpinRow,SpinCln>,double> pt;

// double data[4],dota[4];

void testSimdifiedAssign()
{
  ASM_BOOKMARK_BEGIN("TEST_SIMDIFIED_ASSIGN");
  //std::bool_constant<dt.canSimdify>& a=1;
  //_mm256_store_pd(data,_mm256_load_pd(dota));
  dt=pt;
  ASM_BOOKMARK_END("TEST_SIMDIFIED_ASSIGN");
}

int main(int narg,char**arg)
{
  esnort::runProgram(narg,arg,[](int narg,char** arg){testSimdifiedAssign();});
  
  return 0;
}
