#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

#include <grill.hpp>

using namespace grill;


DECLARE_TRANSPOSABLE_COMP(Spin,int,4,spin);

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

double d=1;

void testGrill()
{
  using U4D=Universe<4>;
  // using Dir=U4D::Dir;
  
  using Lattice=Lattice<U4D>;
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
  
  Lattice lattice(glbSides,rankSides,simdRankSides,1);
  
  using F=Field<OfComps<Spin>,double,Lattice,LatticeCoverage::EVEN_ODD,FieldLayout::SIMDIFIABLE,ExecSpace::DEVICE>;
  
  F f(lattice),g(lattice),h(lattice);
  
  // using FF=decltype(ff);
  // auto FC=FF::Comps{};
  
  // auto e=ff(Parity(0),SimdLocEoSite(0),SpinRow(0),NonSimdifiedComp<int, 2>(0));
  // loopOnAllComps<F::Comps>(f.getDynamicSizes(),[&f,&g](const auto...c){
  //   f(c...)=1;
  //   g(c...)=1;
  // });
  
  f=d;
  g=d;
  
  ASM_BOOKMARK_BEGIN("feq0");
  h=g+f;
  ASM_BOOKMARK_END("feq0");

#if ENABLE_SIMD
  LOGGER<<"Is two: "<<h(Parity(0),SimdLocEoSite(0),SpinRow(0),SimdRank(0));
#endif
}

int main(int narg,char**arg)
{
  grill::runProgram(narg,arg,[](int narg,char** arg){testGrill();});
  
  return 0;
}
