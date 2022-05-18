#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

#include <grill.hpp>

using namespace grill;

DECLARE_TRANSPOSABLE_COMP(Col,int,3,col);

void test()
{
  using Lat=
    Lattice<U4D>;
  
  using GlbCoords=
    Lat::GlbCoords;
  
  using GlbCoords=
    Lat::GlbCoords;
  
  using GlbCoord=
    Lat::GlbCoord;
  
  using RankCoords=
    Lat::RankCoords;
  
  using SimdRankCoords=
    Lat::SimdRankCoords;
  
  using SimdRank=
    Lat::SimdRank;
  
  using SimdLocEoSite=
    Lat::SimdLocEoSite;
  
  using Parity=
    Lat::Parity;
  
  using Dir=U4D::Dir;
  
  constexpr GlbCoord Nt=8,Ns=4;
  
  GlbCoords glbSides{Nt,Ns,Ns,Ns};
  
  RankCoords nRanksPerDir{1,1,1,1};
  SimdRankCoords nSimdRanksPerDir;
  switch(maxAvailableSimdSize)
    {
    case 8:
      nSimdRanksPerDir={4,2,1,1};
      break;
    case 1:
      nSimdRanksPerDir=1;
      break;
    default:
      CRASH<<"Unknown nSimdRanksPerDir "<<maxAvailableSimdSize;
    }
  Dir parityDir=3;
  
  using Lat=Lattice<U4D>;
  
  Lattice<U4D> lattice(glbSides,nRanksPerDir,nSimdRanksPerDir,parityDir);
  using GaugeConf=Field<OfComps<Dir,ColRow,ColCln,ComplId>,double,Lat,LatticeCoverage::EVEN_ODD,FieldLayout::SIMDIFIABLE,ExecSpace::HOST>;
  using Su3Field=Field<OfComps<ColRow,ColCln,ComplId>,double,Lat,LatticeCoverage::EVEN_ODD,FieldLayout::SIMDIFIABLE,ExecSpace::HOST>;
  using ScalarField=Field<OfComps<>,double,Lat,LatticeCoverage::EVEN_ODD,FieldLayout::SIMDIFIABLE,ExecSpace::HOST>;
  
  GaugeConf conf(lattice,true);
  
  const char* path="/home/francesco/QCD/SORGENTI/grill/buildOpt/L4T8conf";
  FILE* fin=fopen(path,"r");
  if(fin==nullptr)
    CRASH<<"Unable to open the file "<<path;
  
  for(int t=0;t<Nt;t++)
    for(int x=0;x<Ns;x++)
      for(int y=0;y<Ns;y++)
	for(int z=0;z<Ns;z++)
	  for(Dir dir=0;dir<4;dir++)
	    for(ColRow colRow=0;colRow<3;colRow++)
	      for(ColCln colCln=0;colCln<3;colCln++)
		for(ComplId ri=0;ri<2;ri++)
		  {
		    const int id=index(std::tuple<>{},dir,colRow,colCln,ri);
		    
		    const auto [rank,parity,simdLocEoSite,simdRank]=
		      lattice.computeSimdEoRepOfGlbCoords(GlbCoords{t,x,y,z});
		    
		    int _t,_x,_y,_z,_id;
		    double d;
		    const int rc=fscanf(fin,"%d %d %d %d %d %lg",&_t,&_x,&_y,&_z,&_id,&d);
		    if(rc!=6)
		      CRASH<<" read "<<rc<<" instead of "<<6;
		    
		    for(auto p : {std::make_tuple(_t,t,"t"),{_x,x,"x"},{_y,y,"y"},{_z,z,"z"},{_id,id,"id"}})
		      if(std::get<0>(p)!=std::get<1>(p))
			CRASH<<"obtained "<<std::get<0>(p)<<" instead of "<<std::get<1>(p)<<" for "<<std::get<2>(p);
		    
		    if(rank==Mpi::rank)
		      conf(parity,simdLocEoSite,(dir+1)%Dir(4),colRow,colCln,ri,simdRank)=d;
		  }
  conf.updateHalo();
  
  LOGGER<<"Finished reading";
  
  if(0)
    loopOnAllComps<GaugeConf::Comps>(std::make_tuple(lattice.simdLoc.eoVol),
				     [&conf](const Parity& parity,const SimdLocEoSite& simdLocEoSite,const Dir& dir,const ColRow& colRow,const ColCln& colCln,const ComplId& ri,const SimdRank& simdRank)
				     {
				       LOGGER<<"Parity "<<parity<<" SimdLocEoSite "<<simdLocEoSite<<" Dir "<<dir<<" ColRow "<<colRow<<" ColCln "<<colCln<<" ComplId "<<ri<<" SimdRank "<<simdRank;
				       LOGGER<<" "<<conf(parity,simdLocEoSite,dir,colRow,colCln,ri,simdRank);
				     });
  
  ScalarField plaquette(lattice,true);
  plaquette=0;
  
  for(Dir dir=0;dir<4;dir++)
    for(Dir othDir=dir+1;othDir<4;othDir++)
      {
	Su3Field prod1(lattice,true);
	prod1=conf(dir)*(shift(conf,FW,dir)(othDir));
	Su3Field prod2(lattice,true);
	prod2=conf(othDir)*(shift(conf,FW,othDir)(dir));;
	
	plaquette=plaquette+real(trace(prod1*dag(prod2)));
      }
  
  double totPlaq=0.0;
  for(Parity parity=0;parity<2;parity++)
    for(SimdLocEoSite simdLocEoSite=0;simdLocEoSite<lattice.simdLoc.eoVol;simdLocEoSite++)
      compLoop<SimdRank>([&totPlaq,&plaquette,simdLocEoSite,parity](const SimdRank& simdRank)
      {
	totPlaq+=plaquette(parity,simdLocEoSite,simdRank);
      });
  
  totPlaq/=lattice.glbVol*6*3;
  LOGGER<<"Plaquette: "<<totPlaq;
}

int main(int narg,char**arg)
{
  grill::runProgram(narg,arg,[](int narg,char** arg){test();});
  
  return 0;
}
