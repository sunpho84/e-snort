#ifndef _QCD_HPP
#define _QCD_HPP

#include "expr/nodes/conj.hpp"
#ifdef HAVE_CONFIG_H
# include <config.hpp>
#endif

/// \file grill/qcd.hpp

#include <lattice/lattice.hpp>
#include <lattice/universe.hpp>

namespace grill::QCD
{
  /// 4-D universe
  using U4D=
    Universe<4>;
  
  using Dir=
    U4D::Dir;
  
  using GlbLattice=
    grill::Lattice<U4D>;
  
  using GlbCoords=
    GlbLattice::GlbCoords;
  
  using GlbCoords=
    GlbLattice::GlbCoords;
  
  using GlbCoord=
    GlbLattice::GlbCoord;
  
  using RankCoords=
    GlbLattice::RankCoords;
  
  using SimdRankCoords=
    GlbLattice::SimdRankCoords;
  
  using SimdRank=
    GlbLattice::SimdRank;
  
  using SimdLocEoSite=
    GlbLattice::SimdLocEoSite;
  
  using Parity=
    GlbLattice::Parity;
  
  /////////////////////////////////////////////////////////////////
  
  static constexpr int NCol=3;
  
  DECLARE_TRANSPOSABLE_COMP(Col,int,NCol,col);
  
  static constexpr int NSpin=4;
  
  DECLARE_TRANSPOSABLE_COMP(Spin,int,NSpin,spin);
  
  using ScalarFieldComps=
    CompsList<>;
  
  using ComplScalarFieldComps=
    CompsList<ComplId>;
  
  using LorentzFieldComps=
    CompsList<Dir>;
  
  using U1LinksFieldComps=
    CompsList<Dir,ComplId>;
  
  using SU3FieldComps=
    CompsList<ColRow,ColCln,ComplId>;
  
  using ColorFieldComps=
    CompsList<ColRow,ComplId>;
  
  using SpinColorFieldComps=
    CompsList<SpinRow,ColRow,ComplId>;
  
  using SU3LinksFieldComps=
    CompsList<Dir,ColRow,ColCln,ComplId>;
  
  
}

#endif
