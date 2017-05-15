/*
  Copyright 2017 Dr. Blatt - HPC-Simulation-Software & Services
  Copyright 2017 Statoil ASA.

  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef OPM_DUMMYAUXILIARYMODULE_HEADER_INCLUDED
#define OPM_DUMMYAUXILIARYMODULE_HEADER_INCLUDED

#include <ewoms/aux/baseauxiliarymodule.hh>

double neighbor_add_time = 0.0;

namespace Opm
{
template<class TypeTag>
class DummyAuxiliaryModule
    : public Ewoms::BaseAuxiliaryModule<TypeTag>
{
    typedef typename GET_PROP_TYPE(TypeTag, GlobalEqVector) GlobalEqVector;
    typedef typename GET_PROP_TYPE(TypeTag, JacobianMatrix) JacobianMatrix;

public:

    using NeighborSet = typename
        Ewoms::BaseAuxiliaryModule<TypeTag>::NeighborSet;

    DummyAuxiliaryModule()
    {
    }

    unsigned numDofs() const
    {
        // No extra dofs are inserted for wells.
        return 0;
    }

    void addNeighbors(std::vector<NeighborSet>& neighbors) const
    {
        return;
    }
    
    void applyInitial()
    {
        return;
    }

    void linearize(JacobianMatrix& matrix, GlobalEqVector& residual)
    {
        return;
    }
};

} // end namespace OPM
#endif
