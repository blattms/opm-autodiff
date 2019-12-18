// -*- mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-
// vi: set et ts=4 sw=4 sts=4:
/*
  Copyright 2019 Equinor ASA

  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 2 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.

  Consult the COPYING file in the top-level source directory of this
  module for the precise wording of the license and the list of
  copyright holders.
*/
/*!
 * \file
 *
 * \author Dr. Blatt - HPC-Simulation-Software & Services
 */

#include <opm/parser/eclipse/EclipseState/Grid/EclipseGrid.hpp>
#include <dune/common/parallel/collectivecommunication.hh>
#include <dune/common/parallel/mpihelper.hh>
#include <vector>
namespace Opm
{
/// \brief Utility class that replicates cartesian grid data for porosity
///        and transmissibility calculations.

class EclShallowGrid
{
public:
    EclShallowGrid() = default;
    /// \brief Constructor to be called on non-io ranks.
    /// \param cartesianSize Number of elements in the cartesian grid.
    EclShallowGrid(const std::size_t& cartesianSize)
        : cartesianSize_(cartesianSize),
          cellCenters_(cartesianSize_, {0.0,0.0,0.0}),
          cellVolumes_(cartesianSize_, 0.0),
          broadcasted_()
    {
    }
    /// \brief Constructor to be called on io rank.
    /// \param grid The eclipse grid.
    EclShallowGrid(const EclipseGrid& grid)
        : cartesianSize_(grid.getCartesianSize()),
          cellCenters_(cartesianSize_, {0.0,0.0,0.0}),
          cellVolumes_(cartesianSize_, 0.0),
          broadcasted_(),
          minPvMode_(grid.getMinpvMode()),
          pinchMode_(grid.getPinchOption()),
          multzOption_(grid.getMultzOption())
    {
        for (std::size_t i = 0; i < cartesianSize_; ++i)
        {
            cellCenters_[i] = grid.getCellCenter(i);
            cellVolumes_[i] = grid.getCellVolume(i);
        }
        if (minPvMode_ == Opm::MinpvMode::ModeEnum::OpmFIL)
            minPvVector_ = grid.getMinpvVector();
    }

    /// \brief Broadcast the data from the IO rank to all
    /// \param comm The collective communication object to use.
    void broadcast(const Dune::CollectiveCommunication<typename Dune::MPIHelper::MPICommunicator>& comm )
    {
        comm.broadcast(&minPvMode_, 1, 0);
        comm.broadcast(&pinchMode_, 1, 0);
        comm.broadcast(&multzOption_, 1, 0);
        comm.broadcast(cellCenters_.data(), cellCenters_.size(), 0);
        comm.broadcast(cellVolumes_.data(), cellVolumes_.size(), 0);

        if (minPvMode_ == Opm::MinpvMode::ModeEnum::OpmFIL)
        {
            std::size_t minPvSize = minPvVector_.size();
            comm.broadcast(&minPvSize, 1, 0);
            minPvVector_.resize(minPvSize); // vector has same size on all ranks.
            comm.broadcast(minPvVector_.data(), minPvVector_.size(), 0);
        }
        broadcasted_ = true;
    }

    Opm::MinpvMode::ModeEnum getMinPvMode() const
    {
        assert(broadcasted_);
        return minPvMode_;
    }

    Opm::PinchMode::ModeEnum getPinchMode() const
    {
        assert(broadcasted_);
        return pinchMode_;
    }

    Opm::PinchMode::ModeEnum getMultzOption() const
    {
        assert(broadcasted_);
        return multzOption_;
    }

    const std::array<double, 3> getCellCenter(std::size_t globalIndex) const
    {
        assert(broadcasted_);
        return cellCenters_[globalIndex];
    }

    double getCellVolume(std::size_t globalIndex) const
    {
        assert(broadcasted_);
        return cellVolumes_[globalIndex];
    }
private:
    std::size_t cartesianSize_;
    std::vector<std::array<double, 3> > cellCenters_;
    std::vector<double> cellVolumes_;
    std::vector<double> minPvVector_;
    bool broadcasted_;
    Opm::MinpvMode::ModeEnum minPvMode_;
    Opm::PinchMode::ModeEnum pinchMode_;
    Opm::PinchMode::ModeEnum multzOption_;
};

} // end namespace Opm
