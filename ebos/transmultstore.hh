// -*- mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-
// vi: set et ts=4 sw=4 sts=4:
/*
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

#include <dune/common/parallel/mpihelper.hh>

#include <opm/simulators/utils/ParallelRestart.hpp>

#include <unordered_map>
#include <vector>

#ifndef TRANSMULTSTORE_HH
#define TRANSMULTSTORE_HH

namespace Opm
{


/// \brief A class for storing and broadcasting multipliers from EclipseState
class TransMultStore
{
public:
    TransMultStore() = default;
    TransMultStore(std::size_t cartesianSize)
        : multipliers_(cartesianSize)
    {}
    /// \brief adds/accumulates a multiplier between cells
    ///
    /// If there is already a multiplier, it will be
    /// multiplied by this one. Otherwise the multiplier
    /// is simply stored.
    /// \param ci Cartesian index of first cell
    /// \param cj Cartesian index of second cell
    /// \param mult The multiplier.
    void setMultiplier(int ci, int cj, double mult)
    {
        using std::swap;
        if (ci>cj)
            swap(ci,cj);
        multipliers_[ci][cj] = mult;
    }
    /// \brief Get a stored multiplier between cells

    /// \param ci Cartesian index of first cell
    /// \param cj Cartesian index of second cell
    /// \return The multiplier.
    double getMultiplier(int ci, int cj) const
    {
        using std::swap;
        if (ci>cj)
            swap(ci,cj);
        // \todo maybe change to non-throwing operator[]
        return multipliers_[ci].at(cj);
    }


    void broadcast(const Dune::CollectiveCommunication<typename Dune::MPIHelper::MPICommunicator>& comm)
    {
        if (comm.rank()==0)
        {
            std::size_t size = Mpi::packSize(multipliers_, comm);
            std::vector<char> buffer(size);
            int position=0;
            Mpi::pack(multipliers_, buffer, position, comm);
            comm.broadcast(&size, 1, 0);
            comm.broadcast(buffer.data(), position, 0);
        }
        else
        {
            std::size_t size;
            comm.broadcast(&size, 1, 0);
            std::vector<char> buffer(size);
            comm.broadcast(buffer.data(), size, 0);
            int position{};
            multipliers_.clear();
            Mpi::unpack(multipliers_, buffer, position, comm);
        }
    }
private:
    /// \brief stores the multipliers for transmissibility
    /// mulipliers_[i] is an unsorted array of pairs of cartesian cell index j>i and corresponding
    /// multipliers. Each cell with index j is connected to the cell with cartesian index i
    std::vector<std::unordered_map<int,double> > multipliers_;
};
} // end namespace Opm
#endif // TRANSMULTSTORE_HH
