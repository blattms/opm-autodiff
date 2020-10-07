/*
  Copyright 2020 OPM-OP AS

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
#include <config.h>
#include <opm/simulators/wells/ParallelWellInfo.hpp>

namespace Opm
{

void ParallelWellInfo::DestroyComm::operator()(Communication* comm)
{
#if HAVE_MPI
    // Only delete custom communicators.
    bool del = comm
        && (*comm != Dune::MPIHelper::getLocalCommunicator())
        && (*comm != MPI_COMM_WORLD && *comm != MPI_COMM_NULL);

    if ( del )
    {
        // Not 100% nice but safe as comm is deleted anyway
        // We can only access a copy and no reference.
        MPI_Comm mpi_comm = *comm;
        MPI_Comm_free(&mpi_comm);
    }
#endif
    delete comm;
}

ParallelWellInfo::ParallelWellInfo(const std::string& name)
    : name_(name), hasLocalCells_ (true),
      isOwner_(true), comm_(new Communication(Dune::MPIHelper::getLocalCommunicator()))
    {}

ParallelWellInfo::ParallelWellInfo(const std::pair<std::string,bool>& well_info)
    : name_(well_info.first), hasLocalCells_(well_info.second)
{
#if HAVE_MPI
    Communication allComm;
    MPI_Comm newComm;
    int color = hasLocalCells_ ? 1 : MPI_UNDEFINED;
    MPI_Comm_split(allComm, color, allComm.rank(), &newComm);
    comm_.reset(new Communication(newComm));
    isOwner_ = (comm_->rank() == 0);
#endif
}

bool operator<(const ParallelWellInfo& well1, const ParallelWellInfo& well2)
{
    return well1.name_ < well2.name_ || (! (well2.name_ < well1.name_) && well1.hasLocalCells_ < well2.hasLocalCells_);
}

bool operator==(const ParallelWellInfo& well1, const ParallelWellInfo& well2)
{
    bool ret = well1.name_ == well2.name_ && well1.hasLocalCells_ == well2.hasLocalCells_
        && well1.isOwner_ == well2.isOwner_;
#if HAVE_MPI
    using MPIComm = typename Dune::MPIHelper::MPICommunicator;
    ret = ret && (well1.comm_.get() == well2.comm_.get() // true for nullptr
                  || static_cast<MPIComm>(well1.communication()) == static_cast<MPIComm>(well2.communication()));
#endif
    return ret;
}

bool operator!=(const ParallelWellInfo& well1, const ParallelWellInfo& well2)
{
    return ! (well1 == well2);
}

bool operator<(const std::pair<std::string, bool>& pair, const ParallelWellInfo& well)
{
    return pair.first < well.name_ || ( !( well.name_ < pair.first ) && pair.second < well.hasLocalCells_ );
}

bool operator<( const ParallelWellInfo& well, const std::pair<std::string, bool>& pair)
{
    return well.name_ < pair.first || ( !( pair.first < well.name_ ) && well.hasLocalCells_ < pair.second );
}

bool operator==(const std::pair<std::string, bool>& pair, const ParallelWellInfo& well)
{
    return pair.first == well.name_ && pair.second == well.hasLocalCells_;
}

bool operator==(const ParallelWellInfo& well, const std::pair<std::string, bool>& pair)
{
    return pair == well;
}

bool operator!=(const std::pair<std::string, bool>& pair, const ParallelWellInfo& well)
{
    return pair.first != well.name_ || pair.second != well.hasLocalCells_;
}

bool operator!=(const ParallelWellInfo& well, const std::pair<std::string, bool>& pair)
{
    return pair != well;
}

} // end namespace Opm
