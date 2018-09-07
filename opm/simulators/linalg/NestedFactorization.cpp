/*
  Copyright 2018 Equinor Energy AS

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
/// \file
/// \author  Markus Blatt HPC-Simulation-Software & Services http://www.dr-blatt.de/
#include <config.h>
#include <opm/autodiff/NestedFactorization.hpp>
#include <dune/grid/common/mcmgmapper.hh>

namespace Opm
{
namespace Detail
{
NFGeometryInformation::NFGeometryInformation::Plane::Plane(std::size_t max_lines)
{
    line_start_.reserve(max_lines + 1);
}
void NFGeometryInformation::NFGeometryInformation::Plane::nextLineStart(int start)
{
    line_start_.push_back(start);
}

NFGeometryInformation::NFGeometryInformation(const Dune::CpGrid& grid)
{
    const auto& dims = grid.logicalCartesianSize();
    int plane_index, line_index;

    if ( false ) //grid.eclipseOuputIndexLookup().size() > 0 )
    {
        // this is a reordered grid
        plane_index = 0;
        line_index  = 1;
    }
    else
    {
        plane_index = 2;
        line_index  = 1;
    }

    const int max_no_lines  = dims[line_index], max_no_planes = dims[plane_index];
    int max_line_length{}, no_lines{}, max_plane_width{}, no_planes{};
    planes_.reserve(max_no_planes);

    using GridView = Dune::CpGrid::LeafGridView;
    const auto& grid_view = grid.leafGridView();

#if DUNE_VERSION_NEWER(DUNE_GRID, 2, 6)
    using Mapper = Dune::MultipleCodimMultipleGeomTypeMapper<GridView>;
    Mapper mapper(grid_view, Dune::mcmgElementLayout());
#else
    using Mapper = Dune::MultipleCodimMultipleGeomTypeMapper<GridView, Dune::MCMGElementLayout>;
    Mapper mapper(grid_view);
#endif
    int old_plane_index = -1, old_line_index = -1;
    auto current_plane = planes_.begin() - 1; // one before the first future plane

    for ( auto element = grid_view.template begin<0>(),
              end = grid_view.template end<0>();
          element != end; ++element )
    {
        auto index = mapper.index(*element);
        std::array<int, 3> ijk;
        grid.getIJK(index, ijk);

        if ( ijk[plane_index] != old_plane_index )
        {
            if ( old_plane_index >= 0)
            {
                using std::max;
                max_plane_width = max(max_plane_width, no_lines);
                // indicate end of last line.
                current_plane->nextLineStart(index);
                std::cout<<"old_plane_index="<<old_plane_index<<" max_no_lines="<<max_no_lines
                         <<" line_size="<< current_plane->line_start_.size() << std::endl;
                assert(current_plane->line_start_.size() <= std::size_t(max_no_lines)+1);
            }
            assert(planes_.capacity() > planes_.size()); // Otherwise iterator is invalid
            // new plane starts here.
            planes_.emplace_back(max_no_lines);
            old_plane_index = ijk[plane_index];
            ++current_plane;
            ++no_planes;
            old_line_index = -1;
            no_lines = 0;
        }
        if ( ijk[line_index] != old_line_index )
        {
            // indicate start of next line.
            current_plane->nextLineStart(index);
            if ( old_line_index >= 0)
            {
                using std::max;
                const auto& line_start = current_plane->line_start_;
                auto end   = line_start.back();
                auto start =  *(line_start.end()-2);
                max_line_length = max(max_line_length,
                                      end - start);
            }
            old_line_index = ijk[line_index];
            ++no_lines;
        }
    }
    // end index for last line
    current_plane->nextLineStart(grid.size(0));
    assert(current_plane->line_start_.size() <= std::size_t(max_no_lines)+1);
    if ( old_plane_index >= 0)
    {
        using std::max;
        max_plane_width = max(max_plane_width, no_lines);
    }
}
} // end namespace detail
} // end namespace Opm
