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
///
#ifndef OPM_NESTEDFACTORIZATION_HEADER_INCLUDED
#define OPM_NESTEDFACTORIZATION_HEADER_INCLUDED

#include <vector>
#include <cstdlib>

#include <opm/grid/CpGrid.hpp>

#include <dune/istl/bdmatrix.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/preconditioner.hh>

namespace Opm
{
namespace Detail
{
/// \brief Information about the geometry needed for Nested factorization.
class NFGeometryInformation
{
public:
    /// \brief Index Information of the plane.
    struct Plane
    {
        /// \Constructor
        ///
        /// \max_lines The maximum lines expected (governd by cartesian
        ///            dimension of the grid.
        Plane(std::size_t max_lines);

        /// \brief Indicate first index of next line
        void nextLineStart(int start);

        /// \brief Starting indices of each line.
        ///
        /// line_start[i] is first index of line i.
        /// line_start[i] is one past the last index of line i.
        std::vector<int> line_start_;
    };

    explicit NFGeometryInformation(const Dune::CpGrid& grid);

    std::vector<Plane> planes_;

};
}


template<class Matrix, class Vector>
class NestedFactorization
    : public Dune::Preconditioner<Vector,Vector>
{
    /// \brief Type of the matrix block used.
    using MatrixBlock = typename Matrix::block_type;
    /// \brief Type of the vector block used.
    using VectorBlock = typename Vector::block_type;
    /// \brief Type of the matrix used to store M.
    using DiagMatrix  = Dune::BDMatrix<MatrixBlock>;

public:
    /// \brief Constructor
    /// \param A The matrix of the linear system.
    /// \param grid The corner point grid used for discretization.
    /// \param alpha The alpha factor used when constructing M.
    /// \parame beta The beta factor used when constructing M.
    NestedFactorization(const Dune::CpGrid& grid,
                        double alpha = 1, double beta = 1)
        : nf_geometry_(grid), inverseM_(grid.size(0)),
          inverseMU1_(grid.size(0), grid.size(0), grid.size(0), Matrix::row_wise),
          a_(), alpha_(alpha), beta_(beta)
    {
        for ( auto iter = inverseMU1_.createbegin(), end = inverseMU1_.createend();
              iter != end; ++iter)
        {
                iter.insert(iter.index()+1);
        }
    }

    /// \brief Compute the factorization
    /// \param A The matrix of the linear system.
    void factorize(const Matrix& A)
    {
        a_ = &A;
        std::size_t plane_index=0;
        for ( const auto& plane : nf_geometry_.planes_ )
        {
            if ( plane_index > 0 && beta_ != 0 )
            {
                // Solve P^T_{plane_index-1} x = (L_3^{plane_index-1])^T.1
                // 1. Compute d = (L_3^{plane_index-1])^T.1
                auto end   = plane.line_start_.back();
                auto start = plane.line_start_.front();
                auto start_previous = nf_geometry_.planes_[plane_index - 1].line_start_.front();
                auto d = computeLTransposed(start_previous, start, end);
                assert(d.size() == std::size_t(start-start_previous));
                // 2. Solve P^T_{plane_index-1} x = d
                //    d=x
                solve_plane_transposed(d, plane_index-1);
                // M_i -= \beta U_3^{plane_index-1}^T d
                mAddUTransposedD(d, start_previous, start, end);
            }
            updateFromLines(plane_index, plane);
            ++plane_index;
        }
    }
#if DUNE_VERSION_NEWER(DUNE_ISTL, 2, 6)
    Dune::SolverCategory::Category category() const override
    {
      return Dune::SolverCategory::sequential;
    }

#else
    // define the category
    enum {
        //! \brief The category the preconditioner is part of.
        category = Dune::SolverCategory::sequential
    };
#endif

    void pre(Vector&, Vector&) override {}

    void post(Vector&) override {}

    /// \copydoc Preconditioner::apply(X&,const Y&)
    void apply(Vector& v, const Vector &d)
    {
        assert(a_);
        // Solving (P+L3)(I+P^{-1} U3) =v
        // 1. foward sweep: (solve P+L3) y = v;
        std::size_t plane_index=0;
        std::size_t last_plane_start=0;
        Vector y=d;

        for ( const auto& plane : nf_geometry_.planes_ )
        {
            if ( plane_index > 0 )
            {
                const auto end   = plane.line_start_.back();
                const auto start = plane.line_start_.front();
                auto       row   = (*a_).begin()+start;

                for ( ; row.index() < end; ++row )
                {
                    auto col = row->begin();
                    //Skip all lines despite the previous one
                    // \todo doesn't this neglect NNCs?
                    while ( col != row->end() && col.index() < last_plane_start ){
                        ++col;
                    }
                    // Treat the previous plane
                    for ( ; col != row->end() && col.index() < start; ++col)
                    {
                        col->mmv(y[col.index()], y[row.index()]);
                    }
                }
            }
            solve_plane(y, plane_index, 0);
            ++plane_index;
        }

        // 2. backward sweep: solve y= (I+P^{-1}U3) x
        // overwrite y with x
        auto end_last_plane = nf_geometry_.planes_.back().line_start_.back();
        plane_index = nf_geometry_.planes_.size() - 1;

        for ( auto plane = nf_geometry_.planes_.rbegin(); plane != nf_geometry_.planes_.rend();
              end_last_plane = plane->line_start_.back(),  ++plane, --plane_index)
        {
            const auto end   = plane->line_start_.back();
            const auto start = plane->line_start_.front();
            std::copy(y.begin()+start, y.begin()+end, v.begin()+start); // \todo move out of for loop
            if(end_last_plane != end)
            {
                // Compute t =U3 x
                std::vector<VectorBlock> t(end-start);
                //order does not matter within the plane as we neglect influences
                // within the plane at this stage.
                for ( auto row = (*a_).begin() + start, rend = (*a_).begin() + end;
                      row != rend; ++row)
                {
                    auto col = row->beforeEnd(), cend= row->beforeBegin();
                    // skip planes further away \todo Maybe we should honor them to get NNCs
                    while( col != cend && col.index() >= end_last_plane)
                    {
                        --col;
                    }
                    for ( ; col != cend && col.index() >= end; --col)
                    {
                        col->umv(y[col.index()], t[row.index()-start]);
                    }
                }
                solve_plane(t, plane_index, start);
                auto titer = t.begin();
                for ( auto viter = v.begin() + start, vend = v.begin() + end;
                      viter != vend; ++viter, ++titer)
                {
                    (*viter) -= *titer;
                }
            }
        }
    }
private:
    template<typename V>
    void solve_plane(V& d, std::size_t plane_index, std::size_t offset)
    {
        auto& plane = nf_geometry_.planes_[plane_index];
        // Solve (T+L2) (I+ T^{-1}U2) x = d
        // 1. forward sweep: solve (T+L2) y = d
        auto first_line_start   = plane.line_start_.front();
        assert( offset == 0 || offset == first_line_start );
        auto start = d.begin() + (first_line_start - offset);
        auto end = d.begin() + (plane.line_start_.back() - offset);/*
        std::vector<VectorBlock> y(d.begin() + (first_line_start - offset),
        d.begin() + (plane.line_start_.back() - offset) );*/
        auto size = end-start;
        std::vector<VectorBlock> y;
        y.resize(size);
        auto curr=y.begin();
        for(;start!=end;++start,++curr)
            *curr=*start;
        assert(curr == y.end());
        for ( auto line_start = plane.line_start_.begin(), line_end = line_start + 1, previous_line_start = line_start;
              line_end != plane.line_start_.end();
              previous_line_start = line_start, ++line_start, ++line_end)
        {
            if ( line_start != previous_line_start )
            {
                for ( auto row = (*a_).begin() + *line_start, rend = (*a_).begin() + *line_end;
                      row != rend; ++row)
                {
                    //Skip all lines despite the previous one
                    // \todo doesn't this neglect NNCs?
                    auto col = row->begin();
                    while ( col != row->end() && col.index() < *previous_line_start )
                    {
                        ++col;
                    }
                    // Treat influences from the previous line
                    for ( ; col != row->end() && col.index() < *line_start; ++col )
                    {
                        col->mmv(y[col.index() - first_line_start],
                                 y[row.index() - first_line_start]);
                    }
                }
            }
            solve_line(y.begin() + *line_start, *line_start, *line_end);
        }

        // 2. backward sweep: solve y = (I+P^-1}U2) x
        auto last_line_end = plane.line_start_.rbegin();
        for ( auto line_end = last_line_end, line_start = line_end + 1;
              line_start != plane.line_start_.rend();
              last_line_end = line_end, ++line_start, ++line_end)
        {
             // \todo move copy out of for loop
            std::copy(y.begin() + *line_start, y.begin() + *line_end, d.begin() + *line_start);
            if ( last_line_end != line_end )
            {
                // Compute t = U2 y
                std::vector<VectorBlock> t(*line_end -*line_start);
                //order does not matter within the line as we neglect influnces
                // within the line at this stage. Hence traverse forward.
                for ( auto row = (*a_).begin() + *line_start, rend = (*a_).begin() + *line_end;
                      row != rend; ++row)
                {
                    auto col = row->beforeEnd(), cend= row->beforeBegin();
                    // skip all lines except the last one
                    // \todo Doesn't this neglect NNCs and should be changed?
                    while( col != cend && col.index() >= *last_line_end)
                    {
                        --col;
                    }
                    for ( ; col != cend && col.index() >= *line_end; --col)
                    {
                        col->umv(y[col.index() - first_line_start], t[row.index() - *line_start]);
                    }
                }
                solve_line(t.begin(), *line_start, *line_end);

                auto titer = t.begin();
                for ( auto diter = d.begin() + *line_start - offset,
                           dend  = d.begin() + *line_end - offset;
                      diter != dend; ++diter, ++titer)
                {
                    (*diter) -= *titer;
                }
            }
        }
    }

    void solve_line(typename std::vector<VectorBlock>::iterator d,
                    std::size_t line_start, std::size_t line_end, std::size_t offset=0)
    {
        // Solve (M + L1) (I + M^{-1} U1) x = d
        // 1. forward sweep: Solve (M + L1) y = d
        std::vector<VectorBlock> y(d, d + line_end - line_start);
        auto miter = inverseM_.begin(); // inverse of M
        auto yiter = y.begin();

        for ( auto row = (*a_).begin() + line_start, rend = (*a_).begin() + line_end;
              row != rend; ++row, ++miter, ++yiter)
        {
            VectorBlock t = *yiter;
            if ( row.index() != line_start )
            {
                auto col = row->find(row.index());
                if(col != row->end() && col.offset() > 0 && (--col).index() == row.index())
                {
                    // Connection to  a previous cell
                    // \todo neglecting NNCs here!
                    col->mmv(y[col.index() - line_start], t);
                }
            }
            assert(miter->begin() != miter->end());
            miter->begin()->mv(t, *yiter);
        }
        // 2. backward sweep: Solve (I + M^{-1} U1 x = y
        auto ryiter = y.rbegin();
        auto dend = d + ( line_end - line_start - 1);
        for ( auto row = inverseMU1_.beforeEnd() - (inverseMU1_.N() - line_end),
                  rend = inverseMU1_.beforeEnd() - (inverseMU1_.N() - line_start);
              row != rend; --row, ryiter++, --dend)
        {
            *dend = *ryiter;
            if ( row.index() != (inverseMU1_.N() - line_end -1) )
            {
                // \todo Still neglecting NNCs
                auto col = row->find(row.index()+1);
                if(col != row->end())
                {
                    col->mmv(*(d + (col.index() - line_start)), *dend);
                }
            }
        }
    }
    void solve_plane_transposed(std::vector<VectorBlock>& d, int plane_index)
    {
        auto& plane = nf_geometry_.planes_[plane_index];
        // Solve P^T x = d with P=(T+L2) (I + T^{-1}U_2)
        // 1. Solve (I+T^{-1}U_2)^T y = d;
        std::vector<VectorBlock> y(plane.line_start_.back() - plane.line_start_.front());
        // 1.1 first line
        auto line_start = plane.line_start_.begin();
        auto first_line_start = *line_start;
        auto previous_line_start = line_start;
        auto line_end   = plane.line_start_.begin() + 1;
        int start = *line_start, end = plane.line_start_[1];
        std::copy(d.begin(), d.begin()+*line_start, y.begin()); // y = d

        // 1.2 all the other lines
        for( ++line_start, ++line_end; line_end != plane.line_start_.end();
             ++line_start, ++line_end, ++previous_line_start)
        {
            std::vector<VectorBlock> y_line(d.begin() + *previous_line_start - first_line_start,
                                            d.begin() + *line_start- first_line_start);
            // First solve line: T^T y = d on previous line
            solve_line_transposed(y_line, *previous_line_start, *line_start);
            // U_2 is in [block previous_line_start : line_start, line_start : line_end
            // Hence product U_2^T x will not add influence from current line and order
            // does not matter.
            // Start with y=d.
            std::copy(d.begin()+*line_start-first_line_start, d.begin()+*line_end-first_line_start,
                      y.begin()+*line_start-first_line_start);
            for( auto row_index = *previous_line_start; row_index < *line_start; ++row_index)
            {
                auto col = (*a_)[row_index].find(row_index);
                while ( col != (*a_)[row_index].end() && col.index() < *line_start)
                {
                    ++col;
                }

                for ( ; col != (*a_)[row_index].end() && col.index() < *line_end; ++col)
                {
                    col->mmtv(y_line[row_index - *previous_line_start], y[col.index() - start]);
                }
            }
        }

        // 2. Solve (T+L2)^T x = y;
        // 2.1 Last line, no L2
        auto line_rend = plane.line_start_.rbegin();
        auto line_rstart = line_rend +1;
        auto last_line_rend = line_rend;
        // \todo prevent the copying by offset parameter to solve_line_transpose
        std::vector<VectorBlock> x_line(y.begin() + *line_rstart - first_line_start,
                                        y.begin() + *line_rend - first_line_start);
        solve_line_transposed(x_line, *line_rstart, *line_rend);
        std::copy(x_line.begin(), x_line.end(), d.begin());

        // 2.2 rest of the lines in reverse order
        for ( ++line_rend, ++line_rstart; line_rstart != plane.line_start_.rend();
              ++line_rend, ++line_rstart, ++last_line_rend)
        {
            x_line.resize(*line_rend - *line_rstart);
            x_line.assign(x_line.size(), VectorBlock{});
            // Compute x=L2^T d
            //std::copy(x_line.begin(), x_line.end(), d.begin() + *line_rstart );

            for( auto row_index = *line_rend; row_index < *last_line_rend; ++row_index)
            {
                auto col = (*a_)[row_index].begin();
                while ( col != (*a_)[row_index].end() && col.index() < *line_rstart)
                {
                    ++col;
                }

                for (; col != (*a_)[row_index].end() && col.index() < *line_rend; ++col)
                {
                    col->mmtv(d[row_index], x_line[col.index() - *line_rstart]);
                }
            }
            // d = T^{-T} x
            solve_line_transposed(x_line, *line_rstart, *line_rend);
            std::copy(x_line.begin(), x_line.end(),
                      d.begin() + *line_rstart - first_line_start);
        }
    }
    void mAddUTransposedD(const std::vector<VectorBlock>& d,
                          int start_previous,
                          std::size_t start, std::size_t end)
    {
        VectorBlock zero;
        zero = 0.0;
        std::vector<VectorBlock> t(end-start, zero);
        for ( auto row_index = start_previous; row_index < start; ++row_index)
        {
            const auto& row = (*a_)[row_index];
            for ( auto col = row.beforeEnd(); col != row.beforeBegin() && col.index() >= start; --col)
            {
                if ( col.index() >= end )
                {
                    continue;
                }
                col->umv(d[row_index - start_previous], t[col.index() - start]);
            }
        }
        auto row_index = start;
        for ( auto&& block : t)
        {
            auto& mblock = (*inverseM_[row_index].begin());
            for ( int i = 0; i < MatrixBlock::rows; ++i)
            {
                mblock[i][i] -= beta_ * block[i];
            }
            ++row_index;
        }
    }
    std::vector<VectorBlock> computeLTransposed(std::size_t start_previous, std::size_t start,
                                                std::size_t end)
    {
        VectorBlock zero;
        zero = 0.0;
        std::vector<VectorBlock> d(start - start_previous, zero);
        for( auto row_index = start; row_index < end; ++row_index )
        {
            const auto& row = (*a_)[row_index];
            for ( auto col = row.begin(); col != row.end() && col.index() < start; ++col)
            {
                if ( col.index() < start_previous )
                {
                    continue;
                }
                auto dindex = col.index() - start_previous;
                for ( int i = 0; i < MatrixBlock::rows; ++i)
                {
                    for ( int j = 0; j < MatrixBlock::cols; ++j)
                    {
                        d[dindex][i] += (*col)[i][j];
                    }
                }
            }
        }
        return d;
    }

    using Plane = Detail::NFGeometryInformation::Plane;

    void updateFromLines(std::size_t plane_index, const Plane& plane)
    {
        using size_type = decltype(plane.line_start_.size());
        const auto& line_start = plane.line_start_;
        auto last_line = line_start.size()-1;
        for ( size_type line = 0; line < last_line; ++line )
        {
            if ( line > 0 && beta_ != 0 )
            {
                // Solve T^T_{line_index-1} x = (L_2^{line_index-1])^T.1
                // 1. d = (L_2^{line_index-1])^T.1
                auto start = line_start[line];
                auto end   = line_start[line+1];
                auto start_previous = line_start[line-1];
                auto d = computeLTransposed(start_previous, start, end);
                assert(d.size() == std::size_t(start-start_previous));
                // 2. Solve T^T_{line_index-1} x = d
                //    d=x
                solve_line_transposed(d, start, end);
                // M_i -= \beta U_2^{line_index-1}
                mAddUTransposedD(d, start_previous, start, end);
            }
            updateFromCells(plane.line_start_[line], plane.line_start_[line+1]);
        }
    }

    void solve_line_transposed(std::vector<VectorBlock>& vec, std::size_t start, std::size_t end)
    {
        // Solve T^Tx=b with T = (M + L_2) (I + M^{-1}U_1)
        assert( end - start == vec.size() );

        if ( vec.empty() )
        {
            return;
        }
        // 1. Solve (I + M^{-1}U_1)^T y = b
        std::vector<VectorBlock> y(vec.size());
        auto upper = inverseMU1_.begin()+start;
        y[0] = vec[0];
        const auto vend = end-start-1; //no upper for last line

        for (std::size_t i = 1; i < vend; ++i, ++upper )
        {
            y[i] = vec[i];
            auto col = upper->begin();
            if ( col != upper->end() )
            {
                assert(upper.index() == start + i - 1 &&
                       col.index() == start + i);
                col->mmv(vec[i-1], y[i]);
            }
        }
        // 2. Solve (M+L_1)^T x = y
        auto mrow = inverseM_.begin() + (end - 1);
        auto arow = (*a_).begin() + (end - 1);
        mrow->begin()->mv(y[vend], vec[vend]);

        for(auto i = vend - 1; i > 0; --i, --arow, --mrow)
        {
            // \todo stop neglecting NNCs
            auto j = i - 1;
            auto col = arow->find(start+j); // entry i, i+1 of transposed
            if ( col != arow->end() )
            {
                col->mmv(vec[i], y[j]);
            }
            mrow->begin()->mv(y[j], vec[j]);
        }
    }

    void updateFromCells(std::size_t start, std::size_t end)
    {
        auto mDiag    = inverseM_[start].begin();
        auto imU1Upper = inverseMU1_[start].begin();
        auto aDiag = (*a_)[start].find(start);
        assert(aDiag != (*a_)[start].end());
        assert(mDiag != inverseM_[start].end());
        *mDiag = *aDiag;
        if ( imU1Upper != inverseMU1_[start].end() )
        {
            ++aDiag;
            assert(aDiag != (*a_)[start].end() && aDiag.index() == start+1);
            assert(imU1Upper.index() == start+1);
            mDiag->invert();
            *imU1Upper = *mDiag;
            imU1Upper->rightmultiply(*aDiag);
        }
        for( start++ ; start < end; start++)
        {
            auto mDiag    = inverseM_[start].begin();
            auto aDiag = (*a_)[start].find(start);
            assert(aDiag != (*a_)[start].end());
            assert(mDiag != inverseM_[start].end());
            *mDiag = *aDiag;
            if ( aDiag == (*a_)[start].begin() )
            {
                // no direct lower neighbor
                continue;
            }
            --aDiag;
            if ( aDiag.index() != start - 1 )
            {
                // no direct lower neighbor
                continue;
            }
            auto tmp = *aDiag;
            tmp.rightmultiply(*imU1Upper);
            tmp *= alpha_;
            *mDiag -= tmp;
            ++(++aDiag); // first entry in upper part

            if( aDiag != (*a_)[start].end() && aDiag.index() == start + 1)
            {
                auto imU1Upper = inverseMU1_[start].begin();
                mDiag->invert();
                *imU1Upper = *mDiag;
                imU1Upper->rightmultiply(*aDiag);
            }
        }
    }
    Detail::NFGeometryInformation nf_geometry_;
    Dune::BDMatrix<MatrixBlock> inverseM_;
    Matrix inverseMU1_;
    const Matrix* a_;
    double alpha_;
    double beta_;
};
}
#endif
