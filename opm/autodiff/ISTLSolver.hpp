/*
  Copyright 2016 IRIS AS

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

#ifndef OPM_ISTLSOLVER_HEADER_INCLUDED
#define OPM_ISTLSOLVER_HEADER_INCLUDED

#include <opm/autodiff/AdditionalObjectDeleter.hpp>
#include <opm/autodiff/CPRPreconditioner.hpp>
#include <opm/autodiff/NewtonIterationBlackoilInterleaved.hpp>
#include <opm/autodiff/NewtonIterationUtilities.hpp>
#include <opm/autodiff/ParallelRestrictedAdditiveSchwarz.hpp>
#include <opm/autodiff/ParallelOverlappingILU0.hpp>
#include <opm/autodiff/AutoDiffHelpers.hpp>

#include <opm/common/Exceptions.hpp>
#include <opm/core/linalg/ParallelIstlInformation.hpp>
#include <opm/common/utility/platform_dependent/disable_warnings.h>

#include <dune/istl/scalarproducts.hh>
#include <dune/istl/operators.hh>
#include <dune/istl/preconditioners.hh>
#include <dune/istl/solvers.hh>
#include <dune/istl/owneroverlapcopy.hh>
#include <dune/istl/paamg/amg.hh>
#include <dune/istl/paamg/properties.hh>
#include <dune/istl/paamg/pinfo.hh>

#include <opm/common/utility/platform_dependent/reenable_warnings.h>

#include <type_traits>

namespace Dune
{

namespace ISTLUtility {

//! invert matrix by calling FMatrixHelp::invert
template <typename K>
static inline void invertMatrix (FieldMatrix<K,1,1> &matrix)
{
    FieldMatrix<K,1,1> A ( matrix );
    FMatrixHelp::invertMatrix(A, matrix );
}

//! invert matrix by calling FMatrixHelp::invert
template <typename K>
static inline void invertMatrix (FieldMatrix<K,2,2> &matrix)
{
    FieldMatrix<K,2,2> A ( matrix );
    FMatrixHelp::invertMatrix(A, matrix );
}

//! invert matrix by calling FMatrixHelp::invert
template <typename K>
static inline void invertMatrix (FieldMatrix<K,3,3> &matrix)
{
    FieldMatrix<K,3,3> A ( matrix );
    FMatrixHelp::invertMatrix(A, matrix );
}

//! invert matrix by calling matrix.invert
template <typename K, int n>
static inline void invertMatrix (FieldMatrix<K,n,n> &matrix)
{
    matrix.invert();
}

} // end ISTLUtility

template <class Scalar, int n, int m>
class MatrixBlock : public Dune::FieldMatrix<Scalar, n, m>
{
public:
    typedef Dune::FieldMatrix<Scalar, n, m>  BaseType;

    using BaseType :: operator= ;
    using BaseType :: rows;
    using BaseType :: cols;
    explicit MatrixBlock( const Scalar scalar = 0 ) : BaseType( scalar ) {}
    void invert()
    {
        ISTLUtility::invertMatrix( *this );
    }
    const BaseType& asBase() const { return static_cast< const BaseType& > (*this); }
    BaseType& asBase() { return static_cast< BaseType& > (*this); }
};

template<class K, int n, int m>
void
print_row (std::ostream& s, const MatrixBlock<K,n,m>& A,
           typename FieldMatrix<K,n,m>::size_type I,
           typename FieldMatrix<K,n,m>::size_type J,
           typename FieldMatrix<K,n,m>::size_type therow, int width,
           int precision)
{
    print_row(s, A.asBase(), I, J, therow, width, precision);
}

template<class K, int n, int m>
K& firstmatrixelement (MatrixBlock<K,n,m>& A)
{
   return firstmatrixelement( A.asBase() );
}



template<typename Scalar, int n, int m>
struct MatrixDimension< MatrixBlock< Scalar, n, m > >
: public MatrixDimension< typename MatrixBlock< Scalar, n, m >::BaseType >
{
};


#if HAVE_UMFPACK

/// \brief UMFPack specialization for MatrixBlock to make AMG happy
///
/// Without this the empty default implementation would be used.
template<typename T, typename A, int n, int m>
class UMFPack<BCRSMatrix<MatrixBlock<T,n,m>, A> >
    : public UMFPack<BCRSMatrix<FieldMatrix<T,n,m>, A> >
{
    typedef UMFPack<BCRSMatrix<FieldMatrix<T,n,m>, A> > Base;
    typedef BCRSMatrix<FieldMatrix<T,n,m>, A> Matrix;

public:
    typedef BCRSMatrix<MatrixBlock<T,n,m>, A> RealMatrix;

    UMFPack(const RealMatrix& matrix, int verbose, bool)
        : Base(reinterpret_cast<const Matrix&>(matrix), verbose)
    {}
};
#endif

#if HAVE_SUPERLU

/// \brief SuperLU specialization for MatrixBlock to make AMG happy
///
/// Without this the empty default implementation would be used.
template<typename T, typename A, int n, int m>
class SuperLU<BCRSMatrix<MatrixBlock<T,n,m>, A> >
    : public SuperLU<BCRSMatrix<FieldMatrix<T,n,m>, A> >
{
    typedef SuperLU<BCRSMatrix<FieldMatrix<T,n,m>, A> > Base;
    typedef BCRSMatrix<FieldMatrix<T,n,m>, A> Matrix;

public:
    typedef BCRSMatrix<MatrixBlock<T,n,m>, A> RealMatrix;

    SuperLU(const RealMatrix& matrix, int verbose, bool reuse=true)
        : Base(reinterpret_cast<const Matrix&>(matrix), verbose, reuse)
    {}
};
#endif


} // end namespace Dune

namespace Opm
{
    /// This class solves the fully implicit black-oil system by
    /// solving the reduced system (after eliminating well variables)
    /// as a block-structured matrix (one block for all cell variables) for a fixed
    /// number of cell variables np .
    template < class MatrixBlockType, class VectorBlockType >
    class ISTLSolver : public NewtonIterationBlackoilInterface
    {
        typedef typename MatrixBlockType :: field_type  Scalar;

        typedef Dune::BCRSMatrix <MatrixBlockType>      Matrix;
        typedef Dune::BlockVector<VectorBlockType>      Vector;

    public:
        typedef Dune::AssembledLinearOperator< Matrix, Vector, Vector > AssembledLinearOperatorType;

        typedef NewtonIterationBlackoilInterface :: SolutionVector  SolutionVector;
        /// Construct a system solver.
        /// \param[in] param   parameters controlling the behaviour of the linear solvers
        /// \param[in] parallelInformation In the case of a parallel run
        ///                                with dune-istl the information about the parallelization.
        ISTLSolver(const NewtonIterationBlackoilInterleavedParameters& param,
                   const boost::any& parallelInformation_arg=boost::any())
        : iterations_( 0 ),
          parallelInformation_(parallelInformation_arg),
          isIORank_(isIORank(parallelInformation_arg)),
          parameters_( param )
        {
        }

        /// Construct a system solver.
        /// \param[in] param   ParameterGroup controlling the behaviour of the linear solvers
        /// \param[in] parallelInformation In the case of a parallel run
        ///                                with dune-istl the information about the parallelization.
        ISTLSolver(const ParameterGroup& param,
                   const boost::any& parallelInformation_arg=boost::any())
        : iterations_( 0 ),
          parallelInformation_(parallelInformation_arg),
          isIORank_(isIORank(parallelInformation_arg)),
          parameters_( param )
        {
        }

        // dummy method that is not implemented for this class
        SolutionVector computeNewtonIncrement(const LinearisedBlackoilResidual&) const
        {
            OPM_THROW(std::logic_error,"This method is not implemented");
            return SolutionVector();
        }

        /// Solve the system of linear equations Ax = b, with A being the
        /// combined derivative matrix of the residual and b
        /// being the residual itself.
        /// \param[in] residual   residual object containing A and b.
        /// \return               the solution x

        /// \copydoc NewtonIterationBlackoilInterface::iterations
        int iterations () const { return iterations_; }

        /// \copydoc NewtonIterationBlackoilInterface::parallelInformation
        const boost::any& parallelInformation() const { return parallelInformation_; }

    public:
        /// \brief construct the CPR preconditioner and the solver.
        /// \tparam P The type of the parallel information.
        /// \param parallelInformation the information about the parallelization.
        template<int category=Dune::SolverCategory::sequential, class LinearOperator, class POrComm>
        void constructPreconditionerAndSolve(LinearOperator& linearOperator,
                                             Vector& x, Vector& istlb,
                                             const POrComm& parallelInformation_arg,
                                             Dune::InverseOperatorResult& result) const
        {
            // Construct scalar product.
            typedef Dune::ScalarProductChooser<Vector, POrComm, category> ScalarProductChooser;
            typedef std::unique_ptr<typename ScalarProductChooser::ScalarProduct> SPPointer;
            SPPointer sp(ScalarProductChooser::construct(parallelInformation_arg));

            // Communicate if parallel.
            parallelInformation_arg.copyOwnerToAll(istlb, istlb);

#if ! HAVE_UMFPACK
            if( parameters_.linear_solver_use_amg_ )
            {
                typedef ISTLUtility::CPRSelector< Matrix, Vector, Vector, POrComm>  CPRSelectorType;
                typedef typename CPRSelectorType::AMG AMG;
                typedef typename CPRSelectorType::Operator MatrixOperator;

                std::unique_ptr< AMG > amg;
                std::unique_ptr< MatrixOperator > opA;

                if( ! std::is_same< LinearOperator, MatrixOperator > :: value )
                {
                    // create new operator in case linear operator and matrix operator differ
                    opA.reset( CPRSelectorType::makeOperator( linearOperator.getmat(), parallelInformation_arg ) );
                }

                const double relax = 1.0;

                // Construct preconditioner.
                constructAMGPrecond( linearOperator, parallelInformation_arg, amg, opA, relax );

                // Solve.
                solve(linearOperator, x, istlb, *sp, *amg, result);
            }
            else
#endif
            {
                maybeSolveOnOne(linearOperator, istlb, x, parallelInformation_arg,
                                *sp, result);
            }
        }

        typedef Dune::SeqILU0<Matrix, Vector, Vector> SeqPreconditioner;

        template <class Operator>
        std::unique_ptr<SeqPreconditioner> constructPrecond(Operator& opA, const Dune::Amg::SequentialInformation&) const
        {
            const double relax = 0.9;
            std::unique_ptr<SeqPreconditioner> precond(new SeqPreconditioner(opA.getmat(), relax));
            return precond;
        }

#if HAVE_MPI
        typedef Dune::OwnerOverlapCopyCommunication<int, int> Comm;
        typedef ParallelOverlappingILU0<Matrix,Vector,Vector,Comm> ParPreconditioner;
        template <class Operator>
        std::unique_ptr<ParPreconditioner>
        constructPrecond(Operator& opA, const Comm& comm) const
        {
            typedef std::unique_ptr<ParPreconditioner> Pointer;
            const double relax = 0.9;
            return Pointer(new ParPreconditioner(opA.getmat(), comm, relax));
        }
#endif

        template <class LinearOperator, class MatrixOperator, class POrComm, class AMG >
        void
        constructAMGPrecond(LinearOperator& /* linearOperator */, const POrComm& comm, std::unique_ptr< AMG >& amg, std::unique_ptr< MatrixOperator >& opA, const double relax ) const
        {
            ISTLUtility::createAMGPreconditionerPointer( *opA, relax, comm, amg );
        }


        template <class MatrixOperator, class POrComm, class AMG >
        void
        constructAMGPrecond(MatrixOperator& opA, const POrComm& comm, std::unique_ptr< AMG >& amg, std::unique_ptr< MatrixOperator >&, const double relax ) const
        {
            ISTLUtility::createAMGPreconditionerPointer( opA, relax, comm, amg );
        }

        /// \brief Solve the system using the given preconditioner and scalar product.
        template <class Operator, class ScalarProd, class Precond>
        void solve(Operator& opA, Vector& x, Vector& istlb, ScalarProd& sp, Precond& precond, Dune::InverseOperatorResult& result) const
        {
            // TODO: Revise when linear solvers interface opm-core is done
            // Construct linear solver.
            // GMRes solver
            int verbosity = ( isIORank_ ) ? parameters_.linear_solver_verbosity_ : 0;

            if ( parameters_.newton_use_gmres_ ) {
                Dune::RestartedGMResSolver<Vector> linsolve(opA, sp, precond,
                          parameters_.linear_solver_reduction_,
                          parameters_.linear_solver_restart_,
                          parameters_.linear_solver_maxiter_,
                          verbosity);
                // Solve system.
                linsolve.apply(x, istlb, result);
            }
            else { // BiCGstab solver
                Dune::BiCGSTABSolver<Vector> linsolve(opA, sp, precond,
                          parameters_.linear_solver_reduction_,
                          parameters_.linear_solver_maxiter_,
                          verbosity);
                // Solve system.
                linsolve.apply(x, istlb, result);
            }
        }


        /// Solve the linear system Ax = b, with A being the
        /// combined derivative matrix of the residual and b
        /// being the residual itself.
        /// \param[in] A   matrix A
        /// \param[inout] x  solution to be computed x
        /// \param[in] b   right hand side b
        template<int n>
        void solve(Matrix& A, Dune::BlockVector<Dune::FieldVector<double,n> >& x, Vector& b ) const
        {
            // Parallel version is deactivated until we figure out how to do it properly.
#if HAVE_MPI
            if (parallelInformation_.type() == typeid(ParallelISTLInformation))
            {
                if(parameters_.linear_solver_sequential_)
                {
                    const ParallelISTLInformation& info =
                        boost::any_cast<const ParallelISTLInformation&>( parallelInformation_);
                    Comm istlComm(info.communicator());
                    Matrix& fullA = A;
                    /// Redistribute system to one process and solve
                    typedef Dune::Amg::MatrixGraph<const Matrix> MatrixGraph;
                    typedef Dune::Amg::PropertiesGraph<MatrixGraph,
                                                       Dune::Amg::VertexProperties,
                                                       Dune::Amg::EdgeProperties> PropertiesGraph;
                    MatrixGraph       graph(fullA);
                    PropertiesGraph pgraph(graph);
                    // Matrix with the whole system on one process
                    Matrix wholefullA;
                    Comm* newComm;
                    Dune::RedistributeInformation<Comm> redist;
                    bool existentOnRedist=Dune::graphRepartition(graph, istlComm, 1,
                                                                 newComm, redist.getInterface(),
                                                                 false);
                    Dune::redistributeMatrix(const_cast<Matrix&>(fullA), wholefullA, istlComm, *newComm, redist);

                    Vector wholeIstlB(wholefullA.N());
                    Vector wholeIstlX(wholefullA.N());
                    wholeIstlX = 0;
                    redist.redistribute(b, wholeIstlB);
                    int converged = 1;

                    Dune::InverseOperatorResult result;
                    if ( existentOnRedist )
                    {
                        Dune::MatrixAdapter<Matrix,Vector,Vector> adapter(wholefullA), fulladapter(wholefullA);
                        Dune::Amg::SequentialInformation seqcomm;
                        auto precond = constructPrecond(adapter, seqcomm);
                        Dune::SeqScalarProduct<Vector> ssp;
                        //solve(fulladapter, wholeIstlX, wholeIstlB, ssp, *precond, result);
                        Dune::SuperLU<Matrix> solver(wholefullA, false);
                        solver.apply(wholeIstlX, wholeIstlB, result);
                        converged = result.converged? 1: 0;
                    }
                    if(  info.communicator().min(converged) == 0 )
                        result.converged=false;
                    else
                        result.converged=true;
                    redist.redistributeBackward(x, wholeIstlX);
                    info.copyOwnerToAll(x,x);
                    checkConvergence(result);
                }
            else
            {
                typedef Dune::OwnerOverlapCopyCommunication<int,int> Comm;
                const ParallelISTLInformation& info =
                    boost::any_cast<const ParallelISTLInformation&>( parallelInformation_);
                Comm istlComm(info.communicator());

                // Construct operator, scalar product and vectors needed.
                typedef Dune::OverlappingSchwarzOperator<Matrix, Vector, Vector,Comm> Operator;
                Operator opA(A, istlComm);
                solve( opA, x, b, istlComm  );
            }
            }
            else
#endif
            {
                if(parameters_.linear_solver_sequential_)
                {
                    Dune::InverseOperatorResult result;
                    Dune::SuperLU<Matrix> solver(A, false);
                    solver.apply(x, b, result);
                    checkConvergence(result);
                }else
                {
                // Construct operator, scalar product and vectors needed.
                Dune::MatrixAdapter< Matrix, Vector, Vector> opA( A );
                solve( opA, x, b );
                }
            }
        }

                void solve(Matrix& A, Vector& x, Vector& b ) const
        {
            // Parallel version is deactivated until we figure out how to do it properly.
#if HAVE_MPI
            if (parallelInformation_.type() == typeid(ParallelISTLInformation))
            {
                typedef Dune::OwnerOverlapCopyCommunication<int,int> Comm;
                const ParallelISTLInformation& info =
                    boost::any_cast<const ParallelISTLInformation&>( parallelInformation_);
                Comm istlComm(info.communicator());

                // Construct operator, scalar product and vectors needed.
                typedef Dune::OverlappingSchwarzOperator<Matrix, Vector, Vector,Comm> Operator;
                Operator opA(A, istlComm);
                solve( opA, x, b, istlComm  );
            }
            else
#endif
            {
                // Construct operator, scalar product and vectors needed.
                Dune::MatrixAdapter< Matrix, Vector, Vector> opA( A );
                solve( opA, x, b );
            }
        }

        /// Solve the linear system Ax = b, with A being the
        /// combined derivative matrix of the residual and b
        /// being the residual itself.
        /// \param[in] A   matrix A
        /// \param[inout] x  solution to be computed x
        /// \param[in] b   right hand side b
        template <class Operator, class Comm >
        void solve(Operator& opA, Vector& x, Vector& b, Comm& comm) const
        {
            Dune::InverseOperatorResult result;
            // Parallel version is deactivated until we figure out how to do it properly.
#if HAVE_MPI
            if (parallelInformation_.type() == typeid(ParallelISTLInformation))
            {
                const size_t size = opA.getmat().N();
                const ParallelISTLInformation& info =
                    boost::any_cast<const ParallelISTLInformation&>( parallelInformation_);

                // As we use a dune-istl with block size np the number of components
                // per parallel is only one.
                info.copyValuesTo(comm.indexSet(), comm.remoteIndices(),
                                  size, 1);
                // Construct operator, scalar product and vectors needed.
                constructPreconditionerAndSolve<Dune::SolverCategory::overlapping>(opA, x, b, comm, result);
            }
            else
#endif
            {
                OPM_THROW(std::logic_error,"this method if for parallel solve only");
            }

            checkConvergence( result );
        }

        /// Solve the linear system Ax = b, with A being the
        /// combined derivative matrix of the residual and b
        /// being the residual itself.
        /// \param[in] A   matrix A
        /// \param[inout] x  solution to be computed x
        /// \param[in] b   right hand side b
        template <class Operator>
        void solve(Operator& opA, Vector& x, Vector& b ) const
        {
            Dune::InverseOperatorResult result;
            // Construct operator, scalar product and vectors needed.
            Dune::Amg::SequentialInformation info;
            constructPreconditionerAndSolve(opA, x, b, info, result);
            checkConvergence( result );
        }

        void checkConvergence( const Dune::InverseOperatorResult& result ) const
        {
            // store number of iterations
            iterations_ = result.iterations;

            // Check for failure of linear solver.
            if (!parameters_.ignoreConvergenceFailure_ && !result.converged) {
                const std::string msg("Convergence failure for linear solver.");
                if (isIORank_) {
                    OpmLog::problem(msg);
                }
                OPM_THROW_NOLOG(LinearSolverProblem, msg);
            }
        }

        template<class OP, class P, class SP>
        typename OP::BaseType* maybeSolveOnOne(OP& linearOperator, Vector& istlb,
                             Vector& x,
                             const P& parallelInformation_arg, SP& sp,
                             Dune::InverseOperatorResult& result) const
        {
            if(parameters_.linear_solver_sequential_)
            {
                Matrix fullA;
                linearOperator.getfullmat(fullA);
                /// Redistribute system to one process and solve
                typedef Dune::Amg::MatrixGraph<const Matrix> MatrixGraph;
                typedef Dune::Amg::PropertiesGraph<MatrixGraph,
                                                   Dune::Amg::VertexProperties,
                                                   Dune::Amg::EdgeProperties> PropertiesGraph;
                MatrixGraph       graph(fullA);
                PropertiesGraph pgraph(graph);
                // Matrix with the whole system on one process
                Matrix wholeA;
                Matrix wholefullA;
                typename OP::communication_type* newComm;
                auto& istlComm = const_cast<typename OP::communication_type&>(parallelInformation_arg);
                Dune::RedistributeInformation<Comm> redist;
                bool existentOnRedist=Dune::graphRepartition(graph, istlComm, 1,
                                                             newComm, redist.getInterface(),
                                                             false);
                Dune::redistributeMatrix(const_cast<Matrix&>(linearOperator.getmat()), wholeA, istlComm, *newComm, redist);
                Dune::redistributeMatrix(fullA, wholefullA, istlComm, *newComm, redist);

                Vector wholeIstlB(wholeA.N());
                Vector wholeIstlX(wholeA.N());
                wholeIstlX = 0;
                redist.redistribute(istlb, wholeIstlB);
                int converged = 1;

                if ( existentOnRedist )
                {
                    Dune::MatrixAdapter<Matrix,Vector,Vector> adapter(wholeA), fulladapter(wholefullA);
                    Dune::Amg::SequentialInformation seqcomm;
                    auto precond = constructPrecond(adapter, seqcomm);
                    Dune::SeqScalarProduct<Vector> ssp;
                    //solve(fulladapter, wholeIstlX, wholeIstlB, ssp, *precond, result);
                    Dune::SuperLU<Matrix> solver(wholefullA, false);
                    solver.apply(wholeIstlX, wholeIstlB, result);
                    converged = result.converged? 1: 0;
                }
                if(  parallelInformation_arg.communicator().min(converged) == 0 )
                    result.converged=false;
                else
                    result.converged=true;
                redist.redistributeBackward(x, wholeIstlX);
                parallelInformation_arg.copyOwnerToAll(x,x);
            }else
            {
                // Construct preconditioner.
                auto precond = constructPrecond(linearOperator, parallelInformation_arg);

                // Solve.
                solve(linearOperator, x, istlb, sp, *precond, result);
            }
            return nullptr;
        }

        template<class OP, class P, class SP, int n>
        void maybeSolveOnOne(OP& linearOperator, Dune::BlockVector<Dune::FieldVector<float,n>>& istlb, Vector& x,
                             const P& parallelInformation_arg, SP& sp,
                             Dune::InverseOperatorResult& result) const
        {
            // Construct preconditioner.
            auto precond = constructPrecond(linearOperator, parallelInformation_arg);
            // Solve.
            solve(linearOperator, x, istlb, sp, *precond, result);
        }

        template<class OP, class P, class SP>
        void maybeSolveOnOne(OP& linearOperator, Vector& istlb, Vector& x,
                             const P& parallelInformation_arg, SP& sp,
                             Dune::InverseOperatorResult& result,
                             typename std::enable_if<
                                 std::is_same<OP,Dune::OverlappingSchwarzOperator<Matrix, Vector, Vector,P>>::value &&
                                 // SuperLU only works for double on old DUNE versions
                                 std::is_same<typename Vector::field_type,double>::value,
                                 void
                             >::type* d = nullptr) const
        {
            ++d;

            if(parameters_.linear_solver_sequential_)
            {
                const Matrix& fullA = linearOperator.getmat();
                /// Redistribute system to one process and solve
                typedef Dune::Amg::MatrixGraph<const Matrix> MatrixGraph;
                typedef Dune::Amg::PropertiesGraph<MatrixGraph,
                                                   Dune::Amg::VertexProperties,
                                                   Dune::Amg::EdgeProperties> PropertiesGraph;
                MatrixGraph       graph(fullA);
                PropertiesGraph pgraph(graph);
                // Matrix with the whole system on one process
                Matrix wholefullA;
                typename OP::communication_type* newComm;
                auto& istlComm = const_cast<typename OP::communication_type&>(parallelInformation_arg);
                Dune::RedistributeInformation<Comm> redist;
                bool existentOnRedist=Dune::graphRepartition(graph, istlComm, 1,
                                                             newComm, redist.getInterface(),
                                                             false);
                Dune::redistributeMatrix(const_cast<Matrix&>(fullA), wholefullA, istlComm, *newComm, redist);

                Vector wholeIstlB(wholefullA.N());
                Vector wholeIstlX(wholefullA.N());
                wholeIstlX = 0;
                redist.redistribute(istlb, wholeIstlB);
                int converged = 1;

                if ( existentOnRedist )
                {
                    Dune::MatrixAdapter<Matrix,Vector,Vector> adapter(wholefullA), fulladapter(wholefullA);
                    Dune::Amg::SequentialInformation seqcomm;
                    auto precond = constructPrecond(adapter, seqcomm);
                    Dune::SeqScalarProduct<Vector> ssp;
                    //solve(fulladapter, wholeIstlX, wholeIstlB, ssp, *precond, result);            }
                    Dune::SuperLU<Matrix> solver(wholefullA, false);
                    solver.apply(wholeIstlX, wholeIstlB, result);
                    converged = result.converged? 1: 0;
                }
                if(  parallelInformation_arg.communicator().min(converged) == 0 )
                    result.converged=false;
                else
                    result.converged=true;
                redist.redistributeBackward(x, wholeIstlX);
                 parallelInformation_arg.copyOwnerToAll(x,x);
            }
            else
            {
                // Construct preconditioner.
                auto precond = constructPrecond(linearOperator, parallelInformation_arg);
                // Solve.
                solve(linearOperator, x, istlb, sp, *precond, result);
            }
        }

        template<class OP,  class SP>
        typename OP::BaseType* maybeSolveOnOne(OP& linearOperator, Vector& istlb, Vector& x,
                             const Dune::Amg::SequentialInformation& parallelInformation_arg, SP& sp,
                             Dune::InverseOperatorResult& result) const
        {
            if(parameters_.linear_solver_sequential_)
            {
                Matrix fullA;
                linearOperator.getfullmat(fullA);

                // Solve.
                Dune::SuperLU<Matrix> solver(fullA);
                solver.apply(x, istlb, result);
            }
            else
            {
                // Construct preconditioner.
                auto precond = constructPrecond(linearOperator, parallelInformation_arg);
                solve(linearOperator, x, istlb, sp, *precond, result);
            }
            return nullptr;
        }

        template<class SP, int n>
        void maybeSolveOnOne(Dune::MatrixAdapter<Matrix,Vector,Vector>& linearOperator, Vector& istlb, Dune::BlockVector<Dune::FieldVector<double,n>>& x,
                             const Dune::Amg::SequentialInformation& parallelInformation_arg, SP& sp,
                             Dune::InverseOperatorResult& result) const
        {
            if(parameters_.linear_solver_sequential_)
            {
                std::cout<<"superlu"<<std::endl;
                Dune::SuperLU<Matrix> solver(linearOperator.getmat(), false);
                solver.apply(x, istlb, result);
            }else
            {
                // Construct preconditioner.
                auto precond = constructPrecond(linearOperator, parallelInformation_arg);

                // Solve.
                solve(linearOperator, x, istlb, sp, *precond, result);
            }
        }

    protected:
        mutable int iterations_;
        boost::any parallelInformation_;
        bool isIORank_;

        NewtonIterationBlackoilInterleavedParameters parameters_;
    }; // end ISTLSolver

} // namespace Opm
#endif
