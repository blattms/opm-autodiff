/*
  Copyright 2017 Dr. Blatt - HPC-Simulation-Software & Services

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
#ifndef OPM_AMG_HEADER_INCLUDED
#define OPM_AMG_HEADER_INCLUDED

#include <opm/autodiff/ParallelOverlappingILU0.hpp>
#include <dune/istl/paamg/twolevelmethod.hh>
#include <dune/istl/paamg/aggregates.hh>
#include <dune/istl/bvector.hh>
#include <dune/istl/bcrsmatrix.hh>
#include <dune/istl/preconditioners.hh>
#include <dune/istl/schwarz.hh>
#include <dune/istl/scalarproducts.hh>
#include <dune/common/fvector.hh>
#include <dune/common/fmatrix.hh>
namespace Dune
{
namespace Amg
{
template<class M, class Norm>
class UnSymmetricCriterion;
}
}

namespace Opm
{

template <class Scalar, int n, int m>
class MatrixBlock;

namespace Detail
{

template<typename NonScalarType>
struct ScalarType
{
};

template<typename FieldType, int SIZE>
struct ScalarType<Dune::FieldVector<FieldType, SIZE> >
{
    typedef Dune::FieldVector<FieldType, 1> value;
};

template<typename FieldType, int ROWS, int COLS>
struct ScalarType<Dune::FieldMatrix<FieldType, ROWS, COLS> >
{
    typedef Dune::FieldMatrix<FieldType, 1, 1> value;
};

template<typename FieldType, int ROWS, int COLS>
struct ScalarType<MatrixBlock<FieldType, ROWS, COLS> >
{
    typedef MatrixBlock<FieldType, 1, 1> value;
};

template<typename BlockType, typename Allocator>
struct ScalarType<Dune::BCRSMatrix<BlockType, Allocator> >
{
    using ScalarBlock = typename ScalarType<BlockType>::value;
    using ScalarAllocator = typename Allocator::template rebind<ScalarBlock>::other;
    typedef Dune::BCRSMatrix<ScalarBlock,ScalarAllocator> value;
};

template<typename BlockType, typename Allocator>
struct ScalarType<Dune::BlockVector<BlockType, Allocator> >
{
    using ScalarBlock = typename ScalarType<BlockType>::value;
    using ScalarAllocator = typename Allocator::template rebind<ScalarBlock>::other;
    typedef Dune::BlockVector<ScalarBlock,ScalarAllocator> value;
};

template<typename X>
struct ScalarType<Dune::SeqScalarProduct<X> >
{
    typedef Dune::SeqScalarProduct<typename ScalarType<X>::value> value;
};

#define ComposeScalarTypeForSeqPrecond(PREC)            \
    template<typename M, typename X, typename Y, int l> \
    struct ScalarType<PREC<M,X,Y,l> >                   \
    {                                                   \
        typedef PREC<typename ScalarType<M>::value,     \
                     typename ScalarType<X>::value,     \
                     typename ScalarType<Y>::value,     \
                     l> value;                          \
    };

ComposeScalarTypeForSeqPrecond(Dune::SeqJac);
ComposeScalarTypeForSeqPrecond(Dune::SeqSOR);
ComposeScalarTypeForSeqPrecond(Dune::SeqSSOR);
ComposeScalarTypeForSeqPrecond(Dune::SeqGS);
ComposeScalarTypeForSeqPrecond(Dune::SeqILU0);
ComposeScalarTypeForSeqPrecond(Dune::SeqILUn);

#undef ComposeScalarTypeForSeqPrecond

template<typename X, typename Y>
struct ScalarType<Dune::Richardson<X,Y> >
{
    typedef Dune::Richardson<typename ScalarType<X>::value,
                             typename ScalarType<Y>::value>  value;
};

template<class M, class X, class Y, class C>
struct ScalarType<Dune::OverlappingSchwarzOperator<M,X,Y,C> >
{
    typedef Dune::OverlappingSchwarzOperator<typename ScalarType<M>::value,
                                             typename ScalarType<X>::value,
                                             typename ScalarType<Y>::value,
                                             C> value;
};

template<class M, class X, class Y>
struct ScalarType<Dune::MatrixAdapter<M,X,Y> >
{
    typedef Dune::MatrixAdapter<typename ScalarType<M>::value,
                                typename ScalarType<X>::value,
                                typename ScalarType<Y>::value> value;
};

template<class X, class C>
struct ScalarType<Dune::OverlappingSchwarzScalarProduct<X,C> >
{
    typedef Dune::OverlappingSchwarzScalarProduct<typename ScalarType<X>::value,
                                                  C> value;
};

template<class X, class C>
struct ScalarType<Dune::NonoverlappingSchwarzScalarProduct<X,C> >
{
    typedef Dune::NonoverlappingSchwarzScalarProduct<typename ScalarType<X>::value,
                                                     C> value;
};

template<class X, class Y, class C, class T>
struct ScalarType<Dune::BlockPreconditioner<X,Y,C,T> >
{
    typedef Dune::BlockPreconditioner<typename ScalarType<X>::value,
                                      typename ScalarType<Y>::value,
                                      C,
                                      typename ScalarType<T>::value> value;
};

template<class M, class X, class Y, class C>
struct ScalarType<ParallelOverlappingILU0<M,X,Y,C> >
{
    typedef ParallelOverlappingILU0<typename ScalarType<M>::value,
                                    typename ScalarType<X>::value,
                                    typename ScalarType<Y>::value,
                                    C> value;
};

template<class B, class N>
struct ScalarType<Dune::Amg::CoarsenCriterion<Dune::Amg::SymmetricCriterion<Dune::BCRSMatrix<B>,N> > >
{
    using value = Dune::Amg::CoarsenCriterion<Dune::Amg::SymmetricCriterion<Dune::BCRSMatrix<typename ScalarType<B>::value>, Dune::Amg::FirstDiagonal> >;
};

template<class B, class N>
struct ScalarType<Dune::Amg::CoarsenCriterion<Dune::Amg::UnSymmetricCriterion<Dune::BCRSMatrix<B>,N> > >
{
    using value = Dune::Amg::CoarsenCriterion<Dune::Amg::UnSymmetricCriterion<Dune::BCRSMatrix<typename ScalarType<B>::value>, Dune::Amg::FirstDiagonal> >;
};

template<class C, std::size_t COMPONENT_INDEX>
struct OneComponentCriterionType
{};

template<class B, class N, std::size_t COMPONENT_INDEX>
struct OneComponentCriterionType<Dune::Amg::CoarsenCriterion<Dune::Amg::SymmetricCriterion<Dune::BCRSMatrix<B>,N> >,COMPONENT_INDEX>
{
    using value = Dune::Amg::CoarsenCriterion<Dune::Amg::SymmetricCriterion<Dune::BCRSMatrix<B>, Dune::Amg::Diagonal<COMPONENT_INDEX> > >;
};

template<class B, class N, std::size_t COMPONENT_INDEX>
struct OneComponentCriterionType<Dune::Amg::CoarsenCriterion<Dune::Amg::UnSymmetricCriterion<Dune::BCRSMatrix<B>,N> >,COMPONENT_INDEX>
{
    using value = Dune::Amg::CoarsenCriterion<Dune::Amg::UnSymmetricCriterion<Dune::BCRSMatrix<B>, Dune::Amg::Diagonal<COMPONENT_INDEX> > >;
};

template<class Operator, class Criterion, class Communication, std::size_t COMPONENT_INDEX>
class OneComponentAggregationLevelTransferPolicy;


/**
 * @brief A policy class for solving the coarse level system using one step of AMG.
 * @tparam O The type of the linear operator used.
 * @tparam S The type of the smoother used in AMG.
 * @tparam C The type of the crition used for the aggregation within AMG.
 * @tparam C1 The type of the information about the communication. Either
 *            Dune::OwnerOverlapCopyCommunication or Dune::SequentialInformation.
 */
template<class O, class S, class C, class P>
class OneStepAMGCoarseSolverPolicy
{
public:
    typedef P LevelTransferPolicy;
    /** @brief The type of the linear operator used. */
    typedef O Operator;
    /** @brief The type of the range and domain of the operator. */
    typedef typename O::range_type X;
    /** @brief The type of the crition used for the aggregation within AMG.*/
    typedef C Criterion;
    /** @brief The type of the communication used for AMG.*/
    typedef typename P::ParallelInformation Communication;
    /** @brief The type of the smoother used in AMG. */
    typedef S Smoother;
    /** @brief The type of the arguments used for constructing the smoother. */
    typedef typename Dune::Amg::SmootherTraits<S>::Arguments SmootherArgs;
    /** @brief The type of the AMG construct on the coarse level.*/
    typedef Dune::Amg::AMG<Operator,X,Smoother,Communication> AMGType;
    /**
     * @brief Constructs the coarse solver policy.
     * @param args The arguments used for constructing the smoother.
     * @param c The crition used for the aggregation within AMG.
     */
    OneStepAMGCoarseSolverPolicy(const SmootherArgs& args, const Criterion& c)
        : smootherArgs_(args), criterion_(c)
    {}
    /** @brief Copy constructor. */
    OneStepAMGCoarseSolverPolicy(const OneStepAMGCoarseSolverPolicy& other)
        : coarseOperator_(other.coarseOperator_), smootherArgs_(other.smootherArgs_),
          criterion_(other.criterion_)
    {}
private:
    /**
     * @brief A wrapper that makes an inverse operator out of AMG.
     *
     * The operator will use one step of AMG to approximately solve
     * the coarse level system.
     */
    struct AMGInverseOperator : public Dune::InverseOperator<X,X>
    {
        AMGInverseOperator(const typename AMGType::Operator& op,
                           const Criterion& crit,
                           const typename AMGType::SmootherArgs& args,
                           const Communication& comm)
            : amg_(op, crit,args, comm), first_(true)
        {}

        void apply(X& x, X& b, double reduction, Dune::InverseOperatorResult& res)
        {
            DUNE_UNUSED_PARAMETER(reduction);
            DUNE_UNUSED_PARAMETER(res);
            if(first_)
            {
                amg_.pre(x,b);
                first_=false;
                x_=x;
            }
            amg_.apply(x,b);
        }

        void apply(X& x, X& b, Dune::InverseOperatorResult& res)
        {
            return apply(x,b,1e-8,res);
        }

        ~AMGInverseOperator()
        {
            if(!first_)
                amg_.post(x_);
        }
        AMGInverseOperator(const AMGInverseOperator& other)
            : x_(other.x_), amg_(other.amg_), first_(other.first_)
        {
        }
    private:
        X x_;
        AMGType amg_;
        bool first_;
    };

public:
    /** @brief The type of solver constructed for the coarse level. */
    typedef AMGInverseOperator CoarseLevelSolver;

    /**
     * @brief Constructs a coarse level solver.
     *
     * @param transferPolicy The policy describing the transfer between levels.
     * @return A pointer to the constructed coarse level solver.
     */
    template<class LTP>
    CoarseLevelSolver* createCoarseLevelSolver(LTP& transferPolicy)
    {
        coarseOperator_=transferPolicy.getCoarseLevelOperator();
        const LevelTransferPolicy& transfer =
            reinterpret_cast<const LevelTransferPolicy&>(transferPolicy);
        AMGInverseOperator* inv = new AMGInverseOperator(*coarseOperator_,
                                                         criterion_,
                                                         smootherArgs_,
                                                         transfer.getCoarseLevelCommunication());

        return inv; //std::shared_ptr<InverseOperator<X,X> >(inv);

    }

private:
    /** @brief The coarse level operator. */
    std::shared_ptr<Operator> coarseOperator_;
    /** @brief The arguments used to construct the smoother. */
    SmootherArgs smootherArgs_;
    /** @brief The coarsening criterion. */
    Criterion criterion_;
};

template<class Smoother, class Operator, class Communication>
Smoother* constructSmoother(const Operator& op,
                            const typename Dune::Amg::SmootherTraits<Smoother>::Arguments& smargs,
                            const Communication& comm)
{
    typename Dune::Amg::ConstructionTraits<Smoother>::Arguments args;
    args.setMatrix(op.getmat());
    args.setComm(comm);
    args.setArgs(smargs);
    return Dune::Amg::ConstructionTraits<Smoother>::construct(args);
}

template<class G, class C, class S>
const Dune::Amg::OverlapVertex<typename G::VertexDescriptor>*
buildOverlapVertices(const G& graph, const C& pinfo,
                     Dune::Amg::AggregatesMap<typename G::VertexDescriptor>& aggregates,
                     const S& overlap,
                     std::size_t& overlapCount)
{
    // count the overlap vertices.
    auto end = graph.end();
    overlapCount = 0;

    const auto& lookup=pinfo.globalLookup();

    for(auto vertex=graph.begin(); vertex != end; ++vertex) {
        const auto* pair = lookup.pair(*vertex);

        if(pair!=0 && overlap.contains(pair->local().attribute()))
            ++overlapCount;
    }
    // Allocate space
    using Vertex =  typename G::VertexDescriptor;
    using OverlapVertex = Dune::Amg::OverlapVertex<Vertex>;

    auto* overlapVertices = new OverlapVertex[overlapCount=0 ? 1 : overlapCount];
    if(overlapCount==0)
        return overlapVertices;

    // Initialize them
    overlapCount=0;
    for(auto vertex=graph.begin(); vertex != end; ++vertex) {
        const auto* pair = lookup.pair(*vertex);

        if(pair!=0 && overlap.contains(pair->local().attribute())) {
            overlapVertices[overlapCount].aggregate = &aggregates[pair->local()];
            overlapVertices[overlapCount].vertex = pair->local();
            ++overlapCount;
        }
    }
    std::sort(overlapVertices, overlapVertices+overlapCount,
              [](const OverlapVertex& v1, const OverlapVertex& v2)
              {
                  return *v1.aggregate < *v2.aggregate;
              });
    // due to the sorting the isolated aggregates (to be skipped) are at the end.

    return overlapVertices;
}

template<class M, class G, class V, class C, class S>
void buildCoarseSparseMatrix(M& coarseMatrix, G& fineGraph,
                             const V& visitedMap,
                             const C& pinfo,
                             Dune::Amg::AggregatesMap<typename G::VertexDescriptor>& aggregates,
                             const S& overlap)
{
    using OverlapVertex = Dune::Amg ::OverlapVertex<typename G::VertexDescriptor>;
    std::size_t count;

    const OverlapVertex* overlapVertices = buildOverlapVertices(fineGraph,
                                                                pinfo,
                                                                aggregates,
                                                                overlap,
                                                                count);

    // Reset the visited flags of all vertices.
    // As the isolated nodes will be skipped we simply mark them as visited
    const auto UNAGGREGATED = Dune::Amg::AggregatesMap<typename G::VertexDescriptor>::UNAGGREGATED;
    const auto ISOLATED = Dune::Amg::AggregatesMap<typename G::VertexDescriptor>::ISOLATED;
    auto vend = fineGraph.end();

    for(auto vertex = fineGraph.begin(); vertex != vend; ++vertex) {
        assert(aggregates[*vertex] != UNAGGREGATED);
        put(visitedMap, *vertex, aggregates[*vertex]==ISOLATED);
    }

    Dune::Amg::SparsityBuilder<M> sparsityBuilder(coarseMatrix);

    Dune::Amg::ConnectivityConstructor<G,C>::examine(fineGraph, visitedMap, pinfo,
                                                     aggregates, overlap,
                                                     overlapVertices,
                                                     overlapVertices+count,
                                                     sparsityBuilder);
    delete[] overlapVertices;
}

template<class M, class G, class V, class S>
void buildCoarseSparseMatrix(M& coarseMatrix, G& fineGraph, const V& visitedMap,
                             const Dune::Amg::SequentialInformation& pinfo,
                             Dune::Amg::AggregatesMap<typename G::VertexDescriptor>& aggregates,
                             const S&)
{
    // Reset the visited flags of all vertices.
    // As the isolated nodes will be skipped we simply mark them as visited
    const auto UNAGGREGATED = Dune::Amg::AggregatesMap<typename G::VertexDescriptor>::UNAGGREGATED;
    const auto ISOLATED = Dune::Amg::AggregatesMap<typename G::VertexDescriptor>::ISOLATED;
    using Vertex = typename G::VertexIterator;
    Vertex vend = fineGraph.end();
    for(Vertex vertex = fineGraph.begin(); vertex != vend; ++vertex) {
        assert(aggregates[*vertex] != UNAGGREGATED);
        put(visitedMap, *vertex, aggregates[*vertex]==ISOLATED);
    }

    Dune::Amg::SparsityBuilder<M> sparsityBuilder(coarseMatrix);

    Dune::Amg::ConnectivityConstructor<G,Dune::Amg::SequentialInformation>
        ::examine(fineGraph, visitedMap, pinfo, aggregates, sparsityBuilder);
}

} // end namespace Detail

/**
 * @brief A LevelTransferPolicy that uses aggregation to construct the coarse level system.
 * @tparam Operator The type of the fine level operator.
 * @tparam Criterion The criterion that describes the aggregation procedure.
 * @tparam Communication The class that describes the communication pattern.
 */
template<class Operator, class Criterion, class Communication, std::size_t COMPONENT_INDEX>
class OneComponentAggregationLevelTransferPolicy
    : public Dune::Amg::LevelTransferPolicy<Operator, typename Detail::ScalarType<Operator>::value>
{
    typedef Dune::Amg::AggregatesMap<typename Operator::matrix_type::size_type> AggregatesMap;
public:
    using CoarseOperator = typename Detail::ScalarType<Operator>::value;
    typedef Dune::Amg::LevelTransferPolicy<Operator,CoarseOperator> FatherType;
    typedef Communication ParallelInformation;

public:
    OneComponentAggregationLevelTransferPolicy(const Criterion& crit, const Communication& comm)
        : criterion_(crit), communication_(&const_cast<Communication&>(comm))
    {}

    void createCoarseLevelSystem(const Operator& fineOperator)
    {
        prolongDamp_ = criterion_.getProlongationDampingFactor();
        typedef Dune::Amg::PropertiesGraphCreator<Operator> GraphCreator;
        typedef typename GraphCreator::PropertiesGraph PropertiesGraph;
        typedef typename GraphCreator::GraphTuple GraphTuple;

        typedef typename PropertiesGraph::VertexDescriptor Vertex;

        std::vector<bool> excluded(fineOperator.getmat().N(), false);

        using OverlapFlags = Dune::NegateSet<typename ParallelInformation::OwnerSet>;
        GraphTuple graphs = GraphCreator::create(fineOperator, excluded,
                                                 *communication_, OverlapFlags());

        aggregatesMap_.reset(new AggregatesMap(std::get<1>(graphs)->maxVertex()+1));

        int noAggregates, isoAggregates, oneAggregates, skippedAggregates;
        using std::get;
        std::tie(noAggregates, isoAggregates, oneAggregates, skippedAggregates) =
            aggregatesMap_->buildAggregates(fineOperator.getmat(), *(get<1>(graphs)),
                                            criterion_, true);
        std::cout<<"no aggregates="<<noAggregates<<" iso="<<isoAggregates<<" one="<<oneAggregates<<" skipped="<<skippedAggregates<<std::endl;

        using CommunicationArgs = typename Dune::Amg::ConstructionTraits<Communication>::Arguments;
        CommunicationArgs commArgs(communication_->communicator(), communication_->getSolverCategory());
        coarseLevelCommunication_.reset(Dune::Amg::ConstructionTraits<Communication>::construct(commArgs));
        using Iterator = typename std::vector<bool>::iterator;
        auto visitedMap = get(Dune::Amg::VertexVisitedTag(), *(get<1>(graphs)));
        communication_->buildGlobalLookup(fineOperator.getmat().N());
        std::size_t aggregates =
            Dune::Amg::IndicesCoarsener<ParallelInformation,OverlapFlags>
            ::coarsen(*communication_, *get<1>(graphs), visitedMap,
                      *aggregatesMap_, *coarseLevelCommunication_,
                      noAggregates);
        GraphCreator::free(graphs);
        coarseLevelCommunication_->buildGlobalLookup(aggregates);
        Dune::Amg::AggregatesPublisher<Vertex,OverlapFlags,ParallelInformation>
            ::publish(*aggregatesMap_,
                      *communication_,
                      coarseLevelCommunication_->globalLookup());
        std::vector<bool>& visited=excluded;

        std::fill(visited.begin(), visited.end(), false);

        Dune::IteratorPropertyMap<Iterator, Dune::IdentityMap>
            visitedMap2(visited.begin(), Dune::IdentityMap());
        using CoarseMatrix = typename CoarseOperator::matrix_type;
        coarseLevelMatrix_.reset(new CoarseMatrix(aggregates, aggregates,
                                                  CoarseMatrix::row_wise));
        Detail::buildCoarseSparseMatrix(*coarseLevelMatrix_, *get<0>(graphs), visitedMap2,
                                        *communication_,
                                        *aggregatesMap_,
                                        OverlapFlags());
        delete get<0>(graphs);
        communication_->freeGlobalLookup();
        if( static_cast<int>(this->coarseLevelMatrix_->N())
            < criterion_.coarsenTarget())
        {
            coarseLevelCommunication_->freeGlobalLookup();
        }
        calculateCoarseEntries(fineOperator.getmat());
        this->lhs_.resize(this->coarseLevelMatrix_->M());
        this->rhs_.resize(this->coarseLevelMatrix_->N());
        using OperatorArgs = typename Dune::Amg::ConstructionTraits<CoarseOperator>::Arguments;
        OperatorArgs oargs(*coarseLevelMatrix_, *coarseLevelCommunication_);
        this->operator_.reset(Dune::Amg::ConstructionTraits<CoarseOperator>::construct(oargs));
    }

    template<class M>
    void calculateCoarseEntries(const M& fineMatrix)
    {
        for(auto row = fineMatrix.begin(), rowEnd = fineMatrix.end();
            row != rowEnd; ++row)
        {
            const auto& i = (*aggregatesMap_)[row.index()];
            for(auto entry = row->begin(), entryEnd = row->end();
                entry != entryEnd; ++entry)
            {
                const auto& j = (*aggregatesMap_)[entry.index()];
                (*coarseLevelMatrix_)[i][j] += (*entry)[COMPONENT_INDEX][COMPONENT_INDEX];
            }
        }
    }

    void moveToCoarseLevel(const typename FatherType::FineRangeType& fine)
    {
        // Set coarse vector to zero
        this->rhs_=0;

        auto end = fine.end(),  begin=fine.begin();
        for(auto block=begin; block != end; ++block)
        {
            const auto& vertex = (*aggregatesMap_)[block-begin];
            if(vertex != AggregatesMap::ISOLATED)
            {
                this->rhs_[vertex] += (*block)[COMPONENT_INDEX];
            }
        }
        this->lhs_=0;
    }

    void moveToFineLevel(typename FatherType::FineDomainType& fine)
    {
        //for(auto& entry: this->lhs_)
        //    entry*=prolongDamp_;
        this->lhs_ *= prolongDamp_;
        auto end=fine.end(), begin=fine.begin();

        for(auto block=begin; block != end; ++block)
        {
            const auto& vertex = (*aggregatesMap_)[block-begin];
            if(vertex != AggregatesMap::ISOLATED)
                (*block)[COMPONENT_INDEX] += this->lhs_[vertex];
        }
        communication_->copyOwnerToAll(fine,fine);
    }

    OneComponentAggregationLevelTransferPolicy* clone() const
    {
        return new OneComponentAggregationLevelTransferPolicy(*this);
    }

    const Communication& getCoarseLevelCommunication() const
    {
        return *coarseLevelCommunication_;
    }
private:
    typename Operator::matrix_type::field_type prolongDamp_;
    std::shared_ptr<AggregatesMap> aggregatesMap_;
    Criterion criterion_;
    Communication* communication_;
    std::shared_ptr<Communication> coarseLevelCommunication_;
    std::shared_ptr<typename CoarseOperator::matrix_type> coarseLevelMatrix_;
};

template<typename Operator, typename Smoother, typename Criterion,
         typename Communication, std::size_t COMPONENT_INDEX>
class BlackoilAmg
    : public Dune::Preconditioner<typename Operator::domain_type, typename Operator::range_type>
{
    using SmootherArgs   = typename Dune::Amg::SmootherTraits<Smoother>::Arguments;
    using CoarseOperator = typename Detail::ScalarType<Operator>::value;
    using CoarseSmoother = typename Detail::ScalarType<Smoother>::value;
    using FineCriterion  =
        typename Detail::OneComponentCriterionType<Criterion,COMPONENT_INDEX>::value;
    using CoarseCriterion =  typename Detail::ScalarType<Criterion>::value;
    using LevelTransferPolicy =
        OneComponentAggregationLevelTransferPolicy<Operator,
                                                   FineCriterion,
                                                   Communication,
                                                   COMPONENT_INDEX>;
    using CoarseSolverPolicy   =
        Detail::OneStepAMGCoarseSolverPolicy<CoarseOperator,
                                             CoarseSmoother,
                                             CoarseCriterion,
                                             LevelTransferPolicy>;
    using TwoLevelMethod =
        Dune::Amg::TwoLevelMethod<Operator,
                                  CoarseSolverPolicy,
                                  Smoother>;
public:
    // define the category
    enum {
        //! \brief The category the precondtioner is part of.
        category = Operator::category
    };
    BlackoilAmg(const Operator& fineOperator, const Criterion& criterion,
                const SmootherArgs& smargs, const Communication& comm)
        : smoother_(Detail::constructSmoother<Smoother>(fineOperator,smargs,comm)),
          levelTransferPolicy_(criterion, comm),
          coarseSolverPolicy_(smargs, criterion),
          twoLevelMethod_(fineOperator, smoother_,
                          levelTransferPolicy_,
                          coarseSolverPolicy_)
    {}

    void pre(typename TwoLevelMethod::FineDomainType& x,
             typename TwoLevelMethod::FineRangeType& b)
    {
        twoLevelMethod_.pre(x,b);
    }

    void post(typename TwoLevelMethod::FineDomainType& x)
    {
        twoLevelMethod_.post(x);
    }

    void apply(typename TwoLevelMethod::FineDomainType& v,
               const typename TwoLevelMethod::FineRangeType& d)
    {
        twoLevelMethod_.apply(v, d);
    }
private:
    std::shared_ptr<Smoother> smoother_;
    LevelTransferPolicy levelTransferPolicy_;
    CoarseSolverPolicy coarseSolverPolicy_;
    TwoLevelMethod twoLevelMethod_;
};
} // end namespace Opm
#endif
