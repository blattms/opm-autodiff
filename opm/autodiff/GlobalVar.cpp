#include <config.h>
#include "GlobalVar.hpp"
std::vector<int> globalids;
std::vector<int> isowner;
int MY_RANK;
int MY_SIZE;
int globalGridSize;
int local_index=-1;
int debugCapPress=0;

void init_global_ids(const Dune::CpGrid& grid, const Dune::CpGrid& globalGrid)
{
    globalGridSize = globalGrid.numCells();
    globalids.resize(grid.numCells());
    isowner.resize(grid.numCells());
    std::cout<<"globalids.size()="<<globalids.size()<<" globalGridSize="
             <<globalGridSize;
#if HAVE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &MY_RANK);
    MPI_Comm_size(MPI_COMM_WORLD, &MY_SIZE);
#else
    MY_RANK=0;
    MY_SIZE=1;
#endif
    auto& idxSet = grid.getCellIndexSet();
    Dune::GlobalLookupIndexSet<typename Dune::CpGrid::ParallelIndexSet>
        gidxset(idxSet, grid.numCells());
    
    for ( int i=0; i < grid.numCells(); i++)
    {
        const int* gc=&(grid.globalCell()[0]);
        globalids[i]=grid.globalCell()[i];
        auto pair=gidxset.pair(i);
        if ( pair==nullptr )
        {
            isowner[i] = 1;
        }
        else
        {
            isowner[i] = pair->local().attribute() ==
                Dune::CpGrid::ParallelIndexSet::LocalIndex::Attribute::owner;
        }
    }
}

void print_parallel_vector(const char* title, int* const x)
{
#ifdef PRINT_PARALLEL
        if(MY_RANK>0)
        {
            MPI_Status stat;
            int i;
            MPI_Recv(&i, 1, MPI_INT, MY_RANK-1,
                     9999, MPI_COMM_WORLD, &stat);
        }
        const std::size_t stride=1;
        std::cout<<title<<" rank "<<MY_RANK<<" x.size="<<globalids.size()<<" stride="<<stride<<std::endl;
        
        for(std::size_t i = 0, j=0; i< stride; i++)
            for(auto it=globalids.begin(), end=globalids.end(); it!=end; ++it, ++j)
            {
                std::cout<<i*globalGridSize+*it <<": ";
                std::cout<< x[j]<<" ";
                std::cout<<std::endl;
            }
        if(MY_RANK<MY_SIZE-1)
        {
            int i;
            MPI_Send(&i, 1, MPI_INT, MY_RANK+1,
                     9999, MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);
#endif
}

