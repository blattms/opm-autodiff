#ifndef OPM_AUTODIFF_GLOBAL_VAR
#define OPM_AUTODIFF_GLOBAL_VAR
#include<vector>
#include<dune/grid/CpGrid.hpp>
#include<opm/autodiff/AutoDiffHelpers.hpp>
#include<dune/istl/bcrsmatrix.hh>
#include<dune/common/fmatrix.hh>

extern std::vector<int> globalids;
extern std::vector<int> isowner;
extern int MY_RANK;
extern int MY_SIZE;
extern int globalGridSize;
extern int localGridSize;
extern int local_index;
extern int debugCapPress;

#define PRINT_PARALLEL
extern void init_global_ids(const Dune::CpGrid& grid, const Dune::CpGrid& globalGrid);

void print_parallel_vector(const char* title, int* const x);

template<class T>
void print_parallel_vector(const char* title, const T& x)
{    
#ifdef  PRINT_PARALLEL
        if(MY_RANK>0)
        {
            MPI_Status stat;
            int i;
            MPI_Recv(&i, 1, MPI_INT, MY_RANK-1,
                     9999, MPI_COMM_WORLD, &stat);
        }
        const std::size_t stride=x.size()/globalids.size();
        std::cout<<title<<" rank "<<MY_RANK<<" x.size="<<x.size()<<" stride="<<stride<<std::endl;
        if(MY_RANK==0)
        {
        for(std::size_t i = 0, j=0; i< stride; i++)
            for(auto it=globalids.begin(), end=globalids.end(); it!=end; ++it, ++j)
            {
                std::cout<<i*globalGridSize+*it <<": ";
                std::cout<< x[j]<<" ";
                std::cout<<isowner[j]<<std::endl;
            }
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


template<class T>
void print_parallel_matrix(const char* title, const T& x)
{
#ifdef PRINT_PARALLEL
        if(MY_RANK>0)
        {
            MPI_Status stat;
            int i;
            MPI_Recv(&i, 1, MPI_INT, MY_RANK-1,
                     9999, MPI_COMM_WORLD, &stat);
        }
        std::cout<<title<<" rank "<<MY_RANK<<" rows="<<x.rows()<<" cols="<<x.cols()<<std::endl;
        const std::size_t stride=x.outerSize()/globalids.size();
        for(int i=0; i<x.outerSize(); ++i)
        {
            for (typename T::InnerIterator it(x,i); it; ++it)
            {
                std::cout<<(it.row()/globalids.size())*globalGridSize+globalids[it.row()%globalids.size()]<<", "
                         <<(it.col()/globalids.size())*globalGridSize+globalids[it.col()%globalids.size()]<<", "<<it.value()<<std::endl;
            }
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

template<class T>
void print_parallel_matrix(const char* title, const Dune::BCRSMatrix<T>& x)
{
#ifdef PRINT_PARALLEL
        if(MY_RANK>0)
        {
            MPI_Status stat;
            int i;
            MPI_Recv(&i, 1, MPI_INT, MY_RANK-1,
                     9999, MPI_COMM_WORLD, &stat);
        }
        int zeros=0;
        int nonzeros=0;
        std::cout<<title<<" rank "<<MY_RANK<<" rows="<<x.N()<<" cols="<<x.M()<<std::endl;
        for(auto row=x.begin(), rend=x.end(); row != rend; ++row)
        {
            if(isowner[row.index()])
            for(auto col=row->begin(), cend=row->end(); col != cend; ++col)
            {
                if(MY_RANK==1)
                {
                    std::cout<<"| "<<globalids[row.index()]<<", "<<globalids[col.index()]<<" | ";
                    for(int i=0; i< col->N(); ++i)
                        for(int j=0; j< col->M(); ++j)
                            std::cout<<(*col)[i][j]<<" ";
                    std::cout<<" |"<<std::endl;
                }
                if(col->frobenius_norm()==0.0)
                    ++zeros;
                ++nonzeros;
            }
        }
        std::cout<<zeros<<" of "<<nonzeros<<" blocks  are zero on rank "
                 <<MY_RANK<<std::endl;
        if(MY_RANK<MY_SIZE-1)
        {
            int i;
            MPI_Send(&i, 1, MPI_INT, MY_RANK+1,
                     9999, MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);
#endif
}

template<class T, class CpGrid>
void print_parallel_face_values(const char* title, const CpGrid& grid, const T& x)
{    
#ifdef PRINT_PARALLEL
        if(MY_RANK>0)
        {
            MPI_Status stat;
            int i;
            MPI_Recv(&i, 1, MPI_INT, MY_RANK-1,
                     9999, MPI_COMM_WORLD, &stat);
        }
        std::map<int,std::vector<double> > values;
        
        
        for ( int i=0; i < grid.numCells(); ++i)
        {
            const auto & ids = grid.getCellIndexSet();
            auto flag = Dune::OwnerOverlapCopyAttributeSet::AttributeSet::owner;
            if (ids.size())
                flag = grid.getCellIndexSet()[globalids[i]].local().attribute();
                        
            if( flag != Dune::OwnerOverlapCopyAttributeSet::AttributeSet::owner )
                continue;
            int key = grid.globalCell()[i];
            std::vector<double> face_values;
            for ( int j=0; j < grid.numCellFaces(i); ++j)
            {
                face_values.push_back(x[grid.cellFace(i,j)]);
            }
            values[key] = face_values;
        }
        const std::size_t stride=x.size()/globalids.size();
        std::cout<<title<<" rank "<<MY_RANK<<" x.size="<<x.size()<<" stride="<<stride<<std::endl;
        for(auto it=values.begin(), itend=values.end(); it!=itend; ++it)
        {
            std::cout<<it->first<<": ";
            std::copy(it->second.begin(), it->second.end(), std::ostream_iterator<double>(std::cout, " "));
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

template<class T, class CpGrid>
void print_parallel_iface_values(const char* title, const CpGrid& grid,
                                 const Opm::HelperOps h, const T& x)
{    
#ifdef PRINT_PARALLEL
        if(MY_RANK>0)
        {
            MPI_Status stat;
            int i;
            MPI_Recv(&i, 1, MPI_INT, MY_RANK-1,
                     9999, MPI_COMM_WORLD, &stat);
        }
        std::map<int,std::list<double> > values;

        auto nif = h.internal_faces.size();
        
        auto face_cells = Opm::AutoDiffGrid::faceCellsToEigen(grid);
        for (int iface = 0; iface < nif; ++iface) {
            const int f  = h.internal_faces[iface];
            const int c1 = face_cells(f,0);
            const int c2 = face_cells(f,1);
            int minval = c1;
            int gval = grid.globalCell()[c1];
            if(grid.globalCell()[c1] > grid.globalCell()[c2])
            {
                minval=c2;
                gval = grid.globalCell()[c2];
            }
            
            auto & fset = values[gval];
            fset.push_back(x[iface]);
        }
        std::cout<<title<<" rank "<<MY_RANK<<" x.size="<<x.size()<<std::endl;
        for(auto it=values.begin(), itend=values.end(); it!=itend; ++it)
        {
            std::cout<<it->first<<": ";
            std::copy(it->second.begin(), it->second.end(), std::ostream_iterator<double>(std::cout, " "));
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
template<class CpGrid>
void print_parallel_ifaces(const CpGrid& grid,
                           const Opm::HelperOps h)
{    
#ifdef PRINT_PARALLEL
        std::map<int,std::list<int> > internal_neighbors;

        auto nif = h.internal_faces.size();
        int maxid = 0;
        
        auto face_cells = Opm::AutoDiffGrid::faceCellsToEigen(grid);
        for (int iface = 0; iface < nif; ++iface) {
            const int f  = h.internal_faces[iface];
            const int c1 = face_cells(f,0);
            const int c2 = face_cells(f,1);
            int minval = c1;
            int nbval = grid.globalCell()[c2];
            int gval = grid.globalCell()[c1];
            if(grid.globalCell()[c1] > grid.globalCell()[c2])
            {
                minval=c2;
                gval = grid.globalCell()[c2];
                nbval = grid.globalCell()[c1];
            }
            maxid=std::max(maxid, gval);
            auto & fset = internal_neighbors[gval];
            fset.push_back(nbval);
        }
        int gmax;
        MPI_Allreduce(&maxid, &gmax, 1, MPI_INT, MPI_MAX,MPI_COMM_WORLD);
        
        std::cout<<"internal faces rank "<<MY_RANK<<std::endl;
        auto it=internal_neighbors.begin(), itend=internal_neighbors.end();
        for(int id=0;id<=gmax; ++id)
        {
            if(MY_RANK>0)
            {
                MPI_Status stat;
                int i;
                MPI_Recv(&i, 1, MPI_INT, MY_RANK-1,
                         9999, MPI_COMM_WORLD, &stat);
            }

            while(it != itend && it->first <id)
                ++it;
            
            if(it != itend && it->first==id)
            {
                std::cout<<it->first<<": ";
                std::copy(it->second.begin(), it->second.end(), std::ostream_iterator<double>(std::cout, " "));
                std::cout<<std::endl<<std::flush;
            }
            if(MY_RANK<MY_SIZE-1)
            {
                int i;
                MPI_Send(&i, 1, MPI_INT, MY_RANK+1,
                         9999, MPI_COMM_WORLD);
            }
            
            MPI_Barrier(MPI_COMM_WORLD);
            std::cout<<std::flush;
        }
        
#endif
}
#endif
