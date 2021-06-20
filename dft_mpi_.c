/* Discrete Fourier Transform using MPI*/
#include <stdio.h>
#include <stdlib.h>

#include "mpi.h"
#include <math.h>
#include <fftw3-mpi.h>


int main(int argc, char * argv[]) {
  int i, myrank, nprocs, count, thenode;
  int master = 0;
  float bufsize, * bufArr;
  double time1, time2, time3, time4, time5, time6;
  MPI_Comm cartComm;

  MPI_Init( & argc, & argv);
  MPI_Comm_rank(MPI_COMM_WORLD, & myrank);
  MPI_Comm_size(MPI_COMM_WORLD, & nprocs);

  //fftw variables
  const ptrdiff_t localN0 = N / nprocs localN1 = localN = N / nprocs;
  fftw_plan planDFT;
  fftw_complex * cmpData;
  ptrdiff_t alloc_local, local_n0, local_0_start, i, j;

  fftw_mpi_init(); //init of mpi-fftw area

  int ndims = 2;
  int dimension[2] = {
    sqrt(nprocs),
    sqrt(nprocs)
  };
  int period[2] = {
    0,
    0
  };
  int reorder = 1;
  int coords[2], cartRank;
  int belongs[2];

  MPI_File thefile, thefileWR;
  MPI_Status status;
  MPI_Offset filesize;
  int fsize;

  //creating cartesian topology
  MPI_Cart_create(MPI_COMM_WORLD, ndims, dimension, period, reorder, & cartComm);
  MPI_Cart_coords(cartComm, myrank, ndims, coords);
  MPI_Cart_rank(cartComm, coords, & cartRank);
  MPI_Barrier(cartComm);

  time1 = MPI_Wtime(); ///
  MPI_File_open(cartComm, "myfile", MPI_MODE_RDONLY,
    MPI_INFO_NULL, & thefile);
  MPI_File_get_size(thefile, & filesize); /* in bytes */
  time2 = MPI_Wtime(); ///

  fsize = filesize / sizeof(float); /* in number of ints */
  bufsize = fsize / nprocs; /* local number to read */
  bufArr = (float * ) malloc(bufsize * sizeof(float));

  //MPI_Type_vector(1, localN, localN, MPI_DOUBLE, &fileType);
  //MPI_Type_commit(&fileType);
  //MPI_Type_extent(fileType, &fileTypeEx);

  time3 = MPI_Wtime(); ///
  MPI_File_set_view(thefile, myrank * bufsize * sizeof(float),
    MPI_FLOAT, MPI_FLOAT, "native", MPI_INFO_NULL);
  MPI_File_read(thefile, bufArr, bufsize, MPI_FLOAT, & status);
  MPI_Get_count( & status, MPI_FLOAT, & count);
  MPI_File_close( & thefile);
  time4 = MPI_Wtime(); ///
  if (myrank == master) {
    printf("process %d read %d ints\n", myrank, count);
  }
  //############################

  /// get local data size and allocate 										
  alloc_local = fftw_mpi_local_size_2d(N0, N1, cartComm, &
    local_n0, & local_0_start);
  cmpData = fftw_alloc_complex(alloc_local);

  ///creating plan for DFT 
  /*protoype: fftw_plan fftw_mpi_plan_dft_r2c_2d(ptrdiff_t n0, ptrdiff_t n1,
                                  double *in, fftw_complex *out, MPI_Comm comm, unsigned flags);*/
  planDFT = fftw_mpi_plan_dft_r2c_2d(localN0, localN1, buffArr, cmpData,
    cartComm, FFTW_FORWARD, FFTW_ESTIMATE);

  fftw_execute(planDFT); /// apply DFT
  fftw_destroy_plan(planDFT); /// release plan

  //##############################

  time5 = MPI_Wtime();
  MPI_File_open(cartComm, "createdFile", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, & thefileWR);
  MPI_File_set_view(thefileWR, myrank * bufsize * sizeof(float), MPI_FLOAT, MPI_FLOAT, "native", MPI_INFO_NULL);
  MPI_File_write(thefileWR, bufArr, bufsize, MPI_FLOAT, MPI_STATUS_IGNORE);
  MPI_File_close( & thefileWR);
  time6 = MPI_Wtime();

  if (myrank == master) {
    printf("nprocs: %d. Matrix Row Size: %f\n", nprocs, sqrt(fsize));
    printf("Parallel  Read time is: %f\n", (time4 - time3 + time2 - time1));
    printf("\n Parallel Write time is: %f\n", (time6 - time5));
  }

  MPI_Finalize();
  return 0;
}
