/*
!========================================================================
!
!                   S P E C F E M 2 D  Version 7 . 0
!                   --------------------------------
!
!     Main historical authors: Dimitri Komatitsch and Jeroen Tromp
!                              CNRS, France
!                       and Princeton University, USA
!                 (there are currently many more authors!)
!                           (c) October 2017
!
! This software is a computer program whose purpose is to solve
! the two-dimensional viscoelastic anisotropic or poroelastic wave equation
! using a spectral-element method (SEM).
!
! This program is free software; you can redistribute it and/or modify
! it under the terms of the GNU General Public License as published by
! the Free Software Foundation; either version 3 of the License, or
! (at your option) any later version.
!
! This program is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
! GNU General Public License for more details.
!
! You should have received a copy of the GNU General Public License along
! with this program; if not, write to the Free Software Foundation, Inc.,
! 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
!
! The full text of the license is available in file "LICENSE".
!
!========================================================================
*/

#include "mesh_constants_omp.h"
#include <cstring>


// prepares a device array with with all inter-element edge-nodes -- this
// is followed by a memcpy and MPI operations

void prepare_boundary_accel_on_omp_device(realw* d_accel, realw* d_send_accel_buffer,
                                      const int ninterface_el,
                                      const int max_nibool_interfaces_ext_mesh,
                                      const int* d_nibool_interfaces_ext_mesh,
                                      const int* d_ibool_interfaces_ext_mesh,
                                      const int* inum_inter_elastic) 
{
    //int id = threadIdx.x + blockIdx.x*blockDim.x + blockIdx.y*gridDim.x*blockDim.x;
    //int ientry,iglob,num_int;

    for( int iinterface=0; iinterface < ninterface_el; iinterface++) {
        int num_int=inum_inter_elastic[iinterface]-1;
        for (int id = 0; id < d_nibool_interfaces_ext_mesh[num_int]; id++) {
            // entry in interface array
            int ientry = id + max_nibool_interfaces_ext_mesh*num_int;
            // global index in wavefield
            int iglob = d_ibool_interfaces_ext_mesh[ientry] - 1;

            d_send_accel_buffer[2*ientry] = d_accel[2*iglob];
            d_send_accel_buffer[2*ientry + 1 ] = d_accel[2*iglob + 1];
        }
    }
}

// prepares and transfers the inter-element edge-nodes to the host to be MPI'd
// (elements on boundary)
extern "C"
void transfer_boun_accel_from_omp_device(long* Mesh_pointer,
                                     realw* send_accel_buffer,
                                     const int* FORWARD_OR_ADJOINT)
{
    Mesh* mp = (Mesh*)(*Mesh_pointer); //get mesh pointer out of fortran integer container

    // checks if anything to do
    if (mp->size_mpi_buffer > 0) {
        //int blocksize = BLOCKSIZE_TRANSFER;
        //int size_padded = ((int)ceil(((double)mp->max_nibool_interfaces_ext_mesh)/((double)blocksize)))*blocksize;
        //int num_blocks_x, num_blocks_y;
        //get_blocks_xy(size_padded/blocksize,&num_blocks_x,&num_blocks_y);
        //dim3 grid(num_blocks_x,num_blocks_y);
        //dim3 threads(blocksize,1,1);

        if (*FORWARD_OR_ADJOINT == 1) {
            //<<<grid,threads,0,mp->compute_stream>>>
            prepare_boundary_accel_on_omp_device(mp->d_accel,mp->d_send_accel_buffer,
                                             mp->ninterface_elastic,
                                             mp->max_nibool_interfaces_ext_mesh,
                                             mp->d_nibool_interfaces_ext_mesh,
                                             mp->d_ibool_interfaces_ext_mesh,
                                             mp->d_inum_interfaces_elastic);

            //cudaStreamSynchronize(mp->compute_stream);
            // copies buffer from GPU to CPU host
            std::memcpy(send_accel_buffer,mp->d_send_accel_buffer,
                        mp->size_mpi_buffer*sizeof(realw));

        }
        else if (*FORWARD_OR_ADJOINT == 3) {
            //<<<grid,threads,0,mp->compute_stream>>>
            prepare_boundary_accel_on_omp_device(mp->d_b_accel,mp->d_b_send_accel_buffer,
                                             mp->ninterface_elastic,
                                             mp->max_nibool_interfaces_ext_mesh,
                                             mp->d_nibool_interfaces_ext_mesh,
                                             mp->d_ibool_interfaces_ext_mesh,
                                             mp->d_inum_interfaces_elastic);

            //cudaStreamSynchronize(mp->compute_stream);

            // copies buffer from GPU to CPU host
            std::memcpy(send_accel_buffer,mp->d_b_send_accel_buffer,
                   mp->size_mpi_buffer*sizeof(realw));
        }
    }
}

extern "C"
void transfer_boundary_from_omp_device_a(long* Mesh_pointer) 
{

    Mesh* mp = (Mesh*)(*Mesh_pointer); // get Mesh from fortran integer wrapper

    if (mp->size_mpi_buffer > 0) {
        //int blocksize = BLOCKSIZE_TRANSFER;
        //int size_padded = ((int)ceil(((double)mp->max_nibool_interfaces_ext_mesh)/((double)blocksize)))*blocksize;
        //int num_blocks_x, num_blocks_y;
        //get_blocks_xy(size_padded/blocksize,&num_blocks_x,&num_blocks_y);
        //dim3 grid(num_blocks_x,num_blocks_y);
        //dim3 threads(blocksize,1,1);

        //<<<grid,threads,0,mp->compute_stream>>>
        prepare_boundary_accel_on_omp_device(mp->d_accel,mp->d_send_accel_buffer,
                                         mp->ninterface_elastic,
                                         mp->max_nibool_interfaces_ext_mesh,
                                         mp->d_nibool_interfaces_ext_mesh,
                                         mp->d_ibool_interfaces_ext_mesh,
                                         mp->d_inum_interfaces_elastic);
        //cudaStreamSynchronize(mp->compute_stream);

        std::memcpy(mp->h_send_accel_buffer,mp->d_send_accel_buffer,
                    mp->size_mpi_buffer*sizeof(realw));
    }
}

extern "C"
void prepare_boundary_on_omp_device(long* Mesh_pointer) 
{
    Mesh* mp = (Mesh*)(*Mesh_pointer); // get Mesh from fortran integer wrapper

    if (mp->size_mpi_buffer > 0) {

        //int blocksize = BLOCKSIZE_TRANSFER;
        //int size_padded = ((int)ceil(((double)mp->max_nibool_interfaces_ext_mesh)/((double)blocksize)))*blocksize;
        //int num_blocks_x, num_blocks_y;
        //get_blocks_xy(size_padded/blocksize,&num_blocks_x,&num_blocks_y);
        //dim3 grid(num_blocks_x,num_blocks_y);
        //dim3 threads(blocksize,1,1);

        //<<<grid,threads,0,mp->compute_stream>>>
        prepare_boundary_accel_on_omp_device(mp->d_accel,mp->d_send_accel_buffer,
                                         mp->ninterface_elastic,
                                         mp->max_nibool_interfaces_ext_mesh,
                                         mp->d_nibool_interfaces_ext_mesh,
                                         mp->d_ibool_interfaces_ext_mesh,
                                         mp->d_inum_interfaces_elastic);
        //cudaStreamSynchronize(mp->compute_stream);
    }
}

extern "C"
void transfer_boundary_to_omp_device_a(long* Mesh_pointer,
                                   realw* buffer_recv_vector_gpu,
                                   const int* max_nibool_interfaces_ext_mesh) 
{
    Mesh* mp = (Mesh*)(*Mesh_pointer); //get mesh pointer out of fortran integer container

    if (mp->size_mpi_buffer > 0) {
        // copy on host memory
        memcpy(mp->h_recv_accel_buffer,buffer_recv_vector_gpu,mp->size_mpi_buffer*sizeof(realw));

        // asynchronous copy to GPU using copy_stream
        std::memcpy(mp->d_send_accel_buffer,mp->h_recv_accel_buffer,
                    mp->size_mpi_buffer*sizeof(realw));
    }
}

void assemble_boundary_accel_on_omp_device(realw* d_accel, realw* d_send_accel_buffer,
                                       const int ninterface_el,
                                       const int max_nibool_interfaces_ext_mesh,
                                       const int* d_nibool_interfaces_ext_mesh,
                                       const int* d_ibool_interfaces_ext_mesh,
                                       const int* inum_inter_elastic) 
{
    //int id = threadIdx.x + blockIdx.x*blockDim.x + blockIdx.y*gridDim.x*blockDim.x;
    //int ientry,iglob,num_int;

    for( int iinterface=0; iinterface < ninterface_el; iinterface++) {
        int num_int=inum_inter_elastic[iinterface]-1;
        for (int id=0; id < d_nibool_interfaces_ext_mesh[num_int];id++) {
            // entry in interface array
            int ientry = id + max_nibool_interfaces_ext_mesh*num_int;
            // global index in wavefield
            int iglob = d_ibool_interfaces_ext_mesh[ientry] - 1;

            //atomicAdd
            d_accel[2*iglob] += d_send_accel_buffer[2*ientry];
            //atomicAdd
            d_accel[2*iglob + 1] += d_send_accel_buffer[2*ientry + 1];
        }
    }
}

// FORWARD_OR_ADJOINT == 1 for accel, and == 3 for b_accel
extern "C"
void transfer_asmbl_accel_to_omp_device(long* Mesh_pointer,
                                    realw* buffer_recv_vector_gpu,
                                    const int* max_nibool_interfaces_ext_mesh,
                                    const int* nibool_interfaces_ext_mesh,
                                    const int* ibool_interfaces_ext_mesh,
                                    const int* FORWARD_OR_ADJOINT) 
{
    Mesh* mp = (Mesh*)(*Mesh_pointer); //get mesh pointer out of fortran integer container

    if (mp->size_mpi_buffer > 0) {
        //daniel: todo - check if this copy is only needed for adjoint simulation, otherwise it is called asynchronously?
        if (*FORWARD_OR_ADJOINT == 1) {
            // Wait until previous copy stream finishes. We assemble while other compute kernels execute.
            //cudaStreamSynchronize(mp->copy_stream);
        }
        else if (*FORWARD_OR_ADJOINT == 3) {
            //synchronize_cuda();
            std::memcpy(mp->d_b_send_accel_buffer, buffer_recv_vector_gpu,
                        mp->size_mpi_buffer*sizeof(realw));
        }

        //int blocksize = BLOCKSIZE_TRANSFER;
        //int size_padded = ((int)ceil(((double)mp->max_nibool_interfaces_ext_mesh)/((double)blocksize)))*blocksize;

        //int num_blocks_x, num_blocks_y;
        //get_blocks_xy(size_padded/blocksize,&num_blocks_x,&num_blocks_y);

        //dim3 grid(num_blocks_x,num_blocks_y);
        //dim3 threads(blocksize,1,1);

        if (*FORWARD_OR_ADJOINT == 1) {
            //assemble forward accel
            //<<<grid,threads,0,mp->compute_stream>>>
            assemble_boundary_accel_on_omp_device(mp->d_accel, mp->d_send_accel_buffer,
                                              mp->ninterface_elastic,
                                              mp->max_nibool_interfaces_ext_mesh,
                                              mp->d_nibool_interfaces_ext_mesh,
                                              mp->d_ibool_interfaces_ext_mesh,
                                              mp->d_inum_interfaces_elastic);
        }
        else if (*FORWARD_OR_ADJOINT == 3) {
            //assemble adjoint accel
            //<<<grid,threads,0,mp->compute_stream>>>
            assemble_boundary_accel_on_omp_device(mp->d_b_accel, mp->d_b_send_accel_buffer,
                                              mp->ninterface_elastic,
                                              mp->max_nibool_interfaces_ext_mesh,
                                              mp->d_nibool_interfaces_ext_mesh,
                                              mp->d_ibool_interfaces_ext_mesh,
                                              mp->d_inum_interfaces_elastic);
        }
    }
}

extern "C"
void sync_copy_from_omp_device(long* Mesh_pointer,
                           int* iphase,
                           realw* send_buffer) 
{
    Mesh* mp = (Mesh*)(*Mesh_pointer); // get Mesh from fortran integer wrapper

    // Wait until async-memcpy of outer elements is finished and start MPI.
    //if (*iphase != 2) { exit_on_cuda_error("sync_copy_from_device must be called for iphase == 2"); }

    if (mp->size_mpi_buffer > 0) {
        // waits for asynchronous copy to finish
        //cudaStreamSynchronize(mp->copy_stream);

        // There have been problems using the pinned-memory with MPI, so
        // we copy the buffer into a non-pinned region.
        memcpy(send_buffer,mp->h_send_accel_buffer,mp->size_mpi_buffer*sizeof(float));
    }
    // memory copy is now finished, so non-blocking MPI send can proceed
}

