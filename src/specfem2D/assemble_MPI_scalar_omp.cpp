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
void prepare_boundary_potential_on_omp_device(realw* d_potential_dot_dot_acoustic,
                                                     realw* d_send_potential_dot_dot_buffer,
                                                     const int ninterface_ac,
                                                     const int max_nibool_interfaces_ext_mesh,
                                                     const int* d_nibool_interfaces_ext_mesh,
                                                     const int* d_ibool_interfaces_ext_mesh,
                                                     const int* inum_inter_acoustic) 
{
    //int id = threadIdx.x + blockIdx.x*blockDim.x + blockIdx.y*gridDim.x*blockDim.x;
    //int ientry,iglob,num_int;

    for(int iinterface=0; iinterface < ninterface_ac; iinterface++) {
        int num_int=inum_inter_acoustic[iinterface]-1;

        //if (id<d_nibool_interfaces_ext_mesh[num_int]) {
        for (int id=0; id<d_nibool_interfaces_ext_mesh[num_int]; id++) {
            // entry in interface array
            int ientry = id + max_nibool_interfaces_ext_mesh*num_int;
            // global index in wavefield
            int iglob = d_ibool_interfaces_ext_mesh[ientry] - 1;

            d_send_potential_dot_dot_buffer[ientry] = d_potential_dot_dot_acoustic[iglob];
        }
    }
}

// prepares and transfers the inter-element edge-nodes to the host to be MPI'd
extern "C"
void transfer_boun_pot_from_omp_device(long* Mesh_pointer,
                                   realw* send_potential_dot_dot_buffer,
                                   const int* FORWARD_OR_ADJOINT)
{
    Mesh* mp = (Mesh*)(*Mesh_pointer); //get mesh pointer out of fortran integer container

    // checks if anything to do
    if (mp->size_mpi_buffer_potential > 0) {
        //int blocksize = BLOCKSIZE_TRANSFER;
        //int size_padded = ((int)ceil(((double)(mp->max_nibool_interfaces_ext_mesh))/((double)blocksize)))*blocksize;
        //int num_blocks_x, num_blocks_y;
        //get_blocks_xy(size_padded/blocksize,&num_blocks_x,&num_blocks_y);
        //dim3 grid(num_blocks_x,num_blocks_y);
        //dim3 threads(blocksize,1,1);

        if (*FORWARD_OR_ADJOINT == 1) {
            //<<<grid,threads,0,mp->compute_stream>>>
            prepare_boundary_potential_on_omp_device(mp->d_potential_dot_dot_acoustic,
                                                 mp->d_send_potential_dot_dot_buffer,
                                                 mp->ninterface_acoustic,
                                                 mp->max_nibool_interfaces_ext_mesh,
                                                 mp->d_nibool_interfaces_ext_mesh,
                                                 mp->d_ibool_interfaces_ext_mesh,
                                                 mp->d_inum_interfaces_acoustic);

            std::memcpy(send_potential_dot_dot_buffer,mp->d_send_potential_dot_dot_buffer,
                        mp->size_mpi_buffer_potential*sizeof(realw));
        }
        else if (*FORWARD_OR_ADJOINT == 3) {
            // backward/reconstructed wavefield buffer
            //<<<grid,threads,0,mp->compute_stream>>>
            prepare_boundary_potential_on_omp_device(mp->d_b_potential_dot_dot_acoustic,
                                                 mp->d_b_send_potential_dot_dot_buffer,
                                                 mp->ninterface_acoustic,
                                                 mp->max_nibool_interfaces_ext_mesh,
                                                 mp->d_nibool_interfaces_ext_mesh,
                                                 mp->d_ibool_interfaces_ext_mesh,
                                                 mp->d_inum_interfaces_acoustic);

            std::memcpy(send_potential_dot_dot_buffer,mp->d_b_send_potential_dot_dot_buffer,
                        mp->size_mpi_buffer_potential*sizeof(realw));
        }
    }
}

void assemble_boundary_potential_on_omp_device(realw* d_potential_dot_dot_acoustic,
                                               realw* d_send_potential_dot_dot_buffer,
                                               const int ninterface_ac,
                                               const int max_nibool_interfaces_ext_mesh,
                                               const int* d_nibool_interfaces_ext_mesh,
                                               const int* d_ibool_interfaces_ext_mesh,
                                               const int* inum_inter_acoustic) 
{
    //int id = threadIdx.x + blockIdx.x*blockDim.x + blockIdx.y*gridDim.x*blockDim.x;
    for( int iinterface=0; iinterface < ninterface_ac; iinterface++) {
        int num_int=inum_inter_acoustic[iinterface]-1;
        //if (id<d_nibool_interfaces_ext_mesh[num_int]) {
        for (int id = 0; id<d_nibool_interfaces_ext_mesh[num_int];id++) {
            int ientry = id + max_nibool_interfaces_ext_mesh*num_int;
            int iglob = d_ibool_interfaces_ext_mesh[ientry] - 1;

            //atomicAdd
            d_potential_dot_dot_acoustic[iglob] += d_send_potential_dot_dot_buffer[ientry];
        }
    }
}


extern "C"
void transfer_asmbl_pot_to_omp_device(long* Mesh_pointer,
                                      realw* buffer_recv_scalar_gpu,
                                      const int* FORWARD_OR_ADJOINT) 
{
    Mesh* mp = (Mesh*)(*Mesh_pointer); //get mesh pointer out of fortran integer container
    if (mp->size_mpi_buffer_potential > 0) {
        //int blocksize = BLOCKSIZE_TRANSFER;
        //int size_padded = ((int)ceil(((double)mp->max_nibool_interfaces_ext_mesh)/((double)blocksize)))*blocksize;
        //int num_blocks_x, num_blocks_y;
        //get_blocks_xy(size_padded/blocksize,&num_blocks_x,&num_blocks_y);
        //dim3 grid(num_blocks_x,num_blocks_y);
        //dim3 threads(blocksize,1,1);

        //synchronize_cuda();

        if (*FORWARD_OR_ADJOINT == 1) {
            // copies buffer onto GPU
            memcpy(mp->d_send_potential_dot_dot_buffer, buffer_recv_scalar_gpu,
                        mp->size_mpi_buffer_potential*sizeof(realw));

            //assemble forward field
            //<<<grid,threads,0,mp->compute_stream>>>
            assemble_boundary_potential_on_omp_device(mp->d_potential_dot_dot_acoustic,
                                                  mp->d_send_potential_dot_dot_buffer,
                                                  mp->ninterface_acoustic,
                                                  mp->max_nibool_interfaces_ext_mesh,
                                                  mp->d_nibool_interfaces_ext_mesh,
                                                  mp->d_ibool_interfaces_ext_mesh,
                                                  mp->d_inum_interfaces_acoustic);


        }
        else if (*FORWARD_OR_ADJOINT == 3) {
            // copies buffer onto GPU
            std::memcpy(mp->d_b_send_potential_dot_dot_buffer, buffer_recv_scalar_gpu,
                        mp->size_mpi_buffer_potential*sizeof(realw));

            //assemble reconstructed/backward field
            //<<<grid,threads,0,mp->compute_stream>>>
            assemble_boundary_potential_on_omp_device(mp->d_b_potential_dot_dot_acoustic,
                                                  mp->d_b_send_potential_dot_dot_buffer,
                                                  mp->ninterface_acoustic,
                                                  mp->max_nibool_interfaces_ext_mesh,
                                                  mp->d_nibool_interfaces_ext_mesh,
                                                  mp->d_ibool_interfaces_ext_mesh,
                                                  mp->d_inum_interfaces_acoustic);
        }
    }
}

