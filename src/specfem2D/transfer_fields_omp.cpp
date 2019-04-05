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

// Transfer functions

// for ELASTIC simulations
extern "C"
void transfer_fields_el_to_device(int* size, realw* displ, realw* veloc, realw* accel,long* Mesh_pointer)
{
    Mesh* mp = (Mesh*)(*Mesh_pointer); //get mesh pointer out of fortran integer container

    cudaMemcpy(mp->d_displ,displ,sizeof(realw)*(*size),cudaMemcpyHostToDevice);
    cudaMemcpy(mp->d_veloc,veloc,sizeof(realw)*(*size),cudaMemcpyHostToDevice);
    cudaMemcpy(mp->d_accel,accel,sizeof(realw)*(*size),cudaMemcpyHostToDevice);
}

extern "C"
void transfer_fields_el_from_device(int* size, realw* displ, realw* veloc, realw* accel,long* Mesh_pointer)
{
    Mesh* mp = (Mesh*)(*Mesh_pointer); //get mesh pointer out of fortran integer container

    cudaMemcpy(displ,mp->d_displ,sizeof(realw)*(*size),cudaMemcpyDeviceToHost);
    cudaMemcpy(veloc,mp->d_veloc,sizeof(realw)*(*size),cudaMemcpyDeviceToHost);
    cudaMemcpy(accel,mp->d_accel,sizeof(realw)*(*size),cudaMemcpyDeviceToHost);
}

extern "C"
void transfer_b_fields_to_device(int* size, realw* b_displ, realw* b_veloc, realw* b_accel, long* Mesh_pointer) 
{
    Mesh* mp = (Mesh*)(*Mesh_pointer); //get mesh pointer out of fortran integer container

    cudaMemcpy(mp->d_b_displ,b_displ,sizeof(realw)*(*size),cudaMemcpyHostToDevice);
    cudaMemcpy(mp->d_b_veloc,b_veloc,sizeof(realw)*(*size),cudaMemcpyHostToDevice);
    cudaMemcpy(mp->d_b_accel,b_accel,sizeof(realw)*(*size),cudaMemcpyHostToDevice);
}

extern "C"
void transfer_b_fields_from_device(int* size, realw* b_displ, realw* b_veloc, realw* b_accel,long* Mesh_pointer)
{
    Mesh* mp = (Mesh*)(*Mesh_pointer); //get mesh pointer out of fortran integer container

    cudaMemcpy(b_displ,mp->d_b_displ,sizeof(realw)*(*size),cudaMemcpyDeviceToHost);
    cudaMemcpy(b_veloc,mp->d_b_veloc,sizeof(realw)*(*size),cudaMemcpyDeviceToHost);
    cudaMemcpy(b_accel,mp->d_b_accel,sizeof(realw)*(*size),cudaMemcpyDeviceToHost);
}

extern "C"
void transfer_accel_to_device(int* size, realw* accel,long* Mesh_pointer)
{
    Mesh* mp = (Mesh*)(*Mesh_pointer); //get mesh pointer out of fortran integer container

    cudaMemcpy(mp->d_accel,accel,sizeof(realw)*(*size),cudaMemcpyHostToDevice);
}

extern "C"
void transfer_accel_from_device(int* size, realw* accel,long* Mesh_pointer)
{
    Mesh* mp = (Mesh*)(*Mesh_pointer); //get mesh pointer out of fortran integer container

    cudaMemcpy(accel,mp->d_accel,sizeof(realw)*(*size),cudaMemcpyDeviceToHost);
}

extern "C"
void transfer_b_accel_from_device(int* size, realw* b_accel,long* Mesh_pointer)
{
    Mesh* mp = (Mesh*)(*Mesh_pointer); //get mesh pointer out of fortran integer container

    cudaMemcpy(b_accel,mp->d_b_accel,sizeof(realw)*(*size),cudaMemcpyDeviceToHost);
}

extern "C"
void transfer_b_displ_from_device(int* size, realw* displ,long* Mesh_pointer)
{
    Mesh* mp = (Mesh*)(*Mesh_pointer); //get mesh pointer out of fortran integer container

    cudaMemcpy(displ,mp->d_b_displ,sizeof(realw)*(*size),cudaMemcpyDeviceToHost);
}

extern "C"
void transfer_displ_from_device(int* size, realw* displ,long* Mesh_pointer)
{
    Mesh* mp = (Mesh*)(*Mesh_pointer); //get mesh pointer out of fortran integer container

    cudaMemcpy(displ,mp->d_displ,sizeof(realw)*(*size),cudaMemcpyDeviceToHost);
}

extern "C"
void transfer_kernels_el_to_host(long* Mesh_pointer,
                                 realw* h_rho_kl,
                                 realw* h_mu_kl,
                                 realw* h_kappa_kl,
                                 int* NSPEC_AB) 
{
    Mesh* mp = (Mesh*)(*Mesh_pointer);

    cudaMemcpy(h_rho_kl,mp->d_rho_kl,*NSPEC_AB*NGLL2*sizeof(realw),cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mu_kl,mp->d_mu_kl,*NSPEC_AB*NGLL2*sizeof(realw),cudaMemcpyDeviceToHost);
    cudaMemcpy(h_kappa_kl,mp->d_kappa_kl,*NSPEC_AB*NGLL2*sizeof(realw),cudaMemcpyDeviceToHost);
}


// for ACOUSTIC simulations
extern "C"
void transfer_fields_ac_to_device(int* size,
                                  realw* potential_acoustic,
                                  realw* potential_dot_acoustic,
                                  realw* potential_dot_dot_acoustic,
                                  long* Mesh_pointer) 
{
    //get mesh pointer out of fortran integer container
    Mesh* mp = (Mesh*)(*Mesh_pointer);

    cudaMemcpy(mp->d_potential_acoustic,potential_acoustic,
                                       sizeof(realw)*(*size),cudaMemcpyHostToDevice);
    cudaMemcpy(mp->d_potential_dot_acoustic,potential_dot_acoustic,
                                       sizeof(realw)*(*size),cudaMemcpyHostToDevice);
    cudaMemcpy(mp->d_potential_dot_dot_acoustic,potential_dot_dot_acoustic,
                                       sizeof(realw)*(*size),cudaMemcpyHostToDevice);
}

extern "C"
void transfer_b_fields_ac_to_device(int* size,
                                    realw* b_potential_acoustic,
                                    realw* b_potential_dot_acoustic,
                                    realw* b_potential_dot_dot_acoustic,
                                    long* Mesh_pointer) 
{
    //get mesh pointer out of fortran integer container
    Mesh* mp = (Mesh*)(*Mesh_pointer);

    cudaMemcpy(mp->d_b_potential_acoustic,b_potential_acoustic,
                                        sizeof(realw)*(*size),cudaMemcpyHostToDevice);
    cudaMemcpy(mp->d_b_potential_dot_acoustic,b_potential_dot_acoustic,
                                        sizeof(realw)*(*size),cudaMemcpyHostToDevice);
    cudaMemcpy(mp->d_b_potential_dot_dot_acoustic,b_potential_dot_dot_acoustic,
                                        sizeof(realw)*(*size),cudaMemcpyHostToDevice);
}

extern "C"
void transfer_fields_ac_from_device(int* size,
                                    realw* potential_acoustic,
                                    realw* potential_dot_acoustic,
                                    realw* potential_dot_dot_acoustic,
                                    long* Mesh_pointer)
{
    Mesh* mp = (Mesh*)(*Mesh_pointer);

    cudaMemcpy(potential_acoustic,mp->d_potential_acoustic,
                                        sizeof(realw)*(*size),cudaMemcpyDeviceToHost)
    cudaMemcpy(potential_dot_acoustic,mp->d_potential_dot_acoustic,
                                        sizeof(realw)*(*size),cudaMemcpyDeviceToHost);
    cudaMemcpy(potential_dot_dot_acoustic,mp->d_potential_dot_dot_acoustic,
                                        sizeof(realw)*(*size),cudaMemcpyDeviceToHost);
}

extern "C"
void transfer_b_fields_ac_from_device(int* size,
                                      realw* b_potential_acoustic,
                                      realw* b_potential_dot_acoustic,
                                      realw* b_potential_dot_dot_acoustic,
                                      long* Mesh_pointer) 
{
    Mesh* mp = (Mesh*)(*Mesh_pointer);

    cudaMemcpy(b_potential_acoustic,mp->d_b_potential_acoustic,
                                        sizeof(realw)*(*size),cudaMemcpyDeviceToHost);
    cudaMemcpy(b_potential_dot_acoustic,mp->d_b_potential_dot_acoustic,
                                        sizeof(realw)*(*size),cudaMemcpyDeviceToHost);
    cudaMemcpy(b_potential_dot_dot_acoustic,mp->d_b_potential_dot_dot_acoustic,
                                        sizeof(realw)*(*size),cudaMemcpyDeviceToHost);
}

extern "C"
void transfer_b_potential_ac_from_device(int* size,
                                         realw* b_potential_acoustic,
                                         long* Mesh_pointer) 
{
    Mesh* mp = (Mesh*)(*Mesh_pointer);
    cudaMemcpy(b_potential_acoustic,mp->d_b_potential_acoustic,
                                        sizeof(realw)*(*size),cudaMemcpyDeviceToHost);
}

extern "C"
void transfer_b_potential_ac_to_device(int* size,
                                       realw* b_potential_acoustic,
                                       long* Mesh_pointer) 
{
    Mesh* mp = (Mesh*)(*Mesh_pointer);
    cudaMemcpy(mp->d_b_potential_acoustic,b_potential_acoustic,
                                        sizeof(realw)*(*size),cudaMemcpyHostToDevice);
}

extern "C"
void transfer_dot_dot_from_device(int* size, realw* potential_dot_dot_acoustic,long* Mesh_pointer) 
{
    Mesh* mp = (Mesh*)(*Mesh_pointer);

    cudaMemcpy(potential_dot_dot_acoustic,mp->d_potential_dot_dot_acoustic,
                sizeof(realw)*(*size),cudaMemcpyDeviceToHost);
}

extern "C"
void transfer_b_dot_dot_from_device(int* size, realw* b_potential_dot_dot_acoustic,long* Mesh_pointer) 
{
    //get mesh pointer out of fortran integer container
    Mesh* mp = (Mesh*)(*Mesh_pointer);

    cudaMemcpy(b_potential_dot_dot_acoustic,mp->d_b_potential_dot_dot_acoustic,
                                        sizeof(realw)*(*size),cudaMemcpyDeviceToHost);
}

extern "C"
void transfer_kernels_ac_to_host(long* Mesh_pointer,realw* h_rho_ac_kl,realw* h_kappa_ac_kl,int* NSPEC_AB) 
{
    //get mesh pointer out of fortran integer container
    Mesh* mp = (Mesh*)(*Mesh_pointer);

    int size = *NSPEC_AB*NGLL2;

    // copies kernel values over to CPU host
    cudaMemcpy(h_rho_ac_kl,mp->d_rho_ac_kl,size*sizeof(realw),
                                        cudaMemcpyDeviceToHost);
    cudaMemcpy(h_kappa_ac_kl,mp->d_kappa_ac_kl,size*sizeof(realw),
                                        cudaMemcpyDeviceToHost);
}

// for Hess kernel calculations
extern "C"
void transfer_kernels_hess_el_tohost(long* Mesh_pointer,realw* h_hess_kl,int* NSPEC_AB) 
{
    Mesh* mp = (Mesh*)(*Mesh_pointer);

    cudaMemcpy(h_hess_kl,mp->d_hess_el_kl,NGLL2*(*NSPEC_AB)*sizeof(realw),
                                        cudaMemcpyDeviceToHost);
}

extern "C"
void transfer_kernels_hess_ac_tohost(long* Mesh_pointer,realw* h_hess_ac_kl,int* NSPEC_AB) 
{
    Mesh* mp = (Mesh*)(*Mesh_pointer);

    cudaMemcpy(h_hess_ac_kl,mp->d_hess_ac_kl,NGLL2*(*NSPEC_AB)*sizeof(realw),
                                        cudaMemcpyDeviceToHost);
}

//For UNDO_ATTENUATION
extern "C"
void transfer_viscoacoustic_b_var_to_device(int* size,
                                            realw* b_e1_acous_sf,
                                            realw* b_sum_forces_old,
                                            long* Mesh_pointer) 
{
    Mesh* mp = (Mesh*)(*Mesh_pointer); //get mesh pointer out of fortran integer container
    cudaMemcpy(mp->d_b_sum_forces_old,b_sum_forces_old,sizeof(realw)*(*size),cudaMemcpyHostToDevice);
    cudaMemcpy(mp->d_b_e1_acous,b_e1_acous_sf,sizeof(realw)*(*size)*N_SLS,cudaMemcpyHostToDevice);
}

extern "C"
void transfer_viscoacoustic_var_from_device(int* size,
                                            realw* e1_acous_sf,
                                            realw* sum_forces_old,
                                            long* Mesh_pointer) 
{
    Mesh* mp = (Mesh*)(*Mesh_pointer); //get mesh pointer out of fortran integer container

    cudaMemcpy(sum_forces_old,mp->d_sum_forces_old,sizeof(realw)*(*size),cudaMemcpyDeviceToHost);
    cudaMemcpy(e1_acous_sf,mp->d_e1_acous,sizeof(realw)*(*size)*N_SLS,cudaMemcpyDeviceToHost);
}

extern "C"
void transfer_async_pot_ac_from_device(realw* pot_buffer,long* Mesh_pointer)
{
    Mesh* mp = (Mesh*)(*Mesh_pointer); //get mesh pointer out of fortran integer container

    // waits for previous transfer to finish
    cudaStreamSynchronize(mp->compute_stream);

    cudaStreamWaitEvent(mp->compute_stream,mp->transfer_is_complete1,0);
    // adds the copy of d_potential_acoustic to the compute_stream stream to make sure it will be not overwritten by this same stream in further operations
    cudaMemcpyAsync(mp->d_potential_acoustic_buffer,mp->d_potential_acoustic,sizeof(realw)*mp->NGLOB_AB,cudaMemcpyDeviceToDevice,mp->compute_stream);
    // We create an event to know when the GPU buffer is ready for the transfer GPU ==> CPU
    cudaEventRecord(mp->transfer_is_complete2,mp->compute_stream);
    cudaStreamWaitEvent(mp->copy_stream_no_backward,mp->transfer_is_complete2,0);

    cudaMemcpyAsync(pot_buffer,mp->d_potential_acoustic_buffer,sizeof(realw)*mp->NGLOB_AB,cudaMemcpyDeviceToHost,mp->copy_stream_no_backward);

    cudaEventRecord(mp->transfer_is_complete1,mp->copy_stream_no_backward);
}

extern "C"
void transfer_async_pot_ac_to_device(realw* pot_buffer, long* Mesh_pointer) 
{
    Mesh* mp = (Mesh*)(*Mesh_pointer); //get mesh pointer out of fortran integer container

    cudaStreamSynchronize(mp->compute_stream);

    cudaStreamWaitEvent(mp->compute_stream,mp->transfer_is_complete1,0);

    cudaMemcpyAsync(mp->d_b_potential_acoustic,mp->d_potential_acoustic_buffer,sizeof(realw)*mp->NGLOB_AB,cudaMemcpyDeviceToDevice,mp->compute_stream);

    cudaEventRecord(mp->transfer_is_complete2,mp->compute_stream);
    cudaStreamWaitEvent(mp->copy_stream_no_backward,mp->transfer_is_complete2,0);
    cudaMemcpyAsync(mp->d_potential_acoustic_buffer,pot_buffer,sizeof(realw)*mp->NGLOB_AB,cudaMemcpyHostToDevice,mp->copy_stream_no_backward);
    cudaEventRecord(mp->transfer_is_complete1,mp->copy_stream_no_backward);
}

