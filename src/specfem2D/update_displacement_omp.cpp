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


// elastic wavefield
void UpdateDispVeloc_omp_kernel(realw* displ,
                                realw* veloc,
                                realw* accel,
                                int size,
                                realw deltat,
                                realw deltatsqover2,
                                realw deltatover2)
{
    for(int id=0; id < size; id++) {
        displ[id] = displ[id] + deltat*veloc[id] + deltatsqover2*accel[id];
        veloc[id] = veloc[id] + deltatover2*accel[id];
        accel[id] = 0.0f; // can do this using memset...not sure if faster,probably not
    }
}

extern "C"
void update_displacement_omp_(long* Mesh_pointer,
                              realw* deltat_F,
                              realw* deltatsqover2_F,
                              realw* deltatover2_F,
                              realw* b_deltat_F,
                              realw* b_deltatsqover2_F,
                              realw* b_deltatover2_F)
{
    Mesh* mp = (Mesh*)(*Mesh_pointer); // get Mesh from fortran integer wrapper
    int size = NDIM * mp->NGLOB_AB;

    UpdateDispVeloc_omp_kernel(mp->d_displ,mp->d_veloc,mp->d_accel,size,*deltat_F, *deltatsqover2_F, *deltatover2_F);

    // kernel for backward fields
    if (mp->simulation_type == 3) {
        UpdateDispVeloc_omp_kernel(mp->d_b_displ,mp->d_b_veloc,mp->d_b_accel,size, *b_deltat_F, *b_deltatsqover2_F, *b_deltatover2_F);
    }
}

// acoustic wavefield
// KERNEL 1
void UpdatePotential_omp_kernel(realw_p potential_acoustic,
                                       realw* potential_dot_acoustic,
                                       realw* potential_dot_dot_acoustic,
                                       int size,
                                       realw deltat,
                                       realw deltatsqover2,
                                       realw deltatover2) {

    for(int id=0; id < size; id++) {
        realw p_dot_dot = potential_dot_dot_acoustic[id];
        potential_acoustic[id] += deltat*potential_dot_acoustic[id] + deltatsqover2*p_dot_dot;
        potential_dot_acoustic[id] += deltatover2*p_dot_dot;
        potential_dot_dot_acoustic[id] = 0.0f;
    }
}

extern "C"
void update_displacement_ac_omp_(long* Mesh_pointer,
                                 realw* deltat_F,
                                 realw* deltatsqover2_F,
                                 realw* deltatover2_F,
                                 realw* b_deltat_F,
                                 realw* b_deltatsqover2_F,
                                 realw* b_deltatover2_F,
                                 int* compute_b_wavefield,
                                 int* UNDO_ATTENUATION_AND_OR_PML)
{
    Mesh* mp = (Mesh*)(*Mesh_pointer); // get Mesh from fortran integer wrapper

    int size = mp->NGLOB_AB;
    // forward wavefields
    realw deltat = *deltat_F;
    realw deltatsqover2 = *deltatsqover2_F;
    realw deltatover2 = *deltatover2_F;

    if(!(*UNDO_ATTENUATION_AND_OR_PML && *compute_b_wavefield)) {
        UpdatePotential_omp_kernel(mp->d_potential_acoustic,
                               mp->d_potential_dot_acoustic,
                               mp->d_potential_dot_dot_acoustic,
                               size,deltat,deltatsqover2,deltatover2);
    }
    // backward/reconstructed wavefields
    if (mp->simulation_type == 3 && *compute_b_wavefield) {
        realw b_deltat = *b_deltat_F;
        realw b_deltatsqover2 = *b_deltatsqover2_F;
        realw b_deltatover2 = *b_deltatover2_F;
        UpdatePotential_omp_kernel(mp->d_b_potential_acoustic,
                               mp->d_b_potential_dot_acoustic,
                               mp->d_b_potential_dot_dot_acoustic,
                               size,b_deltat,b_deltatsqover2,b_deltatover2);
    }
}

// elastic domains
// KERNEL 3
void kernel_3_omp_device( realw* veloc,
                          realw* accel,
                          int size,
                          realw deltatover2,
                          realw* rmassx,
                          realw* rmassz)
{
    //int id = threadIdx.x + blockIdx.x*blockDim.x + blockIdx.y*gridDim.x*blockDim.x;
    //if (id < size) {
    for(int id=0; id < size; id++) {
        accel[2*id] = accel[2*id]*rmassx[id];
        accel[2*id+1] = accel[2*id+1]*rmassz[id];

        veloc[2*id] = veloc[2*id] + deltatover2*accel[2*id];
        veloc[2*id+1] = veloc[2*id+1] + deltatover2*accel[2*id+1];
    }
}

void kernel_3_accel_omp_device( realw* accel,
                                int size,
                                realw* rmassx,
                                realw* rmassy,
                                realw* rmassz)
{
    for(int id=0; id < size; id++) {
        accel[2*id] = accel[2*id]*rmassx[id];
        accel[2*id+1] = accel[2*id+1]*rmassz[id];
    }
}

void kernel_3_veloc_omp_device( realw* veloc, 
                                realw* accel, 
                                int size, 
                                realw deltatover2)
{
    for(int id=0; id < size; id++) {
        veloc[2*id] = veloc[2*id] + deltatover2*accel[2*id];
        veloc[2*id+1] = veloc[2*id+1] + deltatover2*accel[2*id+1];
    }
}

extern "C"
void kernel_3_a_omp_(long* Mesh_pointer, realw* deltatover2_F, realw* b_deltatover2_F)
{
    Mesh* mp = (Mesh*)(*Mesh_pointer); // get Mesh from fortran integer wrapper
    int size = mp->NGLOB_AB;
    realw deltatover2 = *deltatover2_F;
    kernel_3_omp_device(mp->d_veloc, mp->d_accel, size, deltatover2, mp->d_rmassx,mp->d_rmassz);
    if (mp->simulation_type == 3) {
        realw b_deltatover2 = *b_deltatover2_F;
        kernel_3_omp_device(mp->d_b_veloc, mp->d_b_accel, size, b_deltatover2, mp->d_rmassx,mp->d_rmassz);
    }
}

extern "C"
void kernel_3_b_omp_(long* Mesh_pointer,
                     realw* deltatover2_F,
                     realw* b_deltatover2_F) 
{
    Mesh* mp = (Mesh*)(*Mesh_pointer); // get Mesh from fortran integer wrapper
    int size = mp->NGLOB_AB;
    kernel_3_veloc_omp_device(mp->d_veloc, mp->d_accel, size, *deltatover2_F);
    if (mp->simulation_type == 3) {
        kernel_3_veloc_omp_device(mp->d_b_veloc, mp->d_b_accel, size, *b_deltatover2_F);
    }
}

// acoustic domains
// KERNEL 3
void kernel_3_acoustic_omp_device(realw* potential_dot_dot_acoustic,
                                   realw* b_potential_dot_dot_acoustic,
                                   realw* potential_dot_acoustic,
                                   realw* b_potential_dot_acoustic,
                                   int size,
                                   int compute_wavefield_1,
                                   int compute_wavefield_2,
                                   realw deltatover2,
                                   realw b_deltatover2,
                                   realw* rmass_acoustic)
{
    for(int id=0; id < size; id++) {
        realw rmass = rmass_acoustic[id];
        if (compute_wavefield_1) {
            realw p_dot_dot = potential_dot_dot_acoustic[id]*rmass;
            potential_dot_dot_acoustic[id] = p_dot_dot;
            potential_dot_acoustic[id] += deltatover2*p_dot_dot;
        }

        if (compute_wavefield_2){
            realw p_dot_dot = b_potential_dot_dot_acoustic[id]*rmass;
            b_potential_dot_dot_acoustic[id] = p_dot_dot;
            b_potential_dot_acoustic[id] += b_deltatover2*p_dot_dot;
        }
    } // id<size
}

extern "C"
void kernel_3_acoustic_omp_(long* Mesh_pointer,
                            realw* deltatover2,
                            realw* b_deltatover2,
                            int* compute_wavefield_1,
                            int* compute_wavefield_2)
{
    Mesh* mp = (Mesh*)(*Mesh_pointer); // get Mesh from fortran integer wrapper
    int size = mp->NGLOB_AB;
    kernel_3_acoustic_omp_device(mp->d_potential_dot_dot_acoustic,
                                 mp->d_b_potential_dot_dot_acoustic,
                                 mp->d_potential_dot_acoustic,
                                 mp->d_b_potential_dot_acoustic,
                                 size,
                                 *compute_wavefield_1,
                                 *compute_wavefield_2,
                                 *deltatover2,
                                 *b_deltatover2,
                                 mp->d_rmass_acoustic);
}


