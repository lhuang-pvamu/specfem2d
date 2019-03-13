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


// acoustic sources
void compute_add_sources_acoustic_omp_kernel( realw* potential_dot_dot_acoustic,
                                              int* d_ibool,
                                              realw* sourcearrays,
                                              realw* source_time_function,
                                              int myrank,
                                              int* ispec_selected_source,
                                              int* ispec_is_acoustic,
                                              realw* kappastore,
                                              int it,int nsources_local)
{
    for(int isource=0; isource < nsources_local; isource++) {
        int ispec = ispec_selected_source[isource]-1;
        if (ispec_is_acoustic[ispec]) {
            for(int i=0; i<NGLLX; i++) {
                for(int j=0; j<NGLLX; j++) {
                    int iglob = d_ibool[INDEX3_PADDED(NGLLX,NGLLX,i,j,ispec)] - 1;
                    realw kappal = kappastore[INDEX3(NGLLX,NGLLX,i,j,ispec)];
                    realw stf = source_time_function[INDEX2(nsources_local,isource,it)]/kappal;
                    //atomicAdd
                    potential_dot_dot_acoustic[iglob] += sourcearrays[INDEX4(nsources_local,NDIM,NGLLX,isource, 0,i,j)]*stf;
                }
            }
        }
    }
}

extern "C"
void compute_add_sources_ac_omp_(long* Mesh_pointer, int* iphasef, int * itf)
{
    Mesh* mp = (Mesh*)(*Mesh_pointer); //get mesh pointer out of fortran integer container
    if (mp->nsources_local == 0) return;
    if (*iphasef != 1) return;
    compute_add_sources_acoustic_omp_kernel( mp->d_potential_dot_dot_acoustic,
                                             mp->d_ibool,
                                             mp->d_sourcearrays,
                                             mp->d_source_time_function,
                                             mp->myrank,
                                             mp->d_ispec_selected_source,
                                             mp->d_ispec_is_acoustic,
                                             mp->d_kappastore,
                                             (*itf) - 1,
                                             mp->nsources_local);
}

extern "C"
void compute_add_sources_ac_s3_omp_(long* Mesh_pointer, int* iphasef, int* itf)
{
    Mesh* mp = (Mesh*)(*Mesh_pointer); //get mesh pointer out of fortran integer container
    if (mp->nsources_local == 0) return;
    if (*iphasef != 1) return;
    compute_add_sources_acoustic_omp_kernel(mp->d_b_potential_dot_dot_acoustic,
                                        mp->d_ibool,
                                        mp->d_sourcearrays,
                                        mp->d_source_time_function,
                                        mp->myrank,
                                        mp->d_ispec_selected_source,
                                        mp->d_ispec_is_acoustic,
                                        mp->d_kappastore,
                                        (*itf) - 1,
                                        mp->nsources_local);
}

// acoustic adjoint sources
void add_sources_ac_SIM_TYPE_2_OR_3_omp_kernel( realw* potential_dot_dot_acoustic,
        realw* source_adjointe,
        realw* xir_store,
        realw* gammar_store,
        int* d_ibool,
        int* ispec_is_acoustic,
        int* ispec_selected_rec_loc,
        int it,
        int nadj_rec_local,
        realw* kappastore,
        int NSTEP )
{
    for(int irec_local=0; irec_local< nadj_rec_local; irec_local++){
        for(int i=0; i<NGLLX; i++) {
            for(int j=0; j<NGLLX; j++) {
                int ispec = ispec_selected_rec_loc[irec_local]-1;
                if (ispec_is_acoustic[ispec]) {
                    int iglob = d_ibool[INDEX3_PADDED(NGLLX,NGLLX,i,j,ispec)]-1;
                    realw  kappal = kappastore[INDEX3(NGLLX,NGLLX,i,j,ispec)];
                    realw  xir = xir_store[INDEX2(nadj_rec_local,irec_local,i)];
                    realw  gammar = gammar_store[INDEX2(nadj_rec_local,irec_local,j)];
                    realw  source_adj = source_adjointe[INDEX3(nadj_rec_local,NSTEP,irec_local,it,0)];
                    realw stf = source_adj * gammar * xir / kappal ;
                    //atomicAdd
                    potential_dot_dot_acoustic[iglob] += stf;
                }
            }
        }
    }
}


extern "C"
void add_sources_ac_sim_2_or_3_omp_(long* Mesh_pointer,
                                    int* iphasef,
                                    int* it,
                                    int* nadj_rec_local,
                                    int* NSTEP)
{
    Mesh* mp = (Mesh*)(*Mesh_pointer); //get mesh pointer out of fortran integer container
    if (*iphasef != 1) return;
    add_sources_ac_SIM_TYPE_2_OR_3_omp_kernel( mp->d_potential_dot_dot_acoustic,
                                           mp->d_source_adjointe,
                                           mp->d_xir_store_loc,
                                           mp->d_gammar_store_loc,
                                           mp->d_ibool,
                                           mp->d_ispec_is_acoustic,
                                           mp->d_ispec_selected_rec_loc,
                                           (*it) - 1,
                                           mp->nadj_rec_local,
                                           mp->d_kappastore,
                                           *NSTEP);
}
