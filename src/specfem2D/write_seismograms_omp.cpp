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
#include <cstdio>
#include <cstring>


using std::printf;


// ELASTIC simulations
void compute_elastic_seismogram_kernel_omp(int nrec_local,
                                       realw* field,
                                       int* d_ibool,
                                       realw* hxir, realw* hgammar,
                                       realw* seismograms,
                                       realw* cosrot,
                                       realw* sinrot,
                                       int* ispec_selected_rec_loc,
                                       int it,
                                       int NSTEP)
{

    // __shared__
    realw sh_dxd[NGLL2_PADDED];
    realw sh_dzd[NGLL2_PADDED];

    for( int irec_local=0; irec_local < nrec_local; irec_local++) {
        //Should this be one loop over NGLL or 2?
        for(int tx = 0; tx < NGLLX; tx++) {
            int J = (tx/NGLLX);
            int I = (tx-J*NGLLX);
            int ispec = ispec_selected_rec_loc[irec_local] - 1;
            sh_dxd[tx] = 0;
            sh_dzd[tx] = 0;
            if (tx < NGLL2) {
                realw hlagrange = hxir[irec_local + nrec_local*I] * hgammar[irec_local + nrec_local*J];
                int iglob = d_ibool[tx+NGLL2_PADDED*ispec] - 1;
        
                sh_dxd[tx] = hlagrange * field[0 + 2*iglob];
                sh_dzd[tx] = hlagrange * field[1 + 2*iglob];
            }
        //__syncthreads();

        // reduction
            for (int s=1; s<NGLL2_PADDED ; s *= 2) {
            if (tx % (2*s) == 0) {
                    sh_dxd[tx] += sh_dxd[tx + s];
                    sh_dzd[tx] += sh_dzd[tx + s];
                }
                //__syncthreads();
            }
    
            if (tx == 0) {
                seismograms[irec_local*NSTEP + it]                    = cosrot[irec_local]*sh_dxd[0]  + sinrot[irec_local]*sh_dzd[0];
            }
            if (tx == 1) {
                seismograms[irec_local*NSTEP + it + nrec_local*NSTEP] = cosrot[irec_local]*sh_dzd[0]  - sinrot[irec_local]*sh_dxd[0];
            }
        }
    }
}

void compute_acoustic_seismogram_kernel_omp(int nrec_local,
                                        realw* pressure,
                                        int* d_ibool,
                                        realw* hxir, realw* hgammar,
                                        realw* seismograms,
                                        int* ispec_selected_rec_loc,
                                        int it,
                                        int NSTEP)
{
    //__shared__ 
    realw sh_dxd[NGLL2_PADDED];

    for( int irec_local=0; irec_local < nrec_local; irec_local++) {
        //Should this be one loop over NGLL or 2?
        for(int tx = 0; tx < NGLLX; tx++) {
            int J = (tx/NGLLX);
            int I = (tx-J*NGLLX);
            int ispec = ispec_selected_rec_loc[irec_local]-1;
            sh_dxd[tx] = 0;

            if (tx < NGLL2) {
                int iglob = d_ibool[tx+NGLL2_PADDED*ispec]-1;
                realw hlagrange = hxir[irec_local + nrec_local*I]*hgammar[irec_local + nrec_local*J];
                sh_dxd[tx] = hlagrange*pressure[iglob];
            }
            //__syncthreads();
            for (int s=1; s<NGLL2_PADDED ; s *= 2) {
                if (tx % (2*s) == 0) sh_dxd[tx] += sh_dxd[tx + s];
                //__syncthreads();
            }
            // Signe moins car pression = -potential_dot_dot
            if (tx == 0) {
                seismograms[irec_local*NSTEP + it ]                   = -sh_dxd[0];
            }
            if (tx == 1) {
                seismograms[irec_local*NSTEP + it + nrec_local*NSTEP] = 0;
            }
        }
    }
}

extern "C"
void compute_seismograms_omp_(long* Mesh_pointer_f,
                              int* seismotypef,
                              double* sisux, double* sisuz,
                              int* seismo_currentf,
                              int* NSTEPf,
                              int* ELASTIC_SIMULATION,
                              int* ACOUSTIC_SIMULATION,
                              int* USE_TRICK_FOR_BETTER_PRESSURE)
{

    Mesh* mp = (Mesh*)(*Mesh_pointer_f); // get Mesh from fortran integer wrapper

    if (mp->nrec_local == 0) return;

    int seismotype = *seismotypef;
    int seismo_current = *seismo_currentf - 1 ;
    int NSTEP = *NSTEPf;

    // warnings
    if (seismo_current == 0){
        switch (seismotype){
            case 1 :
                //Deplacement
                // warnings
                if (! *ELASTIC_SIMULATION)
                    printf("\nWarning: Wrong type of seismogram for a pure fluid simulation, use pressure in seismotype\n");
                if (*ELASTIC_SIMULATION && *ACOUSTIC_SIMULATION)
                    printf("\nWarning: Coupled elastic/fluid simulation has only valid displacement seismograms in elastic domain for GPU simulation\n\n");
                break;
            case 2 :
                //Vitesse
                if (! *ELASTIC_SIMULATION)
                    printf("\nWarning: Wrong type of seismogram for a pure fluid simulation, use pressure in seismotype\n");
                if (*ELASTIC_SIMULATION && *ACOUSTIC_SIMULATION)
                    printf("\nWarning: Coupled elastic/fluid simulation has only valid velocity seismograms in elastic domain for GPU simulation\n\n");
                break;
            case 3 :
                //Acceleration
                if (! *ELASTIC_SIMULATION)
                    printf("\nWarning: Wrong type of seismogram for a pure fluid simulation, use pressure in seismotype\n");
                if (*ELASTIC_SIMULATION && *ACOUSTIC_SIMULATION)
                    printf("\nWarning: Coupled elastic/fluid simulation has only valid acceleration seismograms in elastic domain for GPU simulation\n\n");
                break;
            case 4 :
                //Pression
                if (! *ACOUSTIC_SIMULATION)
                    printf("\nWarning: Wrong type of seismogram for a pure elastic simulation, use displ veloc or accel in seismotype\n");
                if (*ELASTIC_SIMULATION && *ACOUSTIC_SIMULATION)
                    printf("\nWarning: Coupled elastic/fluid simulation has only valid pressure seismograms in fluid domain for GPU simulation\n\n");
                break;
        }
    }

    // computes current seismograms value
    switch (seismotype){
        case 1 :
            //Deplacement
            compute_elastic_seismogram_kernel_omp(mp->nrec_local,
                                              mp->d_displ,
                                              mp->d_ibool,
                                              mp->d_xir_store_loc, mp->d_gammar_store_loc,
                                              mp->d_seismograms,
                                              mp->d_cosrot,
                                              mp->d_sinrot,
                                              mp->d_ispec_selected_rec_loc,
                                              seismo_current,
                                              NSTEP);
            break;
        case 2 :
            //Vitesse
            compute_elastic_seismogram_kernel_omp(mp->nrec_local,
                                              mp->d_veloc,
                                              mp->d_ibool,
                                              mp->d_xir_store_loc, mp->d_gammar_store_loc,
                                              mp->d_seismograms,
                                              mp->d_cosrot,
                                              mp->d_sinrot,
                                              mp->d_ispec_selected_rec_loc,
                                              seismo_current,
                                              NSTEP);
            break;
        case 3 :
            //Acceleration
            compute_elastic_seismogram_kernel_omp(mp->nrec_local,
                                              mp->d_accel,
                                              mp->d_ibool,
                                              mp->d_xir_store_loc, mp->d_gammar_store_loc,
                                              mp->d_seismograms,
                                              mp->d_cosrot,
                                              mp->d_sinrot,
                                              mp->d_ispec_selected_rec_loc,
                                              seismo_current,
                                              NSTEP);
            break;
        case 4 :
            //Pression
            if (*USE_TRICK_FOR_BETTER_PRESSURE){
                compute_acoustic_seismogram_kernel_omp(mp->nrec_local,
                                                   mp->d_potential_acoustic,
                                                   mp->d_ibool,
                                                   mp->d_xir_store_loc, mp->d_gammar_store_loc,
                                                   mp->d_seismograms,
                                                   mp->d_ispec_selected_rec_loc,
                                                   seismo_current,
                                                   NSTEP);
            } else {
                compute_acoustic_seismogram_kernel_omp(mp->nrec_local,
                                                   mp->d_potential_dot_dot_acoustic,
                                                   mp->d_ibool,
                                                   mp->d_xir_store_loc, mp->d_gammar_store_loc,
                                                   mp->d_seismograms,
                                                   mp->d_ispec_selected_rec_loc,
                                                   seismo_current,
                                                   NSTEP);
            }
            break;
    }

    if (seismo_current == NSTEP - 1 ) {
        int size = mp->nrec_local*NSTEP;
        std::memcpy(mp->h_seismograms,mp->d_seismograms,sizeof(realw)* 2 * size);

        for (int i=0;i < mp->nrec_local;i++) {
            for (int j=0;j<NSTEP;j++) {
                sisux[j + NSTEP * i ] = (double)*(mp->h_seismograms + j + NSTEP * i);
                sisuz[j + NSTEP * i ] = (double)*(mp->h_seismograms + j + NSTEP * i + size);
            }
        }
    }
}
