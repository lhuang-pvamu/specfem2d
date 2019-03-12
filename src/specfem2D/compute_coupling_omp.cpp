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


void
compute_coupling_acoustic_el_kernel( realw* displ,
                                     realw* potential_dot_dot_acoustic,
                                     int num_coupling_ac_el_faces,
                                     int* coupling_ac_el_ispec,
                                     int* coupling_ac_el_ij,
                                     realw* coupling_ac_el_normal,
                                     realw* coupling_ac_el_jacobian1Dw,
                                     int* d_ibool)
{
    for(int iface=0; iface < num_coupling_ac_el_faces; i++) {
        for(int igll=0; igll<NGLLX; igll++) {
            int ispec = coupling_ac_el_ispec[iface] - 1;
            int i = coupling_ac_el_ij[INDEX3(NDIM,NGLLX,0,igll,iface)] - 1;
            int j = coupling_ac_el_ij[INDEX3(NDIM,NGLLX,1,igll,iface)] - 1;
            int iglob = d_ibool[INDEX3_PADDED(NGLLX,NGLLX,i,j,ispec)] - 1;
            realw displ_x = displ[iglob*2] ; // (1,iglob)
            realw displ_z = displ[iglob*2+1] ; // (2,iglob)
            realw nx = coupling_ac_el_normal[INDEX3(NDIM,NGLLX,0,igll,iface)]; // (1,igll,iface)
            realw nz = coupling_ac_el_normal[INDEX3(NDIM,NGLLX,1,igll,iface)]; // (2,igll,iface)
            realw displ_n = displ_x*nx + displ_z*nz;
            realw jacobianw = coupling_ac_el_jacobian1Dw[INDEX2(NGLLX,igll,iface)];
            //atomicAdd
            potential_dot_dot_acoustic[iglob] += jacobianw*displ_n;
        }
    }
}

extern "C"
void compute_coupling_ac_el_omp( long* Mesh_pointer, int* iphasef,
                                 int* num_coupling_ac_el_facesf )
{
    Mesh* mp = (Mesh*)(*Mesh_pointer); //get mesh pointer out of fortran integer container
    int iphase            = *iphasef;
    // only add this contribution for first pass
    if (iphase != 1) return;
    //int num_coupling_ac_el_faces  = *num_coupling_ac_el_facesf;

    compute_coupling_acoustic_el_kernel( mp->d_displ,
                                         mp->d_potential_dot_dot_acoustic,
                                         *num_coupling_ac_el_facesf,
                                         mp->d_coupling_ac_el_ispec,
                                         mp->d_coupling_ac_el_ijk,
                                         mp->d_coupling_ac_el_normal,
                                         mp->d_coupling_ac_el_jacobian2Dw,
                                         mp->d_ibool);
    //  adjoint simulations
    if (mp->simulation_type == 3) {
        compute_coupling_acoustic_el_kernel( mp->d_b_displ,
                                             mp->d_b_potential_dot_dot_acoustic,
                                             num_coupling_ac_el_faces,
                                             mp->d_coupling_ac_el_ispec,
                                             mp->d_coupling_ac_el_ijk,
                                             mp->d_coupling_ac_el_normal,
                                             mp->d_coupling_ac_el_jacobian2Dw,
                                             mp->d_ibool);
    }
}

// ELASTIC - ACOUSTIC coupling
void
compute_coupling_elastic_ac_kernel( realw* potential_dot_dot_acoustic,
                                    realw* accel,
                                    int num_coupling_ac_el_faces,
                                    int* coupling_ac_el_ispec,
                                    int* coupling_ac_el_ij,
                                    realw* coupling_ac_el_normal,
                                    realw* coupling_ac_el_jacobian1Dw,
                                    int* d_ibool)
{
    for(int iface=0; iface < num_coupling_ac_el_faces; i++){
        for(int igll=0; igll<NGLLX; igll++) {
            // "-1" from index values to convert from Fortran-> C indexing
            int ispec = coupling_ac_el_ispec[iface] - 1;
            int i = coupling_ac_el_ij[INDEX3(NDIM,NGLLX,0,igll,iface)] - 1;
            int j = coupling_ac_el_ij[INDEX3(NDIM,NGLLX,1,igll,iface)] - 1;
            int iglob = d_ibool[INDEX3_PADDED(NGLLX,NGLLX,i,j,ispec)] - 1;
            realw nx = coupling_ac_el_normal[INDEX3(NDIM,NGLLX,0,igll,iface)]; // (1,igll,iface)
            realw nz = coupling_ac_el_normal[INDEX3(NDIM,NGLLX,1,igll,iface)]; // (2,igll,iface)
            realw jacobianw = coupling_ac_el_jacobian1Dw[INDEX2(NGLLX,igll,iface)];
            realw pressure = - potential_dot_dot_acoustic[iglob];
            //atomicAdd
            accel[iglob*2] += jacobianw*nx*pressure;
            //atomicAdd
            accel[iglob*2+1] += jacobianw*nz*pressure;
        }
    }
}

extern "C"
void compute_coupling_el_ac_omp( long* Mesh_pointer,
                                 int* iphasef,
                                 int* num_coupling_ac_el_facesf)
{
    Mesh* mp = (Mesh*)(*Mesh_pointer); //get mesh pointer out of fortran integer container
    //int iphase            = *iphasef;
    // only add this contribution for first pass
    if (*iphasef != 1) return;
    int num_coupling_ac_el_faces  = *num_coupling_ac_el_facesf;
    compute_coupling_elastic_ac_kernel( mp->d_potential_dot_dot_acoustic,
                                        mp->d_accel,
                                        num_coupling_ac_el_faces,
                                        mp->d_coupling_ac_el_ispec,
                                        mp->d_coupling_ac_el_ijk,
                                        mp->d_coupling_ac_el_normal,
                                        mp->d_coupling_ac_el_jacobian2Dw,
                                        mp->d_ibool);
    //  adjoint simulations
    if (mp->simulation_type == 3) {
        compute_coupling_elastic_ac_kernel( mp->d_b_potential_dot_dot_acoustic,
                                            mp->d_b_accel,
                                            num_coupling_ac_el_faces,
                                            mp->d_coupling_ac_el_ispec,
                                            mp->d_coupling_ac_el_ijk,
                                            mp->d_coupling_ac_el_normal,
                                            mp->d_coupling_ac_el_jacobian2Dw,
                                            mp->d_ibool);
    }
}
