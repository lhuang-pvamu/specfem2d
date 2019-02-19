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

#include "mesh_constants_cuda.h"

// KERNEL 2 - acoustic compute forces kernel

template<int FORWARD_OR_ADJOINT> __global__ void
Kernel_2_acoustic_impl(const int nb_blocks_to_compute,
        const int* d_ibool,
        const int* d_phase_ispec_inner_acoustic,
        const int num_phase_ispec_acoustic,
        const int d_iphase,
        realw_const_p d_potential_acoustic,
        realw_p d_potential_dot_dot_acoustic,
        realw_const_p d_b_potential_acoustic,
        realw_p d_b_potential_dot_dot_acoustic,
        const realw* d_xix, const realw* d_xiz,
        const realw* d_gammax,const realw* d_gammaz,
        realw_const_p d_hprime_xx,
        realw_const_p d_hprimewgll_xx,
        realw_const_p d_wxgll,
        const realw* d_rhostore){

    for(int bx=0; bx< nb_blocks_to_compute; bx++) {
        for(int tx=0; tx< NGLL2; tx++) {

            realw xixl,xizl,gammaxl,gammazl;
            realw dpotentialdxl,dpotentialdzl;
            realw rho_invl_times_jacobianl;
            realw sum_terms;

            __shared__ realw s_dummy_loc[2*NGLL2];
            __shared__ realw s_temp1[NGLL2];
            __shared__ realw s_temp3[NGLL2];
            __shared__ realw sh_hprime_xx[NGLL2];
            __shared__ realw sh_hprimewgll_xx[NGLL2];
            __shared__ realw sh_wxgll[NGLLX];

            if (bx >= nb_blocks_to_compute ) return;

            int offset = (d_phase_ispec_inner_acoustic[bx + num_phase_ispec_acoustic*(d_iphase-1)]-1)*NGLL2_PADDED + tx;

            int iglob = d_ibool[offset] - 1;

            // changing iglob indexing to match fortran row changes fast style
            s_dummy_loc[tx] = d_potential_acoustic[iglob];
            if (nb_field==2) s_dummy_loc[NGLL2+tx]=d_b_potential_acoustic[iglob];

            int J = (tx/NGLLX);
            int I = (tx-J*NGLLX);

            xixl = get_global_cr( &d_xix[offset] );
            xizl = d_xiz[offset];
            gammaxl = d_gammax[offset];
            gammazl = d_gammaz[offset];

            rho_invl_times_jacobianl = 1.f /(d_rhostore[offset] * (xixl*gammazl-gammaxl*xizl));

            sh_hprime_xx[tx] = d_hprime_xx[tx];
            sh_hprimewgll_xx[tx] = d_hprimewgll_xx[tx];

            if (threadIdx.x < NGLLX){
                sh_wxgll[tx] = d_wxgll[tx];
            }
            for (int k=0 ; k < nb_field ; k++) {
                __syncthreads();

                realw temp1l = 0.f;
                realw temp3l = 0.f;

                for (int l=0;l<NGLLX;l++) {
                    temp1l += s_dummy_loc[NGLL2*k+J*NGLLX+l] * sh_hprime_xx[l*NGLLX+I];
                    temp3l += s_dummy_loc[NGLL2*k+l*NGLLX+I] * sh_hprime_xx[l*NGLLX+J];
                }

                dpotentialdxl = xixl*temp1l +  gammaxl*temp3l;
                dpotentialdzl = xizl*temp1l +  gammazl*temp3l;

                s_temp1[tx] = sh_wxgll[J]*rho_invl_times_jacobianl  * (dpotentialdxl*xixl  + dpotentialdzl*xizl)  ;
                s_temp3[tx] = sh_wxgll[I]*rho_invl_times_jacobianl  * (dpotentialdxl*gammaxl + dpotentialdzl*gammazl)  ;

                __syncthreads();

                sum_terms = 0.f;
                for (int l=0;l<NGLLX;l++) {
                    sum_terms -= s_temp1[J*NGLLX+l] * sh_hprimewgll_xx[I*NGLLX+l] + s_temp3[l*NGLLX+I] * sh_hprimewgll_xx[J*NGLLX+l];
                }

                if (k==0) {
                    atomicAdd(&d_potential_dot_dot_acoustic[iglob],sum_terms);
                } else {
                    atomicAdd(&d_b_potential_dot_dot_acoustic[iglob],sum_terms);
                }
            }
        }
    }
}

// KERNEL 2 - viscoacoustic compute forces kernel
template<int FORWARD_OR_ADJOINT> __global__ void
Kernel_2_viscoacoustic_impl(const int nb_blocks_to_compute,
        const int* d_ibool,
        const int* d_phase_ispec_inner_acoustic,
        const int num_phase_ispec_acoustic,
        const int d_iphase,
        realw_const_p d_potential_acoustic,
        realw_p d_potential_dot_dot_acoustic,
        const realw* d_xix, const realw* d_xiz,
        const realw* d_gammax,const realw* d_gammaz,
        realw_const_p d_hprime_xx,
        realw_const_p d_hprimewgll_xx,
        realw_const_p d_wxgll,
        const realw* d_rhostore,
        realw_p d_e1_acous,
        const realw* d_A_newmark,
        const realw* d_B_newmark,
        realw_p d_sum_forces_old){

/*
    // block-id == number of local element id in phase_ispec array
    int bx = blockIdx.y*gridDim.x+blockIdx.x;
    int tx = threadIdx.x;
    int iglob,offset,offset_align;

    realw temp1l,temp3l;
    realw xixl,xizl,gammaxl,gammazl;
    realw dpotentialdxl,dpotentialdzl;
    realw rho_invl_times_jacobianl;
    realw sum_terms;
    realw sum_forces_old,forces_attenuation,a_newmark;
    realw e1_acous_load[N_SLS];

    __shared__ realw s_dummy_loc[NGLL2];
    __shared__ realw s_temp1[NGLL2];
    __shared__ realw s_temp3[NGLL2];
    __shared__ realw sh_hprime_xx[NGLL2];
    __shared__ realw sh_hprimewgll_xx[NGLL2];
    __shared__ realw sh_wxgll[NGLLX];

    if (bx >= nb_blocks_to_compute ) return;

    int I =d_phase_ispec_inner_acoustic[bx + num_phase_ispec_acoustic*(d_iphase-1)]-1;
    offset = I*NGLL2_PADDED + tx;
    offset_align = I*NGLL2 + tx;
    iglob = d_ibool[offset] - 1;

    s_dummy_loc[tx] = d_potential_acoustic[iglob];

    // local index
    int J = (tx/NGLLX);
    I = (tx-J*NGLLX);

    xixl = get_global_cr( &d_xix[offset] );
    xizl = d_xiz[offset];
    gammaxl = d_gammax[offset];
    gammazl = d_gammaz[offset];

    rho_invl_times_jacobianl = 1.f /(d_rhostore[offset] * (xixl*gammazl-gammaxl*xizl));

    for (int i_sls=0;i_sls<N_SLS;i_sls++)  
        e1_acous_load[i_sls] = d_e1_acous[N_SLS*offset_align+i_sls];

    sh_hprime_xx[tx] = d_hprime_xx[tx];
    // loads hprimewgll into shared memory
    sh_hprimewgll_xx[tx] = d_hprimewgll_xx[tx];

    if (threadIdx.x < NGLLX){
        sh_wxgll[tx] = d_wxgll[tx];
    }

    __syncthreads();

    // computes first matrix product
    temp1l = 0.f;
    temp3l = 0.f;

    for (int l=0;l<NGLLX;l++) {
        //assumes that hprime_xx = hprime_yy = hprime_zz
        // 1. cut-plane along xi-direction
        temp1l += s_dummy_loc[J*NGLLX+l] * sh_hprime_xx[l*NGLLX+I];
        // 3. cut-plane along gamma-direction
        temp3l += s_dummy_loc[l*NGLLX+I] * sh_hprime_xx[l*NGLLX+J];
    }

    dpotentialdxl = xixl*temp1l +  gammaxl*temp3l;
    dpotentialdzl = xizl*temp1l +  gammazl*temp3l;
    s_temp1[tx] = sh_wxgll[J]*rho_invl_times_jacobianl  * (dpotentialdxl*xixl  + dpotentialdzl*xizl)  ;
    s_temp3[tx] = sh_wxgll[I]*rho_invl_times_jacobianl  * (dpotentialdxl*gammaxl + dpotentialdzl*gammazl)  ;

    __syncthreads();

    sum_terms = 0.f;
    for (int l=0;l<NGLLX;l++) {
        //assumes hprimewgll_xx = hprimewgll_zz
        sum_terms -= s_temp1[J*NGLLX+l] * sh_hprimewgll_xx[I*NGLLX+l] + s_temp3[l*NGLLX+I] * sh_hprimewgll_xx[J*NGLLX+l];
    }

    sum_forces_old = d_sum_forces_old[offset_align];
    forces_attenuation = 0.f;

    for (int i_sls=0;i_sls<N_SLS;i_sls++){
        a_newmark = d_A_newmark[N_SLS * offset_align + i_sls];
        e1_acous_load[i_sls] = a_newmark * a_newmark * e1_acous_load[i_sls] + d_B_newmark[N_SLS * offset_align + i_sls] * (sum_terms + a_newmark * sum_forces_old);
        forces_attenuation += e1_acous_load[i_sls];
        d_e1_acous[N_SLS*offset_align+i_sls] = e1_acous_load[i_sls];
    }

    d_sum_forces_old[offset_align] = sum_terms;
    sum_terms += forces_attenuation;

    atomicAdd(&d_potential_dot_dot_acoustic[iglob],sum_terms);
    */
}

void Kernel_2_acoustic( int nb_blocks_to_compute, Mesh* mp, int d_iphase,
                        int* d_ibool,
                        realw* d_xix,realw* d_xiz,
                        realw* d_gammax,realw* d_gammaz,
                        realw* d_rhostore,
                        int ATTENUATION_VISCOACOUSTIC,
                        int compute_wavefield_1,
                        int compute_wavefield_2) {

    if (compute_wavefield_1 && compute_wavefield_2){
        nb_field=2;
    }else{
        nb_field=1;
    }
    if ( ! ATTENUATION_VISCOACOUSTIC){
        if (!compute_wavefield_1 && compute_wavefield_2){
            Kernel_2_acoustic_impl<3>(nb_blocks_to_compute,
                    d_ibool,
                    mp->d_phase_ispec_inner_acoustic,
                    mp->num_phase_ispec_acoustic,
                    d_iphase,
                    mp->d_b_potential_acoustic, mp->d_b_potential_dot_dot_acoustic,
                    mp->d_b_potential_acoustic,mp->d_b_potential_dot_dot_acoustic,
                    nb_field,
                    d_xix, d_xiz,
                    d_gammax, d_gammaz,
                    mp->d_hprime_xx,
                    mp->d_hprimewgll_xx,
                    mp->d_wxgll,
                    d_rhostore);
        } else {
            Kernel_2_acoustic_impl<1>(nb_blocks_to_compute,
                    d_ibool,
                    mp->d_phase_ispec_inner_acoustic,
                    mp->num_phase_ispec_acoustic,
                    d_iphase,
                    mp->d_potential_acoustic, mp->d_potential_dot_dot_acoustic,
                    mp->d_b_potential_acoustic,mp->d_b_potential_dot_dot_acoustic,
                    nb_field,
                    d_xix, d_xiz,
                    d_gammax, d_gammaz,
                    mp->d_hprime_xx,
                    mp->d_hprimewgll_xx,
                    mp->d_wxgll,
                    d_rhostore);
        }
    }else{ // ATTENUATION_VISCOACOUSTIC== .true. below
        if (compute_wavefield_1) {
            Kernel_2_viscoacoustic_impl<1>(nb_blocks_to_compute,
                    d_ibool,
                    mp->d_phase_ispec_inner_acoustic,
                    mp->num_phase_ispec_acoustic,
                    d_iphase,
                    mp->d_potential_acoustic, mp->d_potential_dot_dot_acoustic,
                    d_xix, d_xiz,
                    d_gammax, d_gammaz,
                    mp->d_hprime_xx,
                    mp->d_hprimewgll_xx,
                    mp->d_wxgll,
                    d_rhostore,
                    mp->d_e1_acous,
                    mp->d_A_newmark_acous,
                    mp->d_B_newmark_acous,
                    mp->d_sum_forces_old);
        }
        if (compute_wavefield_2) {
            Kernel_2_viscoacoustic_impl<3>(nb_blocks_to_compute,
                    d_ibool,
                    mp->d_phase_ispec_inner_acoustic,
                    mp->num_phase_ispec_acoustic,
                    d_iphase,
                    mp->d_b_potential_acoustic, mp->d_b_potential_dot_dot_acoustic,
                    d_xix, d_xiz,
                    d_gammax, d_gammaz,
                    mp->d_hprime_xx,
                    mp->d_hprimewgll_xx,
                    mp->d_wxgll,
                    d_rhostore,
                    mp->d_b_e1_acous,
                    mp->d_A_newmark_acous,
                    mp->d_B_newmark_acous,
                    mp->d_b_sum_forces_old);
        }
    } // ATTENUATION_VISCOACOUSTIC
}

// main compute_forces_acoustic CUDA routine
extern "C"
void compute_forces_acoustic_omp( long* Mesh_pointer, int* iphase, int* nspec_outer_acoustic,
                                  int* nspec_inner_acoustic, int* ATTENUATION_VISCOACOUSTIC,
                                  int* compute_wavefield_1, int* compute_wavefield_2) {

            Mesh* mp = (Mesh*)(*Mesh_pointer); // get Mesh from fortran integer wrapper
            int num_elements;

            if (*iphase == 1)
                num_elements = *nspec_outer_acoustic;
            else
                num_elements = *nspec_inner_acoustic;
            if (num_elements == 0) 
                return;

            // no mesh coloring: uses atomic updates
            Kernel_2_acoustic(num_elements, mp, *iphase,
                              mp->d_ibool,
                              mp->d_xix,mp->d_xiz,
                              mp->d_gammax,mp->d_gammaz,
                              mp->d_rhostore,
                              *ATTENUATION_VISCOACOUSTIC,
                              *compute_wavefield_1,
                              *compute_wavefield_2);
}
/*

// KERNEL for enforce free surface 
__global__ void enforce_free_surface_cuda_kernel(realw_p potential_acoustic,
        realw_p potential_dot_acoustic,
        realw_p potential_dot_dot_acoustic,
        const int num_free_surface_faces,
        const int* free_surface_ispec,
        const int* free_surface_ij,
        const int* d_ibool,
        const int* ispec_is_acoustic) {
    // gets spectral element face id
    int iface = blockIdx.x + gridDim.x*blockIdx.y;

    // for all faces on free surface
    if (iface < num_free_surface_faces) {

        int ispec = free_surface_ispec[iface]-1;

        // checks if element is in acoustic domain
        if (ispec_is_acoustic[ispec]) {

            // gets global point index
            int igll = threadIdx.x + threadIdx.y*blockDim.x;

            int i = free_surface_ij[INDEX3(NDIM,NGLLX,0,igll,iface)] - 1; // (1,igll,iface)
            int j = free_surface_ij[INDEX3(NDIM,NGLLX,1,igll,iface)] - 1;

            int iglob = d_ibool[INDEX3_PADDED(NGLLX,NGLLX,i,j,ispec)] - 1;

            // sets potentials to zero at free surface
            potential_acoustic[iglob] = 0.f;
            potential_dot_acoustic[iglob] = 0.f;
            potential_dot_dot_acoustic[iglob] = 0.f;
        }
    }
}

extern "C"
void acoustic_enforce_free_surf_cuda(long* Mesh_pointer,int* compute_wavefield_1,int* compute_wavefield_2) {

    Mesh* mp = (Mesh*)(*Mesh_pointer); //get mesh pointer out of fortran integer container

    // does not absorb free surface, thus we enforce the potential to be zero at surface

    // checks if anything to do
    if (mp->num_free_surface_faces == 0) return;

    // block sizes
    int num_blocks_x, num_blocks_y;
    get_blocks_xy(mp->num_free_surface_faces,&num_blocks_x,&num_blocks_y);

    dim3 grid(num_blocks_x,num_blocks_y,1);
    dim3 threads(NGLLX,1,1);


    // sets potentials to zero at free surface
    if (*compute_wavefield_1) {
        enforce_free_surface_cuda_kernel<<<grid,threads,0,mp->compute_stream>>>(mp->d_potential_acoustic,
                mp->d_potential_dot_acoustic,
                mp->d_potential_dot_dot_acoustic,
                mp->num_free_surface_faces,
                mp->d_free_surface_ispec,
                mp->d_free_surface_ijk,
                mp->d_ibool,
                mp->d_ispec_is_acoustic);
    }
    // for backward/reconstructed potentials
    if (*compute_wavefield_2) {
        enforce_free_surface_cuda_kernel<<<grid,threads,0,mp->compute_stream>>>(mp->d_b_potential_acoustic,
                mp->d_b_potential_dot_acoustic,
                mp->d_b_potential_dot_dot_acoustic,
                mp->num_free_surface_faces,
                mp->d_free_surface_ispec,
                mp->d_free_surface_ijk,
                mp->d_ibool,
                mp->d_ispec_is_acoustic);
    }
}
*/
