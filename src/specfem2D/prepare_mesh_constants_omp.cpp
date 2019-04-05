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
#include "prepare_constants_omp.h"


// copies integer array from CPU host to GPU device
void copy_todevice_int(void** d_array_addr_ptr,int* h_array,int size)
{
    malloc((void**)d_array_addr_ptr,size*sizeof(int));
    memcpy((int*) *d_array_addr_ptr,h_array,size*sizeof(int),memcpyHostToDevice);
}

// copies integer array from CPU host to GPU device
void copy_todevice_realw(void** d_array_addr_ptr,realw* h_array,int size){

  malloc((void**)d_array_addr_ptr,size*sizeof(realw));
  memcpy((realw*) *d_array_addr_ptr,h_array,size*sizeof(realw),memcpyHostToDevice);
}

extern "C"
void prepare_constants_device(long* Mesh_pointer,
                              int* h_NGLLX, int* NSPEC_AB, int* NGLOB_AB,
                              realw* h_xix, realw* h_xiz,
                              realw* h_gammax, realw* h_gammaz,
                              realw* h_kappav, realw* h_muv,
                              int* h_ibool,
                              int* num_interfaces_ext_mesh, int* max_nibool_interfaces_ext_mesh,
                              int* h_nibool_interfaces_ext_mesh, int* h_ibool_interfaces_ext_mesh,
                              realw* h_hprime_xx, realw* h_hprimewgll_xx,
                              realw* h_wxgll,
                              int* ABSORBING_CONDITIONS,
                              int* h_nspec_bottom,
                              int* h_nspec_left,
                              int* h_nspec_right,
                              int* h_nspec_top,
                              int* h_abs_boundary_ispec, int* h_abs_boundary_ij,
                              realw* h_abs_boundary_normal,
                              realw* h_abs_boundary_jacobian1Dw,
                              int* h_num_abs_boundary_faces,
                              int* h_cote_abs,
                              int* h_ib_bottom,
                              int* h_ib_left,
                              int* h_ib_right,
                              int* h_ib_top,
                              int* h_ispec_is_inner,
                              int* nsources_local_f,
                              realw* h_sourcearrays, realw * h_source_time_function,
                              int* NSTEP,
                              int* h_ispec_selected_source,
                              int* h_ispec_selected_rec_loc,
                              int* nrec_local,
                              realw * h_cosrot,realw * h_sinrot,
                              int* SIMULATION_TYPE,
                              int* nspec_acoustic,int* nspec_elastic,
                              int* h_myrank,
                              int* SAVE_FORWARD,
                              realw* h_xir_store, realw* h_gammar_store)
{
    Mesh* mp = (Mesh*) malloc( sizeof(Mesh) );
    if (mp == NULL) exit_on_error("error allocating mesh pointer");
    *Mesh_pointer = (long)mp;

    if (*h_NGLLX != NGLLX) {
        exit_on_error("NGLLX defined in constants.h must match the NGLLX defined in src/cuda/mesh_constants_cuda.h");
    }

    mp->myrank = *h_myrank;

    // sets global parameters
    mp->NSPEC_AB = *NSPEC_AB;
    mp->NGLOB_AB = *NGLOB_AB;

    // constants
    mp->simulation_type = *SIMULATION_TYPE;
    mp->absorbing_conditions = *ABSORBING_CONDITIONS;
    mp->save_forward = *SAVE_FORWARD;

    // sets constant arrays
    setConst_hprime_xx(h_hprime_xx,mp);

    setConst_hprimewgll_xx(h_hprimewgll_xx,mp);

    setConst_wxgll(h_wxgll,mp);

    // Assuming NGLLX=5. Padded is then 32 (5^2+3)
    int size_padded = NGLL2_PADDED * (mp->NSPEC_AB);

    malloc((void**) &mp->d_xix, size_padded*sizeof(realw));
    malloc((void**) &mp->d_xiz, size_padded*sizeof(realw));
    malloc((void**) &mp->d_gammax, size_padded*sizeof(realw));
    malloc((void**) &mp->d_gammaz, size_padded*sizeof(realw));
    malloc((void**) &mp->d_kappav, size_padded*sizeof(realw));
    malloc((void**) &mp->d_muv, size_padded*sizeof(realw));

    memcpy2D(mp->d_xix, NGLL2_PADDED*sizeof(realw),
                h_xix, NGLL2*sizeof(realw), NGLL2*sizeof(realw),
                mp->NSPEC_AB, memcpyHostToDevice);
    memcpy2D(mp->d_xiz, NGLL2_PADDED*sizeof(realw),
                h_xiz, NGLL2*sizeof(realw), NGLL2*sizeof(realw),
                mp->NSPEC_AB, memcpyHostToDevice);
    memcpy2D(mp->d_gammax, NGLL2_PADDED*sizeof(realw),
                h_gammax, NGLL2*sizeof(realw), NGLL2*sizeof(realw),
                mp->NSPEC_AB, memcpyHostToDevice);
    memcpy2D(mp->d_gammaz, NGLL2_PADDED*sizeof(realw),
                h_gammaz, NGLL2*sizeof(realw), NGLL2*sizeof(realw),
                mp->NSPEC_AB, memcpyHostToDevice);
    memcpy2D(mp->d_kappav, NGLL2_PADDED*sizeof(realw),
                h_kappav, NGLL2*sizeof(realw), NGLL2*sizeof(realw),
                mp->NSPEC_AB, memcpyHostToDevice);
    memcpy2D(mp->d_muv, NGLL2_PADDED*sizeof(realw),
                h_muv, NGLL2*sizeof(realw), NGLL2*sizeof(realw),
                mp->NSPEC_AB, memcpyHostToDevice);

    // global indexing (padded)
    malloc((void**) &mp->d_ibool, size_padded*sizeof(int));
    memcpy2D(mp->d_ibool, NGLL2_PADDED*sizeof(int),
                h_ibool, NGLL2*sizeof(int), NGLL2*sizeof(int),
                mp->NSPEC_AB, memcpyHostToDevice);

    // prepare interprocess-edge exchange information
    mp->num_interfaces_ext_mesh = *num_interfaces_ext_mesh;
    mp->max_nibool_interfaces_ext_mesh = *max_nibool_interfaces_ext_mesh;
    if (mp->num_interfaces_ext_mesh > 0) {
        copy_todevice_int((void**)&mp->d_nibool_interfaces_ext_mesh,h_nibool_interfaces_ext_mesh,
                mp->num_interfaces_ext_mesh);
        copy_todevice_int((void**)&mp->d_ibool_interfaces_ext_mesh,h_ibool_interfaces_ext_mesh,
                (mp->num_interfaces_ext_mesh)*(mp->max_nibool_interfaces_ext_mesh));
        //int blocksize = BLOCKSIZE_TRANSFER;
        //int size_padded = ((int)ceil(((double)(mp->max_nibool_interfaces_ext_mesh))/((double)blocksize)))*blocksize;
    }
    mp->size_mpi_buffer = 0;
    mp->size_mpi_buffer_potential = 0;

    // streams
    cudaStreamCreate(&mp->compute_stream);
    // copy stream (needed to transfer mpi buffers)
    if (mp->num_interfaces_ext_mesh * mp->max_nibool_interfaces_ext_mesh > 0) {
        cudaStreamCreate(&mp->copy_stream);
    }

    // inner elements
    copy_todevice_int((void**)&mp->d_ispec_is_inner,h_ispec_is_inner,mp->NSPEC_AB);

    // absorbing boundaries
    mp->d_num_abs_boundary_faces = *h_num_abs_boundary_faces;
    if (mp->absorbing_conditions && mp->d_num_abs_boundary_faces > 0) {
        copy_todevice_int((void**)&mp->d_abs_boundary_ispec,h_abs_boundary_ispec,mp->d_num_abs_boundary_faces);
        copy_todevice_int((void**)&mp->d_abs_boundary_ijk,h_abs_boundary_ij,
                2*NGLLX*(mp->d_num_abs_boundary_faces));
        copy_todevice_realw((void**)&mp->d_abs_boundary_normal,h_abs_boundary_normal,
                NDIM*NGLLX*(mp->d_num_abs_boundary_faces));
        copy_todevice_realw((void**)&mp->d_abs_boundary_jacobian2Dw,h_abs_boundary_jacobian1Dw,
                NGLLX*(mp->d_num_abs_boundary_faces));
        copy_todevice_int((void**)&mp->d_cote_abs,h_cote_abs,(mp->d_num_abs_boundary_faces));
        copy_todevice_int((void**)&mp->d_ib_left,h_ib_left,(mp->d_num_abs_boundary_faces));
        copy_todevice_int((void**)&mp->d_ib_right,h_ib_right,(mp->d_num_abs_boundary_faces));
        copy_todevice_int((void**)&mp->d_ib_top,h_ib_top,(mp->d_num_abs_boundary_faces));
        copy_todevice_int((void**)&mp->d_ib_bottom,h_ib_bottom,(mp->d_num_abs_boundary_faces));

        mp->d_nspec_bottom = *h_nspec_bottom;
        mp->d_nspec_left = *h_nspec_left;
        mp->d_nspec_right = *h_nspec_right;
        mp->d_nspec_top = *h_nspec_top;
    }

    // sources
    mp->nsources_local = *nsources_local_f;

    if (mp->nsources_local > 0){
        copy_todevice_realw((void**)&mp->d_source_time_function,h_source_time_function,(*NSTEP)*(mp->nsources_local));
        copy_todevice_realw((void**)&mp->d_sourcearrays,h_sourcearrays,mp->nsources_local*NDIM*NGLL2);
        copy_todevice_int((void**)&mp->d_ispec_selected_source,h_ispec_selected_source,mp->nsources_local);
    }

    // receiver stations
    mp->nrec_local = *nrec_local; // number of receiver located in this partition
    // note that: size of size(ispec_selected_rec_loc) = nrec_local
    if (mp->nrec_local > 0) {
        malloc((void**)&mp->d_seismograms,2*(*NSTEP)*sizeof(realw)*(mp->nrec_local)*2);
        // pinned memory
        //mallocHost
        malloc((void**)&(mp->h_seismograms),2*(*NSTEP)*sizeof(realw)*(mp->nrec_local)*2);
        // host memory
        //mp->h_seismograms = (float*)malloc((mp->nrec_local)*2*sizeof(float));
        //if (mp->h_seismograms == NULL) exit_on_error("h_seismograms not allocated \n");

        copy_todevice_realw((void**)&mp->d_cosrot,h_cosrot,mp->nrec_local);
        copy_todevice_realw((void**)&mp->d_sinrot,h_sinrot,mp->nrec_local);

        copy_todevice_realw((void**)&mp->d_xir_store_loc,h_xir_store,(mp->nrec_local)*NGLLX);
        copy_todevice_realw((void**)&mp->d_gammar_store_loc,h_gammar_store,(mp->nrec_local)*NGLLX);

        copy_todevice_int((void**)&mp->d_ispec_selected_rec_loc,h_ispec_selected_rec_loc,mp->nrec_local);
    }
    // number of elements per domain
    mp->nspec_acoustic = *nspec_acoustic;
    mp->nspec_elastic  = *nspec_elastic;
}

// for ACOUSTIC simulations
extern "C"
void prepare_fields_acoustic_device(long* Mesh_pointer,
                                    realw* rmass_acoustic, realw* rhostore, realw* kappastore,
                                    int* num_phase_ispec_acoustic, int* phase_ispec_inner_acoustic,
                                    int* ispec_is_acoustic,
                                    int* num_free_surface_faces,
                                    int* free_surface_ispec,
                                    int* free_surface_ijk,
                                    int* ELASTIC_SIMULATION,
                                    int* num_coupling_ac_el_faces,
                                    int* coupling_ac_el_ispec,
                                    int* coupling_ac_el_ijk,
                                    realw* coupling_ac_el_normal,
                                    realw* coupling_ac_el_jacobian2Dw,
                                    int * h_ninterface_acoustic,int * h_inum_interfaces_acoustic,int* ATTENUATION_VISCOACOUSTIC,
                                    realw* h_A_newmark,realw* h_B_newmark,int* NO_BACKWARD_RECONSTRUCTION,realw* h_no_backward_acoustic_buffer) 
{
    Mesh* mp = (Mesh*)(*Mesh_pointer);

    // allocates arrays on device (GPU)
    int size = mp->NGLOB_AB;
    malloc((void**)&(mp->d_potential_acoustic),sizeof(realw)*size);
    malloc((void**)&(mp->d_potential_dot_acoustic),sizeof(realw)*size);
    malloc((void**)&(mp->d_potential_dot_dot_acoustic),sizeof(realw)*size);
    // initializes values to zero
    memset(mp->d_potential_acoustic,0,sizeof(realw)*size);
    memset(mp->d_potential_dot_acoustic,0,sizeof(realw)*size);
    memset(mp->d_potential_dot_dot_acoustic,0,sizeof(realw)*size);

    // mpi buffer
    mp->size_mpi_buffer_potential = (mp->num_interfaces_ext_mesh) * (mp->max_nibool_interfaces_ext_mesh);
    if (mp->size_mpi_buffer_potential > 0) {
        malloc((void**)&(mp->d_send_potential_dot_dot_buffer),mp->size_mpi_buffer_potential *sizeof(realw));
    }

    // mass matrix
    copy_todevice_realw((void**)&mp->d_rmass_acoustic,rmass_acoustic,mp->NGLOB_AB);

    // density
    // Assuming NGLLX==5. Padded is then 32 (5^2+3)
    int size_padded = NGLL2_PADDED * mp->NSPEC_AB;
    malloc((void**)&(mp->d_rhostore),size_padded*sizeof(realw));
    // transfer constant element data with padding
    memcpy2D(mp->d_rhostore, NGLL2_PADDED*sizeof(realw),
                rhostore, NGLL2*sizeof(realw), NGLL2*sizeof(realw),
                mp->NSPEC_AB, memcpyHostToDevice);

    // non-padded array
    copy_todevice_realw((void**)&mp->d_kappastore,kappastore,NGLL2*mp->NSPEC_AB);

    // phase elements
    mp->num_phase_ispec_acoustic = *num_phase_ispec_acoustic;
    copy_todevice_int((void**)&mp->d_phase_ispec_inner_acoustic,phase_ispec_inner_acoustic,
            2*mp->num_phase_ispec_acoustic);
    copy_todevice_int((void**)&mp->d_ispec_is_acoustic,ispec_is_acoustic,mp->NSPEC_AB);

    // allocate surface arrays
    mp->num_free_surface_faces = *num_free_surface_faces;
    if (mp->num_free_surface_faces > 0) {
        copy_todevice_int((void**)&mp->d_free_surface_ispec,free_surface_ispec,mp->num_free_surface_faces);
        copy_todevice_int((void**)&mp->d_free_surface_ijk,free_surface_ijk,2*NGLLX*mp->num_free_surface_faces);
    }

    // absorbing boundaries
    if (mp->absorbing_conditions && mp->d_num_abs_boundary_faces > 0) {
        // absorb_field array used for file i/o
        if (mp->simulation_type == 3 || ( mp->simulation_type == 1 && mp->save_forward )){
            malloc((void**)&mp->d_b_absorb_potential_left,mp->d_nspec_left*sizeof(realw)*NGLLX);
            malloc((void**)&mp->d_b_absorb_potential_right,mp->d_nspec_right*sizeof(realw)*NGLLX);
            malloc((void**)&mp->d_b_absorb_potential_top,mp->d_nspec_top*sizeof(realw)*NGLLX);
            malloc((void**)&mp->d_b_absorb_potential_bottom,mp->d_nspec_bottom*sizeof(realw)*NGLLX);

        }
    }

    // coupling with elastic parts
    if (*ELASTIC_SIMULATION && *num_coupling_ac_el_faces > 0) {
        copy_todevice_int((void**)&mp->d_coupling_ac_el_ispec,coupling_ac_el_ispec,(*num_coupling_ac_el_faces));
        copy_todevice_int((void**)&mp->d_coupling_ac_el_ijk,coupling_ac_el_ijk,2*NGLLX*(*num_coupling_ac_el_faces));
        copy_todevice_realw((void**)&mp->d_coupling_ac_el_normal,coupling_ac_el_normal,
                2*NGLLX*(*num_coupling_ac_el_faces));
        copy_todevice_realw((void**)&mp->d_coupling_ac_el_jacobian2Dw,coupling_ac_el_jacobian2Dw,
                NGLLX*(*num_coupling_ac_el_faces));
    }

    mp->ninterface_acoustic = *h_ninterface_acoustic;
    copy_todevice_int((void**)&mp->d_inum_interfaces_acoustic,h_inum_interfaces_acoustic,mp->num_interfaces_ext_mesh);

    if (*ATTENUATION_VISCOACOUSTIC) {
        copy_todevice_realw((void**)&mp->d_A_newmark_acous,h_A_newmark,NGLL2*mp->NSPEC_AB*N_SLS);
        copy_todevice_realw((void**)&mp->d_B_newmark_acous,h_B_newmark,NGLL2*mp->NSPEC_AB*N_SLS);
        malloc((void**)&mp->d_e1_acous,mp->NSPEC_AB*sizeof(realw)*NGLL2*N_SLS);
        memset(mp->d_e1_acous,0,mp->NSPEC_AB*sizeof(realw)*NGLL2*N_SLS);
        malloc((void**)&mp->d_sum_forces_old,mp->NSPEC_AB*sizeof(realw)*NGLL2);
        memset(mp->d_sum_forces_old,0,mp->NSPEC_AB*sizeof(realw)*NGLL2);
    }

    if (*NO_BACKWARD_RECONSTRUCTION){
        malloc((void**)&(mp->d_potential_acoustic_buffer),mp->NGLOB_AB*sizeof(realw));
        cudaStreamCreateWithFlags(&mp->copy_stream_no_backward,cudaStreamNonBlocking);
        cudaHostRegister(h_no_backward_acoustic_buffer,3*mp->NGLOB_AB*sizeof(realw),0);
        cudaEventCreate(&mp->transfer_is_complete1);
        cudaEventCreate(&mp->transfer_is_complete2);
    }
}

extern "C"
void prepare_fields_acoustic_adj_dev(long* Mesh_pointer,
                                     int* APPROXIMATE_HESS_KL,
                                     int* ATTENUATION_VISCOACOUSTIC,
                                     int* NO_BACKWARD_RECONSTRUCTION)
{
    Mesh* mp = (Mesh*)(*Mesh_pointer);

    // kernel simulations
    if (mp->simulation_type != 3 ) return;

    // allocates backward/reconstructed arrays on device (GPU)
    int size = mp->NGLOB_AB * sizeof(realw);
    malloc((void**)&(mp->d_b_potential_acoustic),size);
    if (! *NO_BACKWARD_RECONSTRUCTION){
        malloc((void**)&(mp->d_b_potential_dot_acoustic),size);
        malloc((void**)&(mp->d_b_potential_dot_dot_acoustic),size);
    }

    // allocates kernels
    size = NGLL2 * mp->NSPEC_AB * sizeof(realw);
    malloc((void**)&(mp->d_rho_ac_kl),size);
    malloc((void**)&(mp->d_kappa_ac_kl),size);
    memset(mp->d_rho_ac_kl,0,size);
    memset(mp->d_kappa_ac_kl,0,size);

    // preconditioner
    if (*APPROXIMATE_HESS_KL) {
        malloc((void**)&(mp->d_hess_ac_kl),size);
        memset(mp->d_hess_ac_kl,0,size);
    }

    if (*ATTENUATION_VISCOACOUSTIC && (! *NO_BACKWARD_RECONSTRUCTION) ) {
        malloc((void**)&(mp->d_b_sum_forces_old),size);
        malloc((void**)&(mp->d_b_e1_acous),size*N_SLS);
    }

    // mpi buffer
    if (mp->size_mpi_buffer_potential > 0 && (! *NO_BACKWARD_RECONSTRUCTION)) {
        malloc((void**)&(mp->d_b_send_potential_dot_dot_buffer),mp->size_mpi_buffer_potential*sizeof(realw));
    }
}

extern "C"
void prepare_fields_elastic_device(long* Mesh_pointer,
                                   realw* rmassx, realw* rmassz,
                                   realw* rho_vp, realw* rho_vs,
                                   int* num_phase_ispec_elastic,
                                   int* phase_ispec_inner_elastic,
                                   int* ispec_is_elastic,
                                   int* h_nspec_left,
                                   int* h_nspec_right,
                                   int* h_nspec_top,
                                   int* h_nspec_bottom,
                                   int* ANISOTROPY,
                                   realw *c11store,realw *c12store,realw *c13store,
                                   realw *c15store,
                                   realw *c23store,
                                   realw *c25store,realw *c33store,
                                   realw *c35store,
                                   realw *c55store,int* h_ninterface_elastic,int * h_inum_interfaces_elastic,int* ATTENUATION_VISCOELASTIC,
                                   realw* h_A_newmark_mu,realw* h_B_newmark_mu,realw* h_A_newmark_kappa,realw* h_B_newmark_kappa)
{
    Mesh* mp = (Mesh*)(*Mesh_pointer);
    int size,size_padded;

    // elastic wavefields
    size = NDIM * mp->NGLOB_AB;
    malloc((void**)&(mp->d_displ),sizeof(realw)*size);
    malloc((void**)&(mp->d_veloc),sizeof(realw)*size);
    malloc((void**)&(mp->d_accel),sizeof(realw)*size);

    // MPI buffer
    mp->size_mpi_buffer = NDIM * (mp->num_interfaces_ext_mesh) * (mp->max_nibool_interfaces_ext_mesh);
    if (mp->size_mpi_buffer > 0) {
        // Allocate pinned mpi-buffers.
        //mallocHost
        malloc((void**)&(mp->h_send_accel_buffer),sizeof(realw)*(mp->size_mpi_buffer));
        //mallocHost
        malloc((void**)&(mp->h_recv_accel_buffer),sizeof(realw)*(mp->size_mpi_buffer));
        //mp->recv_buffer = (float*)malloc((mp->size_mpi_buffer)*sizeof(float));

        // non-pinned buffer
        malloc((void**)&(mp->d_send_accel_buffer),mp->size_mpi_buffer*sizeof(realw));
        malloc((void**)&(mp->d_recv_accel_buffer),mp->size_mpi_buffer*sizeof(realw));

        // adjoint
        if (mp->simulation_type == 3) {
            malloc((void**)&(mp->d_b_send_accel_buffer),mp->size_mpi_buffer*sizeof(realw));
            malloc((void**)&(mp->d_b_recv_accel_buffer),mp->size_mpi_buffer*sizeof(realw));
        }
    }

    // mass matrix
    copy_todevice_realw((void**)&mp->d_rmassx,rmassx,mp->NGLOB_AB);
    copy_todevice_realw((void**)&mp->d_rmassz,rmassz,mp->NGLOB_AB);

    // element indices
    copy_todevice_int((void**)&mp->d_ispec_is_elastic,ispec_is_elastic,mp->NSPEC_AB);

    // phase elements
    mp->num_phase_ispec_elastic = *num_phase_ispec_elastic;

    copy_todevice_int((void**)&mp->d_phase_ispec_inner_elastic,phase_ispec_inner_elastic,2*mp->num_phase_ispec_elastic);

    // absorbing conditions
    if (mp->absorbing_conditions && mp->d_num_abs_boundary_faces > 0){
        // non-padded arrays
        // rho_vp, rho_vs non-padded; they are needed for stacey boundary condition
        copy_todevice_realw((void**)&mp->d_rho_vp,rho_vp,NGLL2*mp->NSPEC_AB);
        copy_todevice_realw((void**)&mp->d_rho_vs,rho_vs,NGLL2*mp->NSPEC_AB);

        // absorb_field array used for file i/o
        if (mp->absorbing_conditions && mp->d_num_abs_boundary_faces > 0) {
            // absorb_field array used for file i/o
            if (mp->simulation_type == 3 || ( mp->simulation_type == 1 && mp->save_forward )){
                mp->d_nspec_left = *h_nspec_left;
                malloc((void**)&mp->d_b_absorb_elastic_left,2*mp->d_nspec_left*sizeof(realw)*NGLLX);

                mp->d_nspec_right = *h_nspec_right;
                malloc((void**)&mp->d_b_absorb_elastic_right,2*mp->d_nspec_right*sizeof(realw)*NGLLX);

                mp->d_nspec_top = *h_nspec_top;
                malloc((void**)&mp->d_b_absorb_elastic_top,2*mp->d_nspec_top*sizeof(realw)*NGLLX);

                mp->d_nspec_bottom = *h_nspec_bottom;
                malloc((void**)&mp->d_b_absorb_elastic_bottom,2*mp->d_nspec_bottom*sizeof(realw)*NGLLX);
            }
        }
    }
    // anisotropy
    if (*ANISOTROPY) {
        // Assuming NGLLX==5. Padded is then 32 (5^2+3)
        size_padded = NGLL2_PADDED * (mp->NSPEC_AB);

        // allocates memory on GPU
        malloc((void**)&(mp->d_c11store),size_padded*sizeof(realw));
        malloc((void**)&(mp->d_c12store),size_padded*sizeof(realw));
        malloc((void**)&(mp->d_c13store),size_padded*sizeof(realw));
        malloc((void**)&(mp->d_c15store),size_padded*sizeof(realw));
        malloc((void**)&(mp->d_c23store),size_padded*sizeof(realw));
        malloc((void**)&(mp->d_c25store),size_padded*sizeof(realw));
        malloc((void**)&(mp->d_c33store),size_padded*sizeof(realw));
        malloc((void**)&(mp->d_c35store),size_padded*sizeof(realw));
        malloc((void**)&(mp->d_c55store),size_padded*sizeof(realw));

        memcpy2D(mp->d_c11store, NGLL2_PADDED*sizeof(realw),
                    c11store, NGLL2*sizeof(realw), NGLL2*sizeof(realw),
                    mp->NSPEC_AB, memcpyHostToDevice);
        memcpy2D(mp->d_c12store, NGLL2_PADDED*sizeof(realw),
                    c12store, NGLL2*sizeof(realw), NGLL2*sizeof(realw),
                    mp->NSPEC_AB, memcpyHostToDevice);
        memcpy2D(mp->d_c13store, NGLL2_PADDED*sizeof(realw),
                    c13store, NGLL2*sizeof(realw), NGLL2*sizeof(realw),
                    mp->NSPEC_AB, memcpyHostToDevice);
        memcpy2D(mp->d_c15store, NGLL2_PADDED*sizeof(realw),
                    c15store, NGLL2*sizeof(realw), NGLL2*sizeof(realw),
                    mp->NSPEC_AB, memcpyHostToDevice);
        memcpy2D(mp->d_c23store, NGLL2_PADDED*sizeof(realw),
                    c23store, NGLL2*sizeof(realw), NGLL2*sizeof(realw),
                    mp->NSPEC_AB, memcpyHostToDevice);
        memcpy2D(mp->d_c25store, NGLL2_PADDED*sizeof(realw),
                    c25store, NGLL2*sizeof(realw), NGLL2*sizeof(realw),
                    mp->NSPEC_AB, memcpyHostToDevice);
        memcpy2D(mp->d_c33store, NGLL2_PADDED*sizeof(realw),
                    c33store, NGLL2*sizeof(realw), NGLL2*sizeof(realw),
                    mp->NSPEC_AB, memcpyHostToDevice);
        memcpy2D(mp->d_c35store, NGLL2_PADDED*sizeof(realw),
                    c35store, NGLL2*sizeof(realw), NGLL2*sizeof(realw),
                    mp->NSPEC_AB, memcpyHostToDevice);
        memcpy2D(mp->d_c55store, NGLL2_PADDED*sizeof(realw),
                    c55store, NGLL2*sizeof(realw), NGLL2*sizeof(realw),
                    mp->NSPEC_AB, memcpyHostToDevice);
    }

    mp->ninterface_elastic = *h_ninterface_elastic;
    copy_todevice_int((void**)&mp->d_inum_interfaces_elastic,h_inum_interfaces_elastic,mp->num_interfaces_ext_mesh);


    if (*ATTENUATION_VISCOELASTIC) {
        copy_todevice_realw((void**)&mp->d_A_newmark_mu,h_A_newmark_mu,NGLL2*mp->NSPEC_AB*N_SLS);
        copy_todevice_realw((void**)&mp->d_B_newmark_mu,h_B_newmark_mu,NGLL2*mp->NSPEC_AB*N_SLS);
        copy_todevice_realw((void**)&mp->d_A_newmark_kappa,h_A_newmark_kappa,NGLL2*mp->NSPEC_AB*N_SLS);
        copy_todevice_realw((void**)&mp->d_B_newmark_kappa,h_B_newmark_kappa,NGLL2*mp->NSPEC_AB*N_SLS);
        malloc((void**)&mp->d_e1,mp->NSPEC_AB*sizeof(realw)*NGLL2*N_SLS);
        memset(mp->d_e1,0,mp->NSPEC_AB*sizeof(realw)*NGLL2*N_SLS);
        malloc((void**)&mp->d_e11,mp->NSPEC_AB*sizeof(realw)*NGLL2*N_SLS);
        memset(mp->d_e11,0,mp->NSPEC_AB*sizeof(realw)*NGLL2*N_SLS);
        malloc((void**)&mp->d_e13,mp->NSPEC_AB*sizeof(realw)*NGLL2*N_SLS);
        memset(mp->d_e13,0,mp->NSPEC_AB*sizeof(realw)*NGLL2*N_SLS);
        malloc((void**)&mp->d_dux_dxl_old,mp->NSPEC_AB*sizeof(realw)*NGLL2);
        memset(mp->d_dux_dxl_old,0,mp->NSPEC_AB*sizeof(realw)*NGLL2);
        malloc((void**)&mp->d_duz_dzl_old,mp->NSPEC_AB*sizeof(realw)*NGLL2);
        memset(mp->d_duz_dzl_old,0,mp->NSPEC_AB*sizeof(realw)*NGLL2);
        malloc((void**)&mp->d_dux_dzl_plus_duz_dxl_old,mp->NSPEC_AB*sizeof(realw)*NGLL2);
        memset(mp->d_dux_dzl_plus_duz_dxl_old,0,mp->NSPEC_AB*sizeof(realw)*NGLL2);
    }
}

extern "C"
void prepare_fields_elastic_adj_dev(long* Mesh_pointer,
                                    int* size_f,
                                    int* APPROXIMATE_HESS_KL)
{
    Mesh* mp = (Mesh*)(*Mesh_pointer);
    if (mp->simulation_type != 3 ) return;

    // kernel simulations
    // backward/reconstructed wavefields
    int size = *size_f;
    malloc((void**)&(mp->d_b_displ),sizeof(realw)*size);
    malloc((void**)&(mp->d_b_veloc),sizeof(realw)*size);
    malloc((void**)&(mp->d_b_accel),sizeof(realw)*size);

    // anisotropic/isotropic kernels

    // allocates kernels
    size = NGLL2 * mp->NSPEC_AB; // note: non-aligned; if align, check memcpy below and indexing
    // density kernel
    malloc((void**)&(mp->d_rho_kl),size*sizeof(realw));
    // initializes kernel values to zero
    memset(mp->d_rho_kl,0,size*sizeof(realw));


    // isotropic kernels
    malloc((void**)&(mp->d_mu_kl),size*sizeof(realw));
    malloc((void**)&(mp->d_kappa_kl),size*sizeof(realw));
    memset(mp->d_mu_kl,0,size*sizeof(realw));
    memset(mp->d_kappa_kl,0,size*sizeof(realw));

    malloc((void**)&(mp->d_dsxx),size*sizeof(realw));
    malloc((void**)&(mp->d_dsxz),size*sizeof(realw));
    malloc((void**)&(mp->d_dszz),size*sizeof(realw));
    malloc((void**)&(mp->d_b_dsxx),size*sizeof(realw));
    malloc((void**)&(mp->d_b_dsxz),size*sizeof(realw));
    malloc((void**)&(mp->d_b_dszz),size*sizeof(realw));

    // approximate hessian kernel
    if (*APPROXIMATE_HESS_KL) {
        size = NGLL2 * mp->NSPEC_AB; // note: non-aligned; if align, check memcpy below and indexing
        malloc((void**)&(mp->d_hess_el_kl),size*sizeof(realw));
        memset(mp->d_hess_el_kl,0,size*sizeof(realw));
    }
}

// purely adjoint & kernel simulations
extern "C"
void prepare_sim2_or_3_const_device(long* Mesh_pointer,
                                    int* nadj_rec_local,
                                    realw* h_source_adjointe,
                                    int* NSTEP) 
{
    Mesh* mp = (Mesh*)(*Mesh_pointer);

    // adjoint source arrays
    mp->nadj_rec_local = *nadj_rec_local;
    if (mp->nadj_rec_local > 0) {
        malloc((void**)&mp->d_adj_sourcearrays,(mp->nadj_rec_local)*2*NGLL2*sizeof(realw));

        copy_todevice_realw((void**)&mp->d_source_adjointe,h_source_adjointe,(*NSTEP)*(*nadj_rec_local)*NDIM);
    }
}

extern "C"
void prepare_cleanup_device(long* Mesh_pointer,
                            int* ACOUSTIC_SIMULATION,
                            int* ELASTIC_SIMULATION,
                            int* ABSORBING_CONDITIONS,
                            int* ANISOTROPY,
                            int* APPROXIMATE_HESS_KL,
                            int* ATTENUATION_VISCOACOUSTIC,
                            int* ATTENUATION_VISCOELASTIC,
                            int* NO_BACKWARD_RECONSTRUCTION,
                            realw * h_no_backward_acoustic_buffer)
{
    Mesh* mp = (Mesh*)(*Mesh_pointer);

    // mesh
    free(mp->d_xix);
    free(mp->d_xiz);
    free(mp->d_gammax);
    free(mp->d_gammaz);
    free(mp->d_muv);
    free(mp->d_kappav);

    // absorbing boundaries
    if (*ABSORBING_CONDITIONS && mp->d_num_abs_boundary_faces > 0) {
        free(mp->d_abs_boundary_ispec);
        free(mp->d_abs_boundary_ijk);
        free(mp->d_abs_boundary_normal);
        free(mp->d_abs_boundary_jacobian2Dw);
        free(mp->d_cote_abs);
        free(mp->d_ib_left);
        free(mp->d_ib_right);
        free(mp->d_ib_top);
        free(mp->d_ib_bottom);
    }
    // interfaces
    if (mp->num_interfaces_ext_mesh > 0) {
        free(mp->d_nibool_interfaces_ext_mesh);
        free(mp->d_ibool_interfaces_ext_mesh);
    }
    // global indexing
    free(mp->d_ispec_is_inner);
    free(mp->d_ibool);
    // sources
    if (mp->nsources_local > 0){
        free(mp->d_sourcearrays);
        free(mp->d_source_time_function);
        free(mp->d_ispec_selected_source);
    }
    // receivers
    if (mp->nrec_local > 0) {
        free(mp->d_seismograms);
        free(mp->d_cosrot),free(mp->d_sinrot);
        free(mp->d_gammar_store_loc);
        free(mp->d_xir_store_loc);
        free(mp->d_ispec_selected_rec_loc);
        freeHost(mp->h_seismograms);
    }
    // ACOUSTIC arrays
    if (*ACOUSTIC_SIMULATION) {
        free(mp->d_potential_acoustic);
        free(mp->d_potential_dot_acoustic);
        free(mp->d_potential_dot_dot_acoustic);
        if (mp->size_mpi_buffer_potential > 0 ) free(mp->d_send_potential_dot_dot_buffer);
        free(mp->d_rmass_acoustic);
        free(mp->d_rhostore);
        free(mp->d_kappastore);
        free(mp->d_phase_ispec_inner_acoustic);
        free(mp->d_ispec_is_acoustic);
        free(mp->d_inum_interfaces_acoustic);

        if (*NO_BACKWARD_RECONSTRUCTION){
            free(mp->d_potential_acoustic_buffer);
            cudaHostUnregister(h_no_backward_acoustic_buffer);
            cudaEventDestroy(mp->transfer_is_complete1);
            cudaEventDestroy(mp->transfer_is_complete2);

        }
        if (mp->simulation_type == 3) {
            free(mp->d_b_potential_acoustic);
            if (! *NO_BACKWARD_RECONSTRUCTION){
                free(mp->d_b_potential_dot_acoustic);
                free(mp->d_b_potential_dot_dot_acoustic);
            }
            free(mp->d_rho_ac_kl);
            free(mp->d_kappa_ac_kl);
            if (*APPROXIMATE_HESS_KL) free(mp->d_hess_ac_kl);
            if (mp->size_mpi_buffer_potential > 0 && ! *NO_BACKWARD_RECONSTRUCTION) free(mp->d_b_send_potential_dot_dot_buffer);
            if (*ATTENUATION_VISCOACOUSTIC && ! *NO_BACKWARD_RECONSTRUCTION) {
                free(mp->d_b_sum_forces_old);
                free(mp->d_b_e1_acous);
            }
        }
        if (*ABSORBING_CONDITIONS && mp->d_num_abs_boundary_faces > 0){
            if (mp->simulation_type == 3 || ( mp->simulation_type == 1 && mp->save_forward )){
                free(mp->d_b_absorb_potential_bottom);
                free(mp->d_b_absorb_potential_left);
                free(mp->d_b_absorb_potential_right);
                free(mp->d_b_absorb_potential_top);
            }
        }
        if (*ATTENUATION_VISCOACOUSTIC){
            free(mp->d_e1_acous);
            free(mp->d_A_newmark_acous);
            free(mp->d_B_newmark_acous);
            free(mp->d_sum_forces_old);
        }

    } 
    // ELASTIC arrays
    if (*ELASTIC_SIMULATION) {
        free(mp->d_displ);
        free(mp->d_veloc);
        free(mp->d_accel);

        if (mp->size_mpi_buffer > 0){
            free(mp->d_send_accel_buffer);
            free(mp->d_recv_accel_buffer);
            freeHost(mp->h_send_accel_buffer);
            freeHost(mp->h_recv_accel_buffer);
            if (mp->simulation_type == 3){
                free(mp->d_b_send_accel_buffer);
                free(mp->d_b_recv_accel_buffer);
            }
        }

        free(mp->d_rmassx);
        free(mp->d_rmassz);

        free(mp->d_phase_ispec_inner_elastic);
        free(mp->d_ispec_is_elastic);
        free(mp->d_inum_interfaces_elastic);

        if (*ABSORBING_CONDITIONS && mp->d_num_abs_boundary_faces > 0){
            free(mp->d_rho_vp);
            free(mp->d_rho_vs);
            free(mp->d_b_absorb_elastic_bottom);
            free(mp->d_b_absorb_elastic_left);
            free(mp->d_b_absorb_elastic_right);
            free(mp->d_b_absorb_elastic_top);
        }
        if (mp->simulation_type == 3) {
            free(mp->d_b_displ);
            free(mp->d_b_veloc);
            free(mp->d_b_accel);
            free(mp->d_rho_kl);
            free(mp->d_mu_kl);
            free(mp->d_kappa_kl);
            free(mp->d_b_dsxx);
            free(mp->d_b_dsxz);
            free(mp->d_b_dszz);
            free(mp->d_dsxx);
            free(mp->d_dsxz);
            free(mp->d_dszz);
            if (*APPROXIMATE_HESS_KL ) free(mp->d_hess_el_kl);
        }
        if (*ANISOTROPY) {
            free(mp->d_c11store);
            free(mp->d_c12store);
            free(mp->d_c13store);
            free(mp->d_c15store);
            free(mp->d_c23store);
            free(mp->d_c25store);
            free(mp->d_c33store);
            free(mp->d_c35store);
            free(mp->d_c55store);
        }
        if (*ATTENUATION_VISCOELASTIC) {
            free(mp->d_A_newmark_mu);
            free(mp->d_B_newmark_mu);
            free(mp->d_A_newmark_kappa);
            free(mp->d_B_newmark_kappa);
            free(mp->d_e1);
            free(mp->d_e11);
            free(mp->d_e13);
            free(mp->d_dux_dxl_old);
            free(mp->d_duz_dzl_old);
            free(mp->d_dux_dzl_plus_duz_dxl_old);
        }
    }
    // purely adjoint & kernel array
    if (mp->simulation_type == 3) {
        if (mp->nadj_rec_local > 0) {
            free(mp->d_adj_sourcearrays);
        }
    }

    // mesh pointer - not needed anymore
    free(mp);
}
