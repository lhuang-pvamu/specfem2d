
#define NGLLX 5
#define NGLL2 25

#define NGLL2_PADDED 32

#define NDIM 2

#define INDEX2(xsize,x,y) x + (y)*xsize
#define INDEX3(xsize,ysize,x,y,z) x + xsize*(y + ysize*z)
#define INDEX4(xsize,ysize,zsize,x,y,z,i) x + xsize*(y + ysize*(z + zsize*i))

#define INDEX3_PADDED(xsize,ysize,x,y,i) x + (y)*xsize + (i)*NGLL2_PADDED
#define INDEX4_PADDED(xsize,ysize,zsize,x,y,z,i) x + xsize*(y + ysize*z) + (i)*NGLL3_PADDED


typedef float realw;
typedef realw* __restrict__ realw_p;
typedef const realw* __restrict__ realw_const_p;

typedef struct mesh_ {
  // mesh resolution
  int NSPEC_AB;
  int NGLOB_AB;
  // mpi process
  int myrank;
  // constants
  int simulation_type;
  int save_forward;
  int absorbing_conditions;
  // ------------------------------------------------------------------ //
  // GLL points & weights
  // ------------------------------------------------------------------ //
  // interpolators
  realw* d_xix; realw* d_xiz;
  realw* d_gammax; realw* d_gammaz;
  // model parameters
  realw* d_kappav; realw* d_muv;
  // global indexing
  int* d_ibool;
  // inner / outer elements
  int* d_ispec_is_inner;
  // pointers to constant memory arrays
  realw* d_hprime_xx;
  realw* d_hprimewgll_xx;
  realw* d_wxgll;

  // A buffer for mpi-send/recv, which is duplicated in fortran but is
  // allocated with pinned memory to facilitate asynchronus device <->
  // host memory transfers
  float* h_send_accel_buffer;
  float* h_send_b_accel_buffer;

  float* send_buffer;
  float* h_recv_accel_buffer;
  float* h_recv_b_accel_buffer;
  //float* recv_buffer;

  int size_mpi_buffer;
  int size_mpi_buffer_potential;

  // mpi interfaces
  int num_interfaces_ext_mesh;
  int max_nibool_interfaces_ext_mesh;
  int* d_nibool_interfaces_ext_mesh;
  int* d_ibool_interfaces_ext_mesh;

  // sources
  int nsources_local;
  realw* d_sourcearrays;
  int* d_ispec_selected_source;
  realw* d_source_time_function;

  // receivers
  int nrec_local;
  int* d_ispec_selected_rec_loc;
  realw* h_seismograms;
  realw* d_seismograms;
  realw* d_cosrot;
  realw* d_sinrot;

  // adjoint receivers/sources
  int nadj_rec_local;
  realw* d_adj_sourcearrays;
  realw* h_adj_sourcearrays_slice;
  realw* d_source_adjointe;
  realw* d_xir_store_loc;
  realw* d_gammar_store_loc;

  // ------------------------------------------------------------------ //
  // elastic wavefield parameters
  // ------------------------------------------------------------------ //

  // displacement, velocity, acceleration
  realw* d_displ; realw* d_veloc; realw* d_accel;
  // backward/reconstructed elastic wavefield
  realw* d_b_displ; realw* d_b_veloc; realw* d_b_accel;

  // elastic elements
  int nspec_elastic;
  int* d_ispec_is_elastic;

  // elastic domain parameters
  int* d_phase_ispec_inner_elastic;
  int num_phase_ispec_elastic;
  int ninterface_elastic;
  int * d_inum_interfaces_elastic;

  realw* d_rmassx;
  realw* d_rmassz;

  //attenuation
  realw* d_e1;
  realw* d_e11;
  realw* d_e13;
  realw* d_A_newmark_mu;
  realw* d_B_newmark_mu;
  realw* d_A_newmark_kappa;
  realw* d_B_newmark_kappa;
  realw* d_dux_dxl_old;
  realw* d_duz_dzl_old;
  realw* d_dux_dzl_plus_duz_dxl_old;

  // mpi buffer
  realw* d_send_accel_buffer;
  realw* d_b_send_accel_buffer;
  realw* d_recv_accel_buffer;
  realw* d_b_recv_accel_buffer;

  //used for absorbing stacey boundaries
  int d_num_abs_boundary_faces;
  int* d_abs_boundary_ispec;
  int* d_abs_boundary_ijk;
  realw* d_abs_boundary_normal;
  realw* d_abs_boundary_jacobian2Dw;
  int* d_cote_abs;
  int* d_ib_left;
  int* d_ib_right;
  int* d_ib_top;
  int* d_ib_bottom;
  realw* d_b_absorb_potential_bottom;
  realw* d_b_absorb_potential_left;
  realw* d_b_absorb_potential_right;
  realw* d_b_absorb_potential_top;
  int d_nspec_bottom;
  int d_nspec_left;
  int d_nspec_right;
  int d_nspec_top;
  realw* d_b_absorb_elastic_bottom;
  realw* d_b_absorb_elastic_left;
  realw* d_b_absorb_elastic_right;
  realw* d_b_absorb_elastic_top;

  realw* d_rho_vp;
  realw* d_rho_vs;

  // surface elements (to save for noise tomography and acoustic simulations)
  int* d_free_surface_ispec;
  int* d_free_surface_ijk;
  int num_free_surface_faces;

  // anisotropy
  realw* d_c11store;
  realw* d_c12store;
  realw* d_c13store;
  realw* d_c15store;
  realw* d_c23store;
  realw* d_c25store;
  realw* d_c33store;
  realw* d_c35store;
  realw* d_c55store;

  // sensitivity kernels
  realw* d_rho_kl;
  realw* d_mu_kl;
  realw* d_kappa_kl;
  realw* d_hess_el_kl;
  realw* d_dsxx;
  realw* d_dsxz;
  realw* d_dszz;
  realw* d_b_dsxx;
  realw* d_b_dsxz;
  realw* d_b_dszz;

  // JC JC here we will need to add GPU support for the new C-PML routines

  // ------------------------------------------------------------------ //
  // acoustic wavefield
  // ------------------------------------------------------------------ //
  // potential and first and second time derivative
  realw* d_potential_acoustic; realw* d_potential_dot_acoustic; realw* d_potential_dot_dot_acoustic;
  // backward/reconstructed wavefield
  realw* d_b_potential_acoustic; realw* d_b_potential_dot_acoustic; realw* d_b_potential_dot_dot_acoustic;
  // buffer for NO_BACKWARD_RECONSTRUCTION
  realw* d_potential_acoustic_buffer;

  // acoustic domain parameters
  int nspec_acoustic;
  int* d_ispec_is_acoustic;

  int* d_phase_ispec_inner_acoustic;
  int num_phase_ispec_acoustic;
  int ninterface_acoustic;
  int * d_inum_interfaces_acoustic;

  realw* d_rhostore;
  realw* d_kappastore;
  realw* d_rmass_acoustic;

  // attenuation
  realw* d_A_newmark_acous;
  realw* d_B_newmark_acous;
  realw* d_e1_acous;
  realw* d_sum_forces_old;
  realw* d_b_e1_acous;
  realw* d_b_sum_forces_old;

  // mpi buffer
  realw* d_send_potential_dot_dot_buffer;
  realw* d_b_send_potential_dot_dot_buffer;

  // sensitivity kernels
  realw* d_rho_ac_kl;
  realw* d_kappa_ac_kl;

  // approximative hessian for preconditioning kernels
  realw* d_hess_ac_kl;

  // coupling acoustic-elastic
  int* d_coupling_ac_el_ispec;
  int* d_coupling_ac_el_ijk;
  realw* d_coupling_ac_el_normal;
  realw* d_coupling_ac_el_jacobian2Dw;

} Mesh;
