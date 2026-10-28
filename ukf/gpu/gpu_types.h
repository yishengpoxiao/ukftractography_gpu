/**
 * \file gpu_types.h
 * \brief POD types shared between the host-side GPU orchestration code and the
 *        CUDA kernels. No Eigen/C++ objects here so that nvcc never sees Eigen.
 */
#ifndef UKF_GPU_TYPES_H_
#define UKF_GPU_TYPES_H_

#include <cstdint>

#define UKF_GPU_MAX_STATE 18
#define UKF_GPU_MAX_SIGMA (2 * UKF_GPU_MAX_STATE + 1)
#define UKF_GPU_MAX_STOP_LABELS 64
#define UKF_GPU_NUM_THREADS 128

/* Specialized Simple1T/2T paths and the general tensor/FreeWater/NODDI path. */
#define UKF_GPU_MODEL_SIMPLE1T 0
#define UKF_GPU_MODEL_SIMPLE2T 1
#define UKF_GPU_MODEL_GENERAL 2

struct GPUBranchSeed
{
  int fiber, step, tensor;
  double pos[3], direction[3], fa;
  double state[UKF_GPU_MAX_STATE], cov[UKF_GPU_MAX_STATE*UKF_GPU_MAX_STATE];
};

struct GPUDarParams
{
  int model;          /* UKF_GPU_MODEL_* */
  int n;              /* state dimension */
  int num_tensors, full, free_water, noddi;
  int eigen_exp_packet, eigen_fma;
  int libm_exp, libm_rotation;
  int eigen_packet, eigen_gemm_mr, eigen_solve_panel;
  int has_process_noise, num_constraints;
  double process_noise[UKF_GPU_MAX_STATE];
  int constraint_index[8];
  double constraint_sign[8], constraint_offset[8];
  double diff_fw;
  int branching;
  double cos_branch_min, cos_branch_max;
  int G;              /* number of gradients; signal_dim = 2G */
  int nx, ny, nz;
  double vx, vy, vz;
  double sigma_signal;/* interpolation bandwidth; 0 -> min voxel size */
  double Qm, Ql, Rs;
  double w1, w2;      /* weights_on_tensors_ */
  double lambda_min;  /* 100.0 */
  double stepLength;
  double mean_signal_min, fa_min, min_radius;
  int max_length;     /* max number of steps per half fiber */
  int steps_per_record;
  int record_nmse, record_trace, record_fa, record_cov;
  double gm_thr, csf_thr;
  int use_gmcsf;      /* isGMCSFProvided */
  int mixed_requested; /* Guarded FP32 signal-exp range selection; sensitive math is FP64. */
  int stop_labels[UKF_GPU_MAX_STOP_LABELS];
  int num_stop_labels;
  int max_rec;        /* record capacity per fiber */
  int mask_type;      /* nrrd type of the mask volume: 1,2,3,4 */
  int stop_type;      /* nrrd type of the stop volume: 2,3,5 */
  int gm_type;        /* 9 or 10 */
  int csf_type;       /* 9 or 10 */
  std::int64_t step1, step2;  /* dwi strides: nz*ny*G, nz*G */
};

/* Device pointers, one allocation set per seed chunk. */
struct GPUDevPointers
{
  /* fiber state */
  void*  pos;  /* FP64 */        /* 3*F */
  void*  state;  /* FP64 */      /* n*F */
  void*  cov;  /* FP64 */        /* n*n*F (row-major within a fiber) */
  void*  fa;  /* FP64 */         /* F */
  void*  fa2;  /* FP64 */        /* F */
  void*  trace;  /* FP64 */      /* F */
  void*  trace2;  /* FP64 */     /* F */
  void*  nmse;  /* FP64 */       /* F (current step's dNormMSE) */
  void*  m1;  /* FP64 */         /* 3*F previous main direction (2T only) */
  int*    active;     /* F */
  int*    reccount;   /* F */
  GPUBranchSeed* branches;
  int* branch_count;
  int*    invalid;    /* scalar counter of numerical failures */

  /* active-list compaction */
  int*    act_idx;      /* current compact list of active fiber ids */
  int*    act_idx_tmp;  /* scratch for the next wave */
  int*    n_active_tmp; /* scalar counter used by k_compact */

  /* per-step scratch */
  void*  z;  /* FP64 */          /* F * 2G     interpolated measured signal */
  void*  X;  /* FP64 */          /* F * (2n+1) * n  sigma points, col i at X[(f*(2n+1)+i)*n] */
  void*  Z;  /* FP64 */          /* F * (2n+1) * 2G predicted signals */
  void*  scratch;  /* FP64 */    /* F * max(2*n*2G,n*n): Pxz/Ht or posterior information */

  /* record buffers, fixed stride max_rec per fiber */
  void*  rec_pos;  /* FP64 */    /* F*max_rec*3 */
  void*  rec_norm;  /* FP64 */   /* F*max_rec */
  void*  rec_state;  /* FP64 */  /* F*max_rec*n */
  void*  rec_cov;  /* FP64 */    /* F*max_rec*n*n (only if record_cov) */
  void*  rec_nmse;  /* FP64 */   /* F*max_rec */
  void*  rec_trace;  /* FP64 */  /* F*max_rec */
  void*  rec_trace2;  /* FP64 */ /* F*max_rec */
  void*  rec_fa;  /* FP64 */     /* F*max_rec */
  void*  rec_fa2;  /* FP64 */    /* F*max_rec */

  /* constant input tables (device copies) */
  void*  noddi_table; /* G * 10: protocol b, parallel component, Eiso, seven integrals */
  void*  free_water_signal; /* FP64, 2G: state-independent isotropic observation */
  void*  gradients;  /* FP64 */  /* 2G*3 */
  void*  bvalues;  /* FP64 */    /* 2G */
  float*  dwi;        /* nx*ny*nz*G (always float) */
  void*   mask;       /* raw mask volume */
  int     mask_type;  /* nrrd type: 1,2,3,4 */
  void*   stop;       /* raw stop volume (may be null) */
  int     stop_type;  /* nrrd type: 2,3,5 */
  float*  gm;         /* may be null */
  int     gm_type;    /* 9 or 10 */
  float*  csf;        /* may be null */
  int     csf_type;   /* 9 or 10 */
};

#endif  /* UKF_GPU_TYPES_H_ */
