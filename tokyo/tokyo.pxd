cimport numpy as np

#
# External imports from Basic Linear Algebra Subroutines (BLAS)
#

cdef extern from "Python.h":

    cdef void Py_INCREF(object)


cdef extern from "numpy/arrayobject.h":

    cdef void import_array()

    cdef object PyArray_ZEROS(int nd, np.npy_intp *dims, int typenum,
                              int fortran)
    cdef object PyArray_SimpleNew(int nd, np.npy_intp *dims, int typenum)
    cdef object PyArray_EMPTY(int nd, np.npy_intp *dims, int typenum,
                              int fortran)

    int PyArray_ISCARRAY(np.ndarray instance) # I can't get this one to work?!?

    int NPY_FLOAT   # PyArray_FLOAT  deprecated.
    int NPY_DOUBLE  # PyArray_DOUBLE deprecated.


cdef extern from "cblas.h":

    enum CBLAS_ORDER:     CblasRowMajor, CblasColMajor
    enum CBLAS_TRANSPOSE: CblasNoTrans, CblasTrans, CblasConjTrans
    enum CBLAS_UPLO:      CblasUpper, CblasLower
    enum CBLAS_DIAG:      CblasNonUnit, CblasUnit
    enum CBLAS_SIDE:      CblasLeft, CblasRight

    ###########################################################################
    # BLAS level 1 routines
    ###########################################################################

    # Swap vectors: x <-> y
    void   lib_sswap  "cblas_sswap"(int M, float  *x, int dx, float  *y, int dy)
    void   lib_dswap  "cblas_dswap"(int M, double *x, int dx, double *y, int dy)

    # Scale a vector: x <- alpha*x
    void   lib_sscal  "cblas_sscal"(int N, float  alpha, float  *x, int dx)
    void   lib_dscal  "cblas_dscal"(int N, double alpha, double *x, int dx)

    # Copy a vector: y <- x
    void   lib_scopy  "cblas_scopy"(int N, float  *x, int dx, float  *y, int dy)
    void   lib_dcopy  "cblas_dcopy"(int N, double *x, int dx, double *y, int dy)

    # Combined multiply/add: y <- alpha*x + y
    void   lib_saxpy  "cblas_saxpy"(int N, float  alpha, float  *x, int dx,
                                                         float  *y, int dy)
    void   lib_daxpy  "cblas_daxpy"(int N, double alpha, double *x, int dx,
                                                         double *y, int dy)

    # Dot product: x'y
    float  lib_sdot   "cblas_sdot"(int N, float  *x, int dx, float  *y, int dy)
    double lib_ddot   "cblas_ddot"(int N, double *x, int dx, double *y, int dy)
    double lib_dsdot  "cblas_dsdot"(int N, float *x, int dx, float  *y, int dy)
    float  lib_sdsdot "cblas_sdsdot"(int N, float alpha, float *x, int dx,
                                                         float *y, int dy)

    # Euclidian (2-)norm: ||x||_2
    float  lib_snrm2  "cblas_snrm2"(int N, float  *x, int dx)
    double lib_dnrm2  "cblas_dnrm2"(int N, double *x, int dx)

    # One norm: ||x||_1 = sum |xi|
    float  lib_sasum  "cblas_sasum"(int N, float  *x, int dx)
    double lib_dasum  "cblas_dasum"(int N, double *x, int dx)

    # Argmax: i = arg max(|xj|)
    int    lib_isamax "cblas_isamax"(int N, float  *x, int dx)
    int    lib_idamax "cblas_idamax"(int N, double *x, int dx)

    # Generate a plane rotation.
    void   lib_srotg  "cblas_srotg"(float  *a, float  *b, float  *c, float  *s)
    void   lib_drotg  "cblas_drotg"(double *a, double *b, double *c, double *s)

    # Generate a modified plane rotation.
    void   lib_srotmg "cblas_srotmg"(float  *d1, float  *d2, float  *b1,
                                     float  b2, float  *P)
    void   lib_drotmg "cblas_drotmg"(double *d1, double *d2, double *b1,
                                     double b2, double *P)

    # Apply a plane rotation.
    void   lib_srot   "cblas_srot"(int N, float  *x, int dx,
                                          float  *y, int dy,
                                          float c, float s)
    void   lib_drot   "cblas_drot"(int N, double *x, int dx,
                                          double *y, int dy,
                                          double c, double s)

    # Apply a modified plane rotation.
    void   lib_srotm  "cblas_srotm"(int N, float *x, int dx,
                                           float *y, int dy,
                                           float *P)
    void   lib_drotm  "cblas_drotm"(int N, double *x, int dx,
                                           double *y, int dy,
                                           double *P)

    ###########################################################################
    # BLAS level 2 routines
    ###########################################################################

    # Combined multiply/add: y <- alpha*Ax + beta*y or y <- alpha*A'x + beta*y
    void lib_sgemv "cblas_sgemv"(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA,
                                 int M, int N, float  alpha, float  *A, int lda,
                                               float  *x, int dx,
                                 float  beta,  float  *y, int dy)

    void lib_dgemv "cblas_dgemv"(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA,
                                 int M, int N, double alpha, double *A, int lda,
                                               double *x, int dx,
                                 double beta,  double *y, int dy)

    void lib_ssymv "cblas_ssymv"(CBLAS_ORDER Order, CBLAS_UPLO Uplo, int N,
                                 float alpha, float *A, int lda,
                                 float *x, int dx, float beta,
                                 float *y, int dy)

    void lib_strmv "cblas_strmv"(CBLAS_ORDER Order, CBLAS_UPLO Uplo,
                                 CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                                 int N, float *A, int lda, float *x, int dx)

    void lib_strsv "cblas_strsv"(CBLAS_ORDER Order, CBLAS_UPLO Uplo,
                                 CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                                 int N, float *A, int lda, float *x, int dx)

    void lib_dtrmv "cblas_dtrmv"(CBLAS_ORDER Order, CBLAS_UPLO Uplo,
                                 CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                                 int N, double *A, int lda, double *x, int dx)

    void lib_dsymv "cblas_dsymv"(CBLAS_ORDER Order, CBLAS_UPLO Uplo, int N,
                                 double alpha, double *A, int lda,
                                 double *x, int dx, double beta,
                                 double *y, int dy)

    void lib_dtrsv "cblas_dtrsv"(CBLAS_ORDER Order, CBLAS_UPLO Uplo,
                                 CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
                                 int N, double *A, int lda, double *x, int dx)

    # Rank-1 update: A <- alpha * x*y' + A
    void lib_sger  "cblas_sger"(CBLAS_ORDER Order, int M, int N, float  alpha,
                                float  *x, int dx, float  *y, int dy,
                                float  *A, int lda)

    void lib_ssyr  "cblas_ssyr"(CBLAS_ORDER Order, CBLAS_UPLO Uplo, int N,
                                float alpha, float *x, int dx, float *A, int lda)

    void lib_ssyr2 "cblas_ssyr2"(CBLAS_ORDER order, CBLAS_UPLO Uplo, int N,
                                 float alpha, float *x, int dx, float *y,
                                 int dy, float *A, int lda)

    void lib_dger  "cblas_dger"(CBLAS_ORDER Order, int M, int N, double alpha,
                                double *x, int dx, double *y, int dy,
                                double *A, int lda)

    void lib_dsyr  "cblas_dsyr"(CBLAS_ORDER Order, CBLAS_UPLO Uplo, int N,
                                double alpha, double *x, int dx,
                                double *A, int lda)

    void lib_dsyr2 "cblas_dsyr2"(CBLAS_ORDER order, CBLAS_UPLO Uplo, int N,
                                 double alpha, double *x, int dx, double *y,
                                 int dy, double *A, int lda)

    ###########################################################################
    # BLAS level 3 routines
    ###########################################################################

    void lib_sgemm "cblas_sgemm"(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA,
                                 CBLAS_TRANSPOSE TransB, int M, int N, int K,
                                 float  alpha, float  *A, int lda,
                                               float  *B, int ldb,
                                 float  beta,  float  *C, int ldc)

    void lib_ssymm "cblas_ssymm"(CBLAS_ORDER Order, CBLAS_SIDE Side,
                                 CBLAS_UPLO Uplo, int M, int N,
                                 float alpha, float *A, int lda,
                                              float *B, int ldb,
                                 float beta,  float *C, int ldc)

    void lib_ssyrk "cblas_ssyrk"(CBLAS_ORDER Order, CBLAS_UPLO Uplo,
                                 CBLAS_TRANSPOSE Trans, int N, int K,
                                 float alpha, float *A, int lda, float beta,
                                 float *C, int ldc)

    void lib_dgemm "cblas_dgemm"(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA,
                                 CBLAS_TRANSPOSE TransB, int M, int N, int K,
                                 double alpha, double *A, int lda,
                                               double *B, int ldb,
                                 double beta,  double *C, int ldc)

    void lib_dsymm "cblas_dsymm"(CBLAS_ORDER Order, CBLAS_SIDE Side,
                                 CBLAS_UPLO Uplo, int M, int N,
                                 double alpha, double *A, int lda,
                                               double *B, int ldb,
                                 double beta,  double *C, int ldc)

    void lib_dsyrk "cblas_dsyrk"(CBLAS_ORDER Order, CBLAS_UPLO Uplo,
                                 CBLAS_TRANSPOSE Trans, int N, int K,
                                 double alpha, double *A, int lda, double beta,
                                 double *C, int ldc)


#####################################
#
# BLAS LEVEL 1 (vector operations)
#
#####################################

# vector swap: x <-> y
cdef void sswap_(int M, float *x, int dx, float *y, int dy)
cdef void sswap(np.ndarray x, np.ndarray y)

cdef void dswap_(int M, double *x, int dx, double *y, int dy)
cdef void dswap(np.ndarray x, np.ndarray y)

# scalar vector multiply: x *= alpha
cdef void sscal_(int N, float alpha, float *x, int dx)
cdef void sscal(float alpha, np.ndarray x)

cdef void dscal_(int N, double alpha, double *x, int dx)
cdef void dscal(double alpha, np.ndarray x)

# vector copy: y <- x
cdef void scopy_(int N, float *x, int dx, float *y, int dy)
cdef void scopy(np.ndarray x, np.ndarray y)

cdef void dcopy_(int N, double *x, int dx, double *y, int dy)
cdef void dcopy(np.ndarray x, np.ndarray y)

# vector addition: y += alpha*x
cdef void saxpy_(int N, float alpha, float *x, int dx, float *y, int dy)
cdef void saxpy(float alpha, np.ndarray x, np.ndarray y)

cdef void daxpy_(int N, double alpha, double *x, int dx, double *y, int dy)
cdef void daxpy(double alpha, np.ndarray x, np.ndarray y)

# vector dot product: x.T y
cdef float sdot_(int N, float *x, int dx, float *y, int dy)
cdef float sdot(np.ndarray x, np.ndarray y)

cdef double ddot_(int N, double *x, int dx, double *y, int dy)
cdef double ddot(np.ndarray x, np.ndarray y)

cdef double dsdot_(int N, float *x, int dx, float *y, int dy)
cdef double dsdot(np.ndarray x, np.ndarray y)
cdef float sdsdot_(int N, float alpha, float *x, int dx, float *y, int dy)
cdef float sdsdot(float alpha, np.ndarray x, np.ndarray y)

# Euclidean norm:  ||x||_2
cdef float snrm2_(int N, float *x, int dx)
cdef float snrm2(np.ndarray)

cdef double dnrm2_(int N, double *x, int dx)
cdef double dnrm2(np.ndarray)

# sum of absolute values: ||x||_1
cdef float sasum_(int N, float *x, int dx)
cdef float sasum(np.ndarray x)

cdef double dasum_(int N, double *x, int dx)
cdef double dasum(np.ndarray x)

# index of maximum absolute value element
cdef int isamax_(int N, float *x, int dx)
cdef int isamax(np.ndarray x)

cdef int idamax_(int N, double *x, int dx)
cdef int idamax(np.ndarray x)

# Generate a Givens plane rotation: [a,b,c,s] <- rotg(a,b).
cdef np.ndarray srotg_(float a, float b)
cdef np.ndarray srotg(float a, float b)

cdef np.ndarray drotg_(double a, double b)
cdef np.ndarray drotg(double a, double b)

###########################################
#
# BLAS LEVEL 2 (matrix-vector operations)
#
###########################################


# single precision general matrix-vector multiply
cdef void sgemv_(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA, int M, int N,
                    float  alpha, float  *A, int lda, float  *x, int dx,
                    float  beta, float  *y, int dy)

# y = alpha * A   x + beta * y
# y = alpha * A.T x + beta * y
cdef void sgemv6(CBLAS_TRANSPOSE TransA, float  alpha, np.ndarray A,
                    np.ndarray x, float  beta, np.ndarray y)

# y = alpha * A x + beta * y
cdef void sgemv5(float  alpha, np.ndarray A,
                    np.ndarray x, float  beta, np.ndarray y)

# y = alpha * A x
cdef void sgemv3(np.ndarray A, np.ndarray x, np.ndarray y)

# return = alpha * A x
cdef np.ndarray sgemv(np.ndarray A, np.ndarray x)


# double precision general matrix-vector multiply
cdef void dgemv_(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA, int M, int N,
                    double alpha, double *A, int lda, double *x, int dx,
                    double beta, double *y, int dy)

# y = alpha * A   x + beta * y
# y = alpha * A.T x + beta * y
cdef void dgemv6(CBLAS_TRANSPOSE TransA, double alpha, np.ndarray A,
                    np.ndarray x, double beta, np.ndarray y)

# y = alpha * A x + beta * y
cdef void dgemv5(double alpha, np.ndarray A,
                    np.ndarray x, double beta, np.ndarray y)

# y = alpha * A x
cdef void dgemv3(np.ndarray A, np.ndarray x, np.ndarray y)

# return = alpha * A x
cdef np.ndarray dgemv(np.ndarray A, np.ndarray x)

####

# single precision symmetric matrix-vector multiply
cdef void ssymv_(CBLAS_ORDER Order, CBLAS_UPLO Uplo, int N, float alpha,
                 float *A, int lda, float *x, int dx, float beta,
                 float *y, int dy)

# y = alpha * A * x + beta * y
cdef void ssymv6(CBLAS_ORDER Order, CBLAS_UPLO Uplo, float alpha, np.ndarray A,
                 np.ndarray x, float beta, np.ndarray y)

# y = alpha * A * x + beta * y
cdef void ssymv5(float alpha, np.ndarray A, np.ndarray x, float beta,
                 np.ndarray y)

# y = alpha * A * x
cdef void ssymv3(np.ndarray A, np.ndarray x, np.ndarray y)

# return = alpha * A * x
cdef np.ndarray ssymv(np.ndarray A, np.ndarray x)

# double precision symmetric matrix-vector multiply
cdef void dsymv_(CBLAS_ORDER Order, CBLAS_UPLO Uplo, int N, double alpha,
                 double *A, int lda, double *x, int dx, double beta,
                 double *y, int dy)

# y = alpha * A * x + beta * y
cdef void dsymv6(CBLAS_ORDER Order, CBLAS_UPLO Uplo, double alpha, np.ndarray A,
                 np.ndarray x, double beta, np.ndarray y)

cdef void dsymv5(double alpha, np.ndarray A, np.ndarray x, double beta,
                 np.ndarray y)

cdef void dsymv3(np.ndarray A, np.ndarray x, np.ndarray y)

cdef np.ndarray dsymv(np.ndarray A, np.ndarray x)

####

# single precision triangular matrix-vector multiply
cdef void strmv_(CBLAS_ORDER Order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA,
                 CBLAS_DIAG Diag, int N, float *A, int lda, float *x, int dx)

cdef void strmv6(CBLAS_ORDER Order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA,
                 CBLAS_DIAG Diag, np.ndarray A, np.ndarray x)

cdef void strmv3(CBLAS_TRANSPOSE TransA, np.ndarray A, np.ndarray x)

cdef void strmv(np.ndarray A, np.ndarray x)

# double precision triangular matrix-vector multiply
cdef void dtrmv_(CBLAS_ORDER Order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA,
                 CBLAS_DIAG Diag, int N, double *A, int lda, double *x, int dx)


cdef void dtrmv6(CBLAS_ORDER Order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA,
                 CBLAS_DIAG Diag, np.ndarray A, np.ndarray x)

cdef void dtrmv3(CBLAS_TRANSPOSE TransA, np.ndarray A, np.ndarray x)

cdef void dtrmv(np.ndarray A, np.ndarray x)

####

# Single precision triangular system solve.
cdef void strsv_(CBLAS_ORDER Order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA,
                 CBLAS_DIAG Diag, int N, float *A, int lda, float *x, int dx)

cdef void strsv6(CBLAS_ORDER Order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA,
                 CBLAS_DIAG Diag, np.ndarray A, np.ndarray x)

cdef void strsv3(CBLAS_TRANSPOSE TransA, np.ndarray A, np.ndarray x)

cdef void strsv(np.ndarray A, np.ndarray x)


# Double precision triangular system solve.
cdef void dtrsv_(CBLAS_ORDER Order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA,
                 CBLAS_DIAG Diag, int N, double *A, int lda, double *x, int dx)

cdef void dtrsv6(CBLAS_ORDER Order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA,
                 CBLAS_DIAG Diag, np.ndarray A, np.ndarray x)

cdef void dtrsv3(CBLAS_TRANSPOSE TransA, np.ndarray A, np.ndarray x)

cdef void dtrsv(np.ndarray A, np.ndarray x)


####

# single precision rank-1 opertion (aka outer product)
cdef void sger_(CBLAS_ORDER Order, int M, int N, float alpha, float *x, int dx,
                float *y, int dy, float *A, int lda)

# A += alpha * x y.T   (outer product)
cdef void sger4(float  alpha, np.ndarray x, np.ndarray y, np.ndarray A)

# A += x y.T   (outer product)
cdef void sger3(np.ndarray x, np.ndarray y, np.ndarray A)

# return = x y.T  (outer product)
cdef np.ndarray sger(np.ndarray x, np.ndarray y)


# double precision rank-1 opertion (aka outer product)
cdef void dger_(CBLAS_ORDER Order, int M, int N, double alpha,
                double *x, int dx, double *y, int dy, double *A, int lda)

# A += alpha * x y.T   (outer product)
cdef void dger4(double alpha, np.ndarray x, np.ndarray y, np.ndarray A)

# A += x y.T   (outer product)
cdef void dger3(np.ndarray x, np.ndarray y, np.ndarray A)

# return = x y.T  (outer product)
cdef np.ndarray dger(np.ndarray x, np.ndarray y)

####

# single precision symmetric rank-1 update: A <- alpha * x * x.T + A
cdef void ssyr_(CBLAS_ORDER Order, CBLAS_UPLO Uplo, int N, float alpha,
                float *x, int dx, float *A, int lda)

cdef void ssyr_3(float alpha, np.ndarray x, np.ndarray A)

# A <- x * x.T + A
cdef void ssyr_2(np.ndarray x, np.ndarray A)

# A = x * x.T
cdef np.ndarray ssyr(np.ndarray x)


# double precision symmetric rank-1 update: A <- alpha * x * x.T + A
cdef void dsyr_(CBLAS_ORDER Order, CBLAS_UPLO Uplo, int N, double alpha,
                double *x, int dx, double *A, int lda)

cdef void dsyr_3(double alpha, np.ndarray x, np.ndarray A)

# A <- x * x.T + A
cdef void dsyr_2(np.ndarray x, np.ndarray A)

# A = x * x.T
cdef np.ndarray dsyr(np.ndarray x)


####

# single precision symmetric rank-2 update:
# A <- alpha * x * y.T + alpha * y * x.T + A
cdef void ssyr2_(CBLAS_ORDER Order, CBLAS_UPLO Uplo, int N, float alpha,
                 float *x, int dx, float *y, int dy, float *A, int lda)

cdef void ssyr2_4(float alpha, np.ndarray x, np.ndarray y, np.ndarray A)

# A <- x * y.T + y * x.T + A
cdef void ssyr2_3(np.ndarray x, np.ndarray y, np.ndarray A)

# A = x * y.T + y * x.T
cdef np.ndarray ssyr2(np.ndarray x, np.ndarray y)


# double precision symmetric rank-2 update:
# A <- alpha * x * y.T + alpha * y * x.T + A
cdef void dsyr2_(CBLAS_ORDER Order, CBLAS_UPLO Uplo, int N, double alpha,
                 double *x, int dx, double *y, int dy, double *A, int lda)

cdef void dsyr2_4(double alpha, np.ndarray x, np.ndarray y, np.ndarray A)

# A <- x * y.T + y * x.T + A
cdef void dsyr2_3(np.ndarray x, np.ndarray y, np.ndarray A)

# A = x * y.T + y * x.T
cdef np.ndarray dsyr2(np.ndarray x, np.ndarray y)


####################################################
#
# BLAS LEVEL 3 (matrix-matrix operations)
#
####################################################


# single precision general matrix-matrix multiply
cdef void sgemm_(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA,
                 CBLAS_TRANSPOSE TransB,
                 int M, int N, int K, float alpha, float *A, int lda, float *B,
                 int ldb, float beta, float *C, int ldc)

# C = alpha * A   B   + beta * C
# C = alpha * A.T B   + beta * C
# C = alpha * A   B.T + beta * C
# C = alpha * A.T B.T + beta * C
cdef void sgemm7(CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB, float alpha,
                     np.ndarray A, np.ndarray B, float beta, np.ndarray C)

# C = alpha * A B + beta * C
cdef void sgemm5(float alpha, np.ndarray A, np.ndarray B, float beta,
                 np.ndarray C)

# C += A B
cdef void sgemm3(np.ndarray A, np.ndarray B, np.ndarray C)

# return = A B
cdef np.ndarray sgemm(np.ndarray A, np.ndarray B)



# double precision general matrix-matrix multiply
cdef void dgemm_(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA,
                 CBLAS_TRANSPOSE TransB,
                 int M, int N, int K, double alpha, double *A, int lda,
                 double *B, int ldb, double beta, double *C, int ldc)

# C = alpha * A   B   + beta * C
# C = alpha * A.T B   + beta * C
# C = alpha * A   B.T + beta * C
# C = alpha * A.T B.T + beta * C
cdef void dgemm7(CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB, double alpha,
                     np.ndarray A, np.ndarray B, double beta, np.ndarray C)

# C = alpha * A B + beta * C
cdef void dgemm5(double alpha, np.ndarray A, np.ndarray B, double beta,
                 np.ndarray C)

# C += A B
cdef void dgemm3(np.ndarray A, np.ndarray B, np.ndarray C)

# return = A B
cdef np.ndarray dgemm(np.ndarray A, np.ndarray B)


# single precision symmetric matrix multiply
cdef void ssymm_(CBLAS_ORDER Order, CBLAS_SIDE Side, CBLAS_UPLO Uplo,
                 int M, int N,
                 float alpha, float *A, int lda,
                              float *B, int ldb,
                 float beta,  float *C, int ldc)

cdef void ssymm8(CBLAS_ORDER Order, CBLAS_SIDE Side, CBLAS_UPLO Uplo,
                 float alpha, np.ndarray A, np.ndarray B,
                 float beta,  np.ndarray C)

cdef void ssymm5(float alpha, np.ndarray A, np.ndarray B, float beta,
                 np.ndarray C)

cdef void ssymm3(np.ndarray A, np.ndarray B, np.ndarray C)

cdef np.ndarray ssymm(np.ndarray A, np.ndarray B)

# double precision symmetric matrix multiply
cdef void dsymm_(CBLAS_ORDER Order, CBLAS_SIDE Side, CBLAS_UPLO Uplo,
                 int M, int N,
                 double alpha, double *A, int lda,
                               double *B, int ldb,
                 double beta,  double *C, int ldc)

cdef void dsymm8(CBLAS_ORDER Order, CBLAS_SIDE Side, CBLAS_UPLO Uplo,
                 double alpha, np.ndarray A, np.ndarray B,
                 double beta,  np.ndarray C)

cdef void dsymm5(double alpha, np.ndarray A, np.ndarray B, double beta,
                 np.ndarray C)

cdef void dsymm3(np.ndarray A, np.ndarray B, np.ndarray C)

cdef np.ndarray dsymm(np.ndarray A, np.ndarray B)


# single precision symmetric rank k update
cdef void ssyrk_(CBLAS_ORDER Order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE Trans,
                 int N, int K, float alpha, float *A, int lda, float beta,
                 float *C, int ldc)

cdef void ssyrk7(CBLAS_ORDER Order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE Trans,
                 float alpha, np.ndarray A, float beta, np.ndarray C)

cdef void ssyrk5(CBLAS_TRANSPOSE Trans, float alpha, np.ndarray A, float beta,
                 np.ndarray C)

cdef void ssyrk2(np.ndarray A, np.ndarray C)

cdef np.ndarray ssyrk(np.ndarray A)


# double precision symmetric rank k update
cdef void dsyrk_(CBLAS_ORDER Order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE Trans,
                 int N, int K, double alpha, double *A, int lda, double beta,
                 double *C, int ldc)

cdef void dsyrk7(CBLAS_ORDER Order, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE Trans,
                 double alpha, np.ndarray A, double beta, np.ndarray C)

cdef void dsyrk5(CBLAS_TRANSPOSE Trans, double alpha, np.ndarray A, double beta,
                 np.ndarray C)

cdef void dsyrk2(np.ndarray A, np.ndarray C)

cdef np.ndarray dsyrk(np.ndarray A)


######################################################################
#
# Utility functions I have put together that aren't in BLAS or LAPACK.
#
######################################################################

# create a new empty matrix
cdef np.ndarray smnewempty(int M, int N)
cdef np.ndarray dmnewempty(int M, int N)

# create a new empty vector
cdef np.ndarray svnewempty(int M)
cdef np.ndarray dvnewempty(int M)

# create a new zero matrix
cdef np.ndarray smnewzero(int M, int N)
cdef np.ndarray dmnewzero(int M, int N)

# create a new zero vector
cdef np.ndarray svnewzero(int M)
cdef np.ndarray dvnewzero(int M)

# set a matrix to all zeros
cdef void smsetzero(np.ndarray A)
cdef void dmsetzero(np.ndarray A)

# set a vector of to all zeros
cdef void svsetzero(np.ndarray x)
cdef void dvsetzero(np.ndarray x)

# matrix addition
# Y += alpha * X
cdef void smaxpy(float  alpha, np.ndarray X, np.ndarray Y)
cdef void dmaxpy(double alpha, np.ndarray X, np.ndarray Y)

