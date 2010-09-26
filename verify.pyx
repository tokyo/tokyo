cimport tokyo
import tokyo
import numpy as np
cimport numpy as np

import time
import sys

tokyo.verbose = True


speed_base = 200000 # increase to get slower but more precise speed test results
test_sizes = [4, 15, 30]
    
print
print "Tokyo BLAS wrapper verification against scipy/numpy"
print "---------------------------------------------------"
print

print
print "VERIFY CORRECTNESS BLAS 1"
print
sswap_verify()
sscal_verify()
scopy_verify()
saxpy_verify()
sdot_verify()
snrm2_verify()
sasum_verify()
isamax_verify()
print

dswap_verify()
dscal_verify()
dcopy_verify()
daxpy_verify()
ddot_verify()
dnrm2_verify()
dasum_verify()
idamax_verify()


print
print "VERIFY CORRECTNESS BLAS 2"
print

sgemv_verify(); print
sger_verify(); print

dgemv_verify(); print
dger_verify(); print


print
print "VERIFY CORRECTNESS BLAS 3"
print

sgemm_verify(); print

dgemm_verify(); print

print
print "VERIFY CORRECTNESS EXTRAS"
print
smsetzero_verify(); 
svsetzero_verify(); 
smaxpy_verify(); 
print
dmsetzero_verify(); 
dvsetzero_verify(); 
dmaxpy_verify(); 

##################################################################################


# A function for checking that different matrices from different
# computations are some sense "equal" in the verification tests.
def approx_eq( X, Y ): return abs(np.sum(X - Y)) < 1e-5


#####################################
#
# BLAS LEVEL 1 (vector operations)
#
#####################################

# vector swap: x <-> y

def sswap_verify():
    x = np.array( np.random.random( (4) ), dtype=np.float32 )
    y = np.array( np.random.random( (4) ), dtype=np.float32 )
    temp1 = x.copy()
    temp2 = y.copy()
    tokyo.sswap(x,y)
    print "sswap:  ", (approx_eq( temp1, y ) and approx_eq( temp2, x ))
    
def dswap_verify():
    x = np.array( np.random.random( (4) ), dtype=np.float64 )
    y = np.array( np.random.random( (4) ), dtype=np.float64 )
    temp1 = x.copy()
    temp2 = y.copy()
    tokyo.dswap(x,y)
    print "dswap:  ", (approx_eq( temp1, y ) and approx_eq( temp2, x ))
    
# scalar vector multiply: x *= alpha

def sscal_verify():
    x = np.array( np.random.random( (4) ), dtype=np.float32 )
    temp = 1.2 * x
    tokyo.sscal( 1.2, x)
    print "sscal:  ", approx_eq( temp, x )

def dscal_verify():
    x = np.array( np.random.random( (4) ), dtype=np.float64 )
    temp = 1.2 * x
    tokyo.dscal( 1.2, x)
    print "dscal:  ", approx_eq( temp, x )

# vector copy: y <- x

def scopy_verify():
    x = np.array( np.random.random( (4) ), dtype=np.float32 )
    y = np.array( np.random.random( (4) ), dtype=np.float32 )
    tokyo.scopy(x,y)
    print "scopy:  ", approx_eq( x, y )

def dcopy_verify():
    x = np.array( np.random.random( (4) ), dtype=np.float64 )
    y = np.array( np.random.random( (4) ), dtype=np.float64 )
    tokyo.dcopy(x,y)
    print "dcopy:  ", approx_eq( x, y )

# vector addition: y += alpha * x

def saxpy_verify():
    x = np.array( np.random.random( (5) ),   dtype=np.float32 )
    y = np.array( np.random.random( (5) ),   dtype=np.float32 )
    temp = 1.2 * x + y
    tokyo.saxpy( 1.2, x, y )
    print "saxpy:  ", approx_eq( temp, y )

def daxpy_verify():
    x = np.array( np.random.random( (5) ),   dtype=np.float64 )
    y = np.array( np.random.random( (5) ),   dtype=np.float64 )
    temp = 1.2 * x + y
    tokyo.daxpy( 1.2, x, y )
    print "daxpy:  ", approx_eq( temp, y )

# vector dot product: x.T y

def sdot_verify():
    x = np.array( np.random.random( (5) ),   dtype=np.float32 )
    y = np.array( np.random.random( (5) ),   dtype=np.float32 )
    print "sdot:   ", approx_eq( np.dot(x,y), tokyo.sdot(x,y) )

def ddot_verify():
    x = np.array( np.random.random( (5) ),   dtype=np.float64 )
    y = np.array( np.random.random( (5) ),   dtype=np.float64 )
    print "ddot:   ", approx_eq( np.dot(x,y), tokyo.ddot(x,y) )

# Euclidean norm:  ||x||_2

def snrm2_verify():
    x = np.array( np.random.random( (5) ),   dtype=np.float32 )
    print "snrm2:  ", approx_eq( np.sqrt(np.sum(np.dot(x,x))), tokyo.snrm2(x) )

def dnrm2_verify():
    x = np.array( np.random.random( (5) ),   dtype=np.float64 )
    print "dnrm2:  ", approx_eq( np.sqrt(np.sum(np.dot(x,x))), tokyo.dnrm2(x) )

# sum of absolute values: ||x||_1

def sasum_verify():
    x = np.array( np.random.random( (5) ),   dtype=np.float32 )
    print "sasum:  ", approx_eq( np.sum(np.abs(x)), tokyo.sasum(x) )

def dasum_verify():
    x = np.array( np.random.random( (5) ),   dtype=np.float64 )
    print "dasum:  ", approx_eq( np.sum(np.abs(x)), tokyo.dasum(x) )

# index of maximum absolute value element

def isamax_verify():
    x = np.array( [0.06, -0.1, -0.05, -0.001, 0.07],   dtype=np.float32 )
    print "isamax: ", (  1 == tokyo.isamax(x) )

def idamax_verify():
    x = np.array( [0.06, -0.1, -0.05, -0.001, 0.07],   dtype=np.float64 )
    print "idamax: ", (  1 == tokyo.idamax(x) )



###########################################
#
# BLAS LEVEL 2 (matrix-vector operations)
#
###########################################


# single precision matrix times vector: y = alpha * A   x + beta * y
#                                   or  y = alpha * A.T x + beta * y

def sgemv_verify():

    A = np.array( np.random.random( (4,5) ), dtype=np.float32 )
    x = np.array( np.random.random( (5) ),   dtype=np.float32 )
    y = np.array( np.random.random( (4) ),   dtype=np.float32 )

    cdef np.ndarray[float, ndim=2, mode='c'] A_
    cdef np.ndarray[float, ndim=1, mode='c'] x_, y_
    A_ = A; x_ = x; y_ = y

    temp = np.dot(A,x)
    temp2 = tokyo.sgemv( A, x )
    print "sgemv:  ", approx_eq( temp, temp2 )

    temp = np.dot(A,x)
    tokyo.sgemv3( A, x, y )
    print "sgemv3: ", approx_eq( temp, y )

    temp = 1.2*np.dot(A,x) + 2.1*y
    tokyo.sgemv5( 1.2, A, x, 2.1, y )
    print "sgemv5: ", approx_eq( temp, y )

    temp = 1.2*np.dot(A,x) + 2.1*y
    tokyo.sgemv6( tokyo.CblasNoTrans, 1.2, A, x, 2.1, y )
    print "sgemv6: ", approx_eq( temp, y )

    temp = 1.2*np.dot(A,x) + 2.1*y
    tokyo.sgemv_( tokyo.CblasRowMajor, tokyo.CblasNoTrans, 4, 5,
                       1.2, <float*>A_.data, 5, <float*>x_.data, 1,
                       2.1, <float*>y_.data, 1 )
    print "sgemv_: ", approx_eq( temp, y )


# double precision matrix times vector: y = alpha * A   x + beta * y
#                                   or  y = alpha * A.T x + beta * y


def dgemv_verify():

    A = np.array( np.random.random( (4,5) ), dtype=np.float64 )
    x = np.array( np.random.random( (5) ),   dtype=np.float64 )
    y = np.array( np.random.random( (4) ),   dtype=np.float64 )

    cdef np.ndarray[double, ndim=2, mode='c'] A_
    cdef np.ndarray[double, ndim=1, mode='c'] x_, y_
    A_ = A; x_ = x; y_ = y

    temp = np.dot(A,x)
    temp2 = tokyo.dgemv( A, x )
    print "dgemv:  ", approx_eq( temp, temp2 )

    temp = np.dot(A,x)
    tokyo.dgemv3( A, x, y )
    print "dgemv3: ", approx_eq( temp, y )

    temp = 1.2*np.dot(A,x) + 2.1*y
    tokyo.dgemv5( 1.2, A, x, 2.1, y )
    print "dgemv5: ", approx_eq( temp, y )

    temp = 1.2*np.dot(A,x) + 2.1*y
    tokyo.dgemv6( tokyo.CblasNoTrans, 1.2, A, x, 2.1, y )
    print "dgemv6: ", approx_eq( temp, y )

    temp = 1.2*np.dot(A,x) + 2.1*y
    tokyo.dgemv_( tokyo.CblasRowMajor, tokyo.CblasNoTrans, 4, 5,
                       1.2, <double*>A_.data, 5, <double*>x_.data, 1,
                       2.1, <double*>y_.data, 1 )
    print "dgemv_: ", approx_eq( temp, y )


# single precision vector outer-product: A = alpha * outer_product( x, y.T )

def sger_verify():

    x = np.array( np.random.random( (4) ),   dtype=np.float32 )
    y = np.array( np.random.random( (5) ),   dtype=np.float32 )
    A = np.array( np.random.random( (4,5) ), dtype=np.float32 )
    
    result = np.outer( x, y )
    print "sger:   ", approx_eq( result, tokyo.sger( x, y ))

    result = A + np.outer( x, y )
    tokyo.sger3( x, y, A )
    print "sger3:  ", approx_eq( result, A )

    result = A + 1.2*np.outer( x, y )
    tokyo.sger4( 1.2, x, y, A )
    print "sger4:  ", approx_eq( result, A )


# double precision vector outer-product: A = alpha * outer_product( x, y.T )

def dger_verify():

    x = np.array( np.random.random( (4) ),   dtype=np.float64 )
    y = np.array( np.random.random( (5) ),   dtype=np.float64 )
    A = np.array( np.random.random( (4,5) ), dtype=np.float64 )
    
    result = np.outer( x, y )
    print "dger:   ", approx_eq( result, tokyo.dger( x, y ))

    result = A + np.outer( x, y )
    tokyo.dger3( x, y, A )
    print "dger3:  ", approx_eq( result, A )

    result = A + 1.2*np.outer( x, y )
    tokyo.dger4( 1.2, x, y, A )
    print "dger4:  ", approx_eq( result, A )



###########################################
#
# BLAS LEVEL 3 (matrix-matrix operations)
#
###########################################


# matrix times matrix: C = alpha * A   B   + beta * C
#                  or  C = alpha * A.T B   + beta * C
#                  or  C = alpha * A   B.T + beta * C
#                  or  C = alpha * A.T B.T + beta * C
#
# single precision

def sgemm_verify():
    
    X = np.array( np.random.random( (3,4) ), dtype=np.float32 )
    Y = np.array( np.random.random( (4,5) ), dtype=np.float32 )

    print "sgemm:  ", approx_eq(np.dot( X, Y ), tokyo.sgemm( X, Y ))

    Z = np.array( np.random.random( (3,5) ), dtype=np.float32 )
    tokyo.sgemm3( X, Y, Z )
    print "sgemm3: ", approx_eq( np.dot( X, Y ), Z )
    
    Z = np.array( np.random.random( (3,5) ), dtype=np.float32 )
    result = 2.3*np.dot( X, Y ) + 1.2*Z
    tokyo.sgemm5( 2.3, X, Y, 1.2, Z )
    print "sgemm5: ", approx_eq( result, Z )

    Z = np.array( np.random.random( (3,5) ), dtype=np.float32 )
    result = 2.3*np.dot( X, Y ) + 1.2*Z
    tokyo.sgemm7( tokyo.CblasNoTrans, tokyo.CblasNoTrans, 2.3, X, Y, 1.2, Z )
    print "sgemm7: ", approx_eq( result, Z )

    
# matrix times matrix: C = alpha * A   B   + beta * C
#                  or  C = alpha * A.T B   + beta * C
#                  or  C = alpha * A   B.T + beta * C
#                  or  C = alpha * A.T B.T + beta * C
#
# double precision

def dgemm_verify():
    
    X = np.array( np.random.random( (3,4) ), dtype=np.float64 )
    Y = np.array( np.random.random( (4,5) ), dtype=np.float64 )

    print "dgemm:  ", approx_eq(np.dot( X, Y ), tokyo.dgemm( X, Y ))

    Z = np.array( np.random.random( (3,5) ), dtype=np.float64 )
    tokyo.dgemm3( X, Y, Z )
    print "dgemm3: ", approx_eq( np.dot( X, Y ), Z )
    
    Z = np.array( np.random.random( (3,5) ), dtype=np.float64 )
    result = 2.3*np.dot( X, Y ) + 1.2*Z
    tokyo.dgemm5( 2.3, X, Y, 1.2, Z )
    print "dgemm5: ", approx_eq( result, Z )

    Z = np.array( np.random.random( (3,5) ), dtype=np.float64 )
    result = 2.3*np.dot( X, Y ) + 1.2*Z
    tokyo.dgemm7( tokyo.CblasNoTrans, tokyo.CblasNoTrans, 2.3, X, Y, 1.2, Z )
    print "dgemm7: ", approx_eq( result, Z )

    

####################################################################
#
# Utility function I have put together that aren't in BLAS or LAPACK
#
####################################################################

# set a matrix of floats to all zeros

def smsetzero_verify():
    A = np.array( np.random.random( (4,4) ), dtype=np.float32 )
    B = np.zeros( (4,4) )
    tokyo.smsetzero(A)
    print "smsetzero:  ", approx_eq( A, B )

# set a matrix of floats to all zeros

def dmsetzero_verify():
    A = np.array( np.random.random( (4,4) ), dtype=np.float64 )
    B = np.zeros( (4,4) )
    tokyo.dmsetzero(A)
    print "dmsetzero:  ", approx_eq( A, B )

# set a vector of doubles to all zeros

def svsetzero_verify():
    x = np.array( np.random.random( (4) ), dtype=np.float32 )
    y = np.zeros( (4) )
    tokyo.svsetzero(x)
    print "svsetzero:  ", approx_eq( x, y )

# set a vector of doubles to all zeros

def dvsetzero_verify():
    x = np.array( np.random.random( (4) ), dtype=np.float64 )
    y = np.zeros( (4) )
    tokyo.dvsetzero(x)
    print "dvsetzero:  ", approx_eq( x, y )

# double precision matrix += scalar * matrix

def smaxpy_verify():
    X = np.array( np.random.random( (4,4) ), dtype=np.float32 )
    Y = np.array( np.random.random( (4,4) ), dtype=np.float32 )
    temp = 1.2 * X + Y
    tokyo.smaxpy( 1.2, X, Y )
    print "smaxpy:     ", approx_eq( temp, Y )

# double precision matrix += scalar * matrix

def dmaxpy_verify():
    X = np.array( np.random.random( (4,4) ), dtype=np.float64 )
    Y = np.array( np.random.random( (4,4) ), dtype=np.float64 )
    temp = 1.2 * X + Y
    tokyo.dmaxpy( 1.2, X, Y )
    print "dmaxpy:     ", approx_eq( temp, Y )

