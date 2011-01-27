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
print "Tokyo BLAS wrapper double precision speed test"
print "----------------------------------------------"
print
print "Make sure your CPU isn't doing frequency scaling, otherwise"
print "the speed results here might be all messed up.   A few percent"
print "variation in speed results from run to run is normal."
print
print "Speed is given in thousands of calls per second (kc/s), and in"
print "some cases how many times faster than scipy/numpy the call is."
print "Naturally the advantage is greatest on small vectors/matrices"
print "because that's when the numpy/scipy overhead is high relative"
print "to the total computation cost."

print
print "SPEED TEST BLAS 1"
print

for size in test_sizes:

    print "Double precision: Vector size = " + str(size)
    print
    dswap_speed(size)
    dscal_speed(size)
    dcopy_speed(size)
    daxpy_speed(size)
    ddot_speed(size)
    dnrm2_speed(size)
    dasum_speed(size)
    idamax_speed(size)
    print


print
print "SPEED TEST BLAS 2"
print

for size in test_sizes:

    print "Double precision: Vector size = " + str(size) + \
        "  Matrix size = " + str(size) + "x" + str(size)
    print
    dgemv_speed(size); print
    dger_speed(size); print

print
print "SPEED TEST BLAS 3"
print

for size in test_sizes:

    print "Double precision: Vector size = " + str(size) + \
        "  Matrix size = " + str(size) + "x" + str(size)
    print
    dgemm_speed(size); print


print
print "SPEED TEST EXTRAS"
print

for size in test_sizes:

    print "Double precision: Vector size = " + str(size) + \
        "  Matrix size = " + str(size) + "x" + str(size)
    print
    dmsetzero_speed(size)
    dvsetzero_speed(size)
    dmaxpy_speed(size)
    print


##################################################################################



#####################################
#
# BLAS LEVEL 1 (vector operations)
#
#####################################

# vector swap: x <-> y

def dswap_speed(int size):
    cdef int i, loops
    loops = speed_base*1000/size
    x = np.array( np.random.random( (size) ), dtype=np.float64 )
    y = np.array( np.random.random( (size) ), dtype=np.float64 )

    print "dswap:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.dswap( x, y )
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s" % (rate/1000)


# scalar vector multiply: x *= alpha

def dscal_speed(int size):
    cdef int i, loops
    loops = speed_base*2500/size
    x = np.array( np.random.random( (size) ), dtype=np.float64 )

    print "dscal:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.dscal( 1.2, x )
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s " % (rate/1000)


# vector copy: y <- x

def dcopy_speed(int size):
    cdef int i, loops
    loops = speed_base*1500/size
    x = np.array( np.random.random( (size) ), dtype=np.float64 )
    y = np.array( np.random.random( (size) ), dtype=np.float64 )

    print "dcopy:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.dcopy( x, y )
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s " % (rate/1000)


# vector addition: y += alpha * x

def daxpy_speed( int size ):
    cdef int i, loops
    loops = speed_base*1500/size
    x = np.array( np.random.random( (size) ), dtype=np.float64 )
    y = np.array( np.random.random( (size) ), dtype=np.float64 )

    print "daxpy:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.daxpy( 1.2, x, y )
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s " % (rate/1000)


# vector dot product: x.T y

def ddot_speed(int size):
    cdef int i, loops
    loops = speed_base*1500/size
    x = np.array( np.random.random( (size) ), dtype=np.float64 )
    y = np.array( np.random.random( (size) ), dtype=np.float64 )

    print "ddot:       ",
    start = time.clock()
    for i in range(loops):
        tokyo.ddot( x, y )
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s " % (rate/1000)


# Euclidean norm:  ||x||_2

def dnrm2_speed(int size):
    cdef int i, loops
    loops = speed_base*700/size
    x = np.array( np.random.random( (size) ), dtype=np.float64 )

    print "dnrm2:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.dnrm2( x )
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s " % (rate/1000)


# sum of absolute values: ||x||_1

def dasum_speed(int size):
    cdef int i, loops
    loops = speed_base*2000/size
    x = np.array( np.random.random( (size) ), dtype=np.float64 )

    print "dasum:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.dasum( x )
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s " % (rate/1000)


# index of maximum absolute value element

def idamax_speed(int size):
    cdef int i, loops
    loops = speed_base*2000/size
    x = np.array( np.random.random( (size) ), dtype=np.float64 )

    print "idamax:     ",
    start = time.clock()
    for i in range(loops):
        tokyo.idamax( x )
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s " % (rate/1000)




###########################################
#
# BLAS LEVEL 2 (matrix-vector operations)
#
###########################################


# double precision matrix times vector: y = alpha * A   x + beta * y
#                                   or  y = alpha * A.T x + beta * y

def dgemv_speed( int size ):

    cdef int i, loops

    loops = speed_base*10/(<int>(size**1.2))

    A = np.array( np.random.random( (size,size) ), dtype=np.float64 )
    x = np.array( np.random.random( (size) ),      dtype=np.float64 )
    y = np.array( np.random.random( (size) ),      dtype=np.float64 )

    cdef np.ndarray[double, ndim=2, mode='c'] A_
    cdef np.ndarray[double, ndim=1, mode='c'] x_, y_
    A_ = A; x_ = x; y_ = y

    print "numpy.dot +: ",
    start = time.clock()
    for i in range(loops):
        y += np.dot(A,x)
    np_rate = loops/(time.clock()-start)
    print "%9.0f kc/s" % (np_rate/1000)

    loops *= 3

    print "dgemv:       ",
    start = time.clock()
    for i in range(loops):
        y = tokyo.dgemv( A, x )
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    loops *= 5

    print "dgemv3:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.dgemv3( A, x, y )
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "dgemv5:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.dgemv5( 1.2, A, x, 2.1, y )
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "dgemv6:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.dgemv6( tokyo.CblasNoTrans, 1.2, A, x, 2.1, y )
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "dgemv_:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.dgemv_( tokyo.CblasRowMajor, tokyo.CblasNoTrans, A_.shape[0], A_.shape[1],
                      1.2, <double*>A_.data, A_.shape[1], <double*>x_.data, 1,
                      2.1, <double*>y_.data, 1 )
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)



# double precision vector outer-product: A = alpha * outer_product( x, y.T )

def dger_speed( int size ):

    cdef int i, loops

    loops = speed_base*10/(<int>(size**1.2))

    x = np.array( np.random.random( (size) ), dtype=np.float64 )
    y = np.array( np.random.random( (size) ), dtype=np.float64 )
    Z = np.array( np.random.random( (size,size) ), dtype=np.float64 )

    cdef np.ndarray[double, ndim=1, mode='c'] x_, y_
    cdef np.ndarray[double, ndim=2, mode='c'] Z_
    x_ = x; y_ = y; Z_ = Z

    print "numpy.outer: ",
    start = time.clock()
    for i in range(loops):
        np.outer( x, y )
    np_rate = loops/(time.clock()-start)
    print "%9.0f kc/s" % (np_rate/1000)

    loops *= 15

    print "dger:        ",
    start = time.clock()
    for i in range(loops):
        tokyo.dger( x, y )
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    loops *= 2

    print "dger3:       ",
    start = time.clock()
    for i in range(loops):
        tokyo.dger3( x, y, Z )
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "dger4:       ",
    start = time.clock()
    for i in range(loops):
        tokyo.dger4( 1.0, x, y, Z )
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "dger_:       ",
    start = time.clock()
    for i in range(loops):
        tokyo.dger_( tokyo.CblasRowMajor, x_.shape[0], y_.shape[0],
          1.0, <double*>x_.data, 1, <double*>y_.data, 1, <double*>Z_.data, Z_.shape[1])
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)



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
# double precision

def dgemm_speed( int size ):

    cdef int i, loops

    loops = speed_base*150/(size*size)

    X = np.array( np.random.random( (size,size) ), dtype=np.float64 )
    Y = np.array( np.random.random( (size,size) ), dtype=np.float64 )
    Z = np.array( np.random.random( (size,size) ), dtype=np.float64 )

    cdef np.ndarray[double, ndim=2, mode='c'] X_, Y_, Z_
    X_ = X; Y_ = Y; Z_ = Z

    print "numpy.dot: ",
    start = time.clock()
    for i in range(loops): np.dot( X, Y )
    np_rate = loops/(time.clock()-start)
    print "%9.0f kc/s" % (np_rate/1000)

    print "dgemm:     ",
    start = time.clock()
    for i in range(loops):
        tokyo.dgemm( X, Y )
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "dgemm3:    ",
    start = time.clock()
    for i in range(loops):
        tokyo.dgemm3( X, Y, Z )
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "dgemm5:    ",
    start = time.clock()
    for i in range(loops):
        tokyo.dgemm5( 1.0, X, Y, 0.0, Z )
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "dgemm7:    ",
    start = time.clock()
    for i in range(loops):
        tokyo.dgemm7( tokyo.CblasNoTrans, tokyo.CblasNoTrans, 1.0, X, Y, 0.0, Z )
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "dgemm_:    ",
    start = time.clock()
    for i in range(loops):
        tokyo.dgemm_( tokyo.CblasRowMajor, tokyo.CblasNoTrans, tokyo.CblasNoTrans,
                size, size, size, 1.0, <double*>X_.data, size, <double*>Y_.data, size,
                     0.0, <double*>Z_.data, size )
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)


####################################################################
#
# Utility function I have put together that aren't in BLAS or LAPACK
#
####################################################################


# set a matrix of double to all zeros

def dmsetzero_speed(int size):
    cdef int i, loops
    loops = speed_base*5000/(size*size)
    A = np.array( np.random.random( (size,size) ), dtype=np.float64 )

    print "dmsetzero:  ",
    start = time.clock()
    for i in range(loops):
        tokyo.dmsetzero( A )
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s " % (rate/1000)


# set a vector of doubles to all zeros

def dvsetzero_speed(int size):
    cdef int i, loops
    loops = speed_base*5000/size
    x = np.array( np.random.random( (size) ), dtype=np.float64 )

    print "dvsetzero:  ",
    start = time.clock()
    for i in range(loops):
        tokyo.dvsetzero( x )
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s " % (rate/1000)


# double precision matrix += scalar * matrix

def dmaxpy_speed( int size ):
    cdef int i, loops
    loops = speed_base*10000/(size*size)
    X = np.array( np.random.random( (size,size) ), dtype=np.float64 )
    Y = np.array( np.random.random( (size,size) ), dtype=np.float64 )

    print "dmaxpy:     ",
    start = time.clock()
    for i in range(loops):
        tokyo.dmaxpy( 1.2, X, Y )
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s " % (rate/1000)

