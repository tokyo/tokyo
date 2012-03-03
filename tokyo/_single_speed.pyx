cimport tokyo
import  tokyo
import  numpy as np
cimport numpy as np

import time
import sys

tokyo.verbose = True


speed_base = 200000 # increase to get slower but more precise speed test results
test_sizes = [4, 15, 30]

print
print "Tokyo BLAS wrapper single precision speed test"
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

    print "Single precision: Vector size = " + str(size)
    print
    sswap_speed(size)
    sscal_speed(size)
    scopy_speed(size)
    saxpy_speed(size)
    sdot_speed(size)
    dsdot_speed(size)
    sdsdot_speed(size)
    snrm2_speed(size)
    sasum_speed(size)
    isamax_speed(size)
    print


print
print "SPEED TEST BLAS 2"
print

for size in test_sizes:

    print "Single Precision: Vector size = " + str(size) + \
        "  Matrix size = " + str(size) + "x" + str(size)
    print
    sgemv_speed(size); print
    sger_speed(size); print


print
print "SPEED TEST BLAS 3"
print

for size in test_sizes:

    print "Single precision: Vector size = " + str(size) + \
        "  Matrix size = " + str(size) + "x" + str(size)
    print
    sgemm_speed(size); print


print
print "SPEED TEST EXTRAS"
print

for size in test_sizes:

    print "Single precision: Vector size = " + str(size) + \
        "  Matrix size = " + str(size) + "x" + str(size)
    print
    smsetzero_speed(size)
    svsetzero_speed(size)
    smaxpy_speed(size)
    print


##################################################################################



#####################################
#
# BLAS LEVEL 1 (vector operations)
#
#####################################

# vector swap: x <-> y

cdef sswap_speed(int size):
    cdef int i, loops
    loops = speed_base*1000/size
    x = np.array( np.random.random( (size) ), dtype=np.float32 )
    y = np.array( np.random.random( (size) ), dtype=np.float32 )

    print "sswap:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.sswap( x, y )
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s" % (rate/1000)

# scalar vector multiply: x *= alpha

cdef sscal_speed(int size):
    cdef int i, loops
    loops = speed_base*2500/size
    x = np.array( np.random.random( (size) ), dtype=np.float32 )

    print "sscal:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.sscal( 1.2, x )
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s " % (rate/1000)

# vector copy: y <- x

cdef scopy_speed(int size):
    cdef int i, loops
    loops = speed_base*1500/size
    x = np.array( np.random.random( (size) ), dtype=np.float32 )
    y = np.array( np.random.random( (size) ), dtype=np.float32 )

    print "scopy:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.scopy( x, y )
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s " % (rate/1000)

# vector addition: y += alpha * x

cdef saxpy_speed( int size ):
    cdef int i, loops
    loops = speed_base*1500/size
    x = np.array( np.random.random( (size) ), dtype=np.float32 )
    y = np.array( np.random.random( (size) ), dtype=np.float32 )

    print "saxpy:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.saxpy( 1.2, x, y )
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s " % (rate/1000)

# vector dot product: x.T y

cdef sdot_speed(int size):
    cdef int i, loops
    loops = speed_base*1500/size
    x = np.array( np.random.random( (size) ), dtype=np.float32 )
    y = np.array( np.random.random( (size) ), dtype=np.float32 )

    print "sdot:       ",
    start = time.clock()
    for i in range(loops):
        tokyo.sdot( x, y )
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s " % (rate/1000)

cdef dsdot_speed(int size):
    cdef int i, loops
    loops = speed_base*1500/size
    x = np.array( np.random.random( (size) ), dtype=np.float32 )
    y = np.array( np.random.random( (size) ), dtype=np.float32 )

    print "dsdot:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.dsdot( x, y )
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s " % (rate/1000)

cdef sdsdot_speed(int size):
    cdef int i, loops
    loops = speed_base*1500/size
    x = np.array( np.random.random( (size) ), dtype=np.float32 )
    y = np.array( np.random.random( (size) ), dtype=np.float32 )
    alpha = np.float32(np.random.random())

    print "sdsdot:     ",
    start = time.clock()
    for i in range(loops):
        tokyo.sdsdot( alpha, x, y )
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s " % (rate/1000)

# Euclidean norm:  ||x||_2

cdef snrm2_speed(int size):
    cdef int i, loops
    loops = speed_base*700/size
    x = np.array( np.random.random( (size) ), dtype=np.float32 )

    print "snrm2:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.snrm2( x )
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s " % (rate/1000)

# sum of absolute values: ||x||_1

cdef sasum_speed(int size):
    cdef int i, loops
    loops = speed_base*2000/size
    x = np.array( np.random.random( (size) ), dtype=np.float32 )

    print "sasum:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.sasum( x )
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s " % (rate/1000)

# index of maximum absolute value element

cdef isamax_speed(int size):
    cdef int i, loops
    loops = speed_base*2000/size
    x = np.array( np.random.random( (size) ), dtype=np.float32 )

    print "isamax:     ",
    start = time.clock()
    for i in range(loops):
        tokyo.isamax( x )
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s " % (rate/1000)



###########################################
#
# BLAS LEVEL 2 (matrix-vector operations)
#
###########################################


# single precision matrix times vector: y = alpha * A   x + beta * y
#                                   or  y = alpha * A.T x + beta * y

cdef sgemv_speed( int size ):

    cdef int i, loops

    loops = speed_base*10/(<int>(size**1.2))

    A = np.array( np.random.random( (size,size) ), dtype=np.float32 )
    x = np.array( np.random.random( (size) ),      dtype=np.float32 )
    y = np.array( np.random.random( (size) ),      dtype=np.float32 )

    cdef np.ndarray[float, ndim=2, mode='c'] A_
    cdef np.ndarray[float, ndim=1, mode='c'] x_, y_
    A_ = A; x_ = x; y_ = y

    print "numpy.dot +: ",
    start = time.clock()
    for i in range(loops):
        y += np.dot(A,x)
    np_rate = loops/(time.clock()-start)
    print "%9.0f kc/s" % (np_rate/1000)

    loops *= 3

    print "sgemv:       ",
    start = time.clock()
    for i in range(loops):
        y = tokyo.sgemv( A, x )
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    loops *= 5

    print "sgemv3:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.sgemv3( A, x, y )
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "sgemv5:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.sgemv5( 1.2, A, x, 2.1, y )
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "sgemv6:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.sgemv6( tokyo.CblasNoTrans, 1.2, A, x, 2.1, y )
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "sgemv_:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.sgemv_( tokyo.CblasRowMajor, tokyo.CblasNoTrans, A_.shape[0], A_.shape[1],
                      1.2, <float*>A_.data, A_.shape[1], <float*>x_.data, 1,
                      2.1, <float*>y_.data, 1 )
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)


# single precision vector outer-product: A = alpha * outer_product( x, y.T )

cdef sger_speed( int size ):

    cdef int i, loops

    loops = speed_base*10/(<int>(size**1.2))

    x = np.array( np.random.random( (size) ), dtype=np.float32 )
    y = np.array( np.random.random( (size) ), dtype=np.float32 )
    Z = np.array( np.random.random( (size,size) ), dtype=np.float32 )

    cdef np.ndarray[float, ndim=1, mode='c'] x_, y_
    cdef np.ndarray[float, ndim=2, mode='c'] Z_
    x_ = x; y_ = y; Z_ = Z

    print "numpy.outer: ",
    start = time.clock()
    for i in range(loops):
        np.outer( x, y )
    np_rate = loops/(time.clock()-start)
    print "%9.0f kc/s" % (np_rate/1000)

    loops *= 15

    print "sger:        ",
    start = time.clock()
    for i in range(loops):
        tokyo.sger( x, y )
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    loops *= 2

    print "sger3:       ",
    start = time.clock()
    for i in range(loops):
        tokyo.sger3( x, y, Z )
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "sger4:       ",
    start = time.clock()
    for i in range(loops):
        tokyo.sger4( 1.0, x, y, Z )
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "sger_:       ",
    start = time.clock()
    for i in range(loops):
        tokyo.sger_( tokyo.CblasRowMajor, x_.shape[0], y_.shape[0],
          1.0, <float*>x_.data, 1, <float*>y_.data, 1, <float*>Z_.data, Z_.shape[1])
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
# single precision

cdef sgemm_speed( int size ):

    cdef int i, loops

    loops = speed_base*150/(size*size)

    X = np.array( np.random.random( (size,size) ), dtype=np.float32 )
    Y = np.array( np.random.random( (size,size) ), dtype=np.float32 )
    Z = np.array( np.random.random( (size,size) ), dtype=np.float32 )

    cdef np.ndarray[float, ndim=2, mode='c'] X_, Y_, Z_
    X_ = X; Y_ = Y; Z_ = Z

    print "numpy.dot: ",
    start = time.clock()
    for i in range(loops): np.dot( X, Y )
    np_rate = loops/(time.clock()-start)
    print "%9.0f kc/s" % (np_rate/1000)

    print "sgemm:     ",
    start = time.clock()
    for i in range(loops):
        tokyo.sgemm( X, Y )
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "sgemm3:    ",
    start = time.clock()
    for i in range(loops):
        tokyo.sgemm3( X, Y, Z )
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "sgemm5:    ",
    start = time.clock()
    for i in range(loops):
        tokyo.sgemm5( 1.0, X, Y, 0.0, Z )
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "sgemm7:    ",
    start = time.clock()
    for i in range(loops):
        tokyo.sgemm7( tokyo.CblasNoTrans, tokyo.CblasNoTrans, 1.0, X, Y, 0.0, Z )
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "sgemm_:    ",
    start = time.clock()
    for i in range(loops):
        tokyo.sgemm_( tokyo.CblasRowMajor, tokyo.CblasNoTrans, tokyo.CblasNoTrans,
                size, size, size, 1.0, <float*>X_.data, size, <float*>Y_.data, size,
                     0.0, <float*>Z_.data, size )
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)


####################################################################
#
# Utility function I have put together that aren't in BLAS or LAPACK
#
####################################################################

# set a matrix of floats to all zeros

cdef smsetzero_speed(int size):
    cdef int i, loops
    loops = speed_base*5000/(size*size)
    A = np.array( np.random.random( (size,size) ), dtype=np.float32 )

    print "smsetzero:  ",
    start = time.clock()
    for i in range(loops):
        tokyo.smsetzero( A )
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s " % (rate/1000)


# set a vector of floats to all zeros

cdef svsetzero_speed(int size):
    cdef int i, loops
    loops = speed_base*5000/size
    x = np.array( np.random.random( (size) ), dtype=np.float32 )

    print "svsetzero:  ",
    start = time.clock()
    for i in range(loops):
        tokyo.svsetzero( x )
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s " % (rate/1000)


# single precision matrix += scalar * matrix

cdef smaxpy_speed( int size ):
    cdef int i, loops
    loops = speed_base*10000/(size*size)
    X = np.array( np.random.random( (size,size) ), dtype=np.float32 )
    Y = np.array( np.random.random( (size,size) ), dtype=np.float32 )

    print "smaxpy:     ",
    start = time.clock()
    for i in range(loops):
        tokyo.smaxpy( 1.2, X, Y )
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s " % (rate/1000)


