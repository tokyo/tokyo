cimport tokyo
import  tokyo
import  numpy as np
cimport numpy as np

import time
import sys

tokyo.verbose = True


speed_base = 200000 # increase for slower but more precise speed test results
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
    ssymv_speed(size); print
    strmv_speed(size); print
    strsv_speed(size); print
    sger_speed(size);  print
    ssyr_speed(size);  print
    ssyr2_speed(size); print


print
print "SPEED TEST BLAS 3"
print

for size in test_sizes:

    print "Single precision: Vector size = " + str(size) + \
        "  Matrix size = " + str(size) + "x" + str(size)
    print
    sgemm_speed(size);  print
    ssymm_speed(size);  print
    ssyrk_speed(size);  print
    ssyr2k_speed(size); print
    strmm_speed(size);  print


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


###############################################################################


#####################################
#
# BLAS LEVEL 1 (vector operations)
#
#####################################

# vector swap: x <-> y

cdef sswap_speed(int size):
    cdef int i, loops
    loops = speed_base*1000/size
    x = np.array(np.random.random((size)), dtype=np.float32)
    y = np.array(np.random.random((size)), dtype=np.float32)

    print "sswap:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.sswap(x, y)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s" % (rate/1000)

# scalar vector multiply: x *= alpha

cdef sscal_speed(int size):
    cdef int i, loops
    loops = speed_base*2500/size
    x = np.array(np.random.random((size)), dtype=np.float32)

    print "sscal:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.sscal(1.2, x)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s " % (rate/1000)

# vector copy: y <- x

cdef scopy_speed(int size):
    cdef int i, loops
    loops = speed_base*1500/size
    x = np.array(np.random.random((size)), dtype=np.float32)
    y = np.array(np.random.random((size)), dtype=np.float32)

    print "scopy:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.scopy(x, y)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s " % (rate/1000)

# vector addition: y += alpha * x

cdef saxpy_speed(int size):
    cdef int i, loops
    loops = speed_base*1500/size
    x = np.array(np.random.random((size)), dtype=np.float32)
    y = np.array(np.random.random((size)), dtype=np.float32)

    print "saxpy:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.saxpy(1.2, x, y)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s " % (rate/1000)

# vector dot product: x.T y

cdef sdot_speed(int size):
    cdef int i, loops
    loops = speed_base*1500/size
    x = np.array(np.random.random((size)), dtype=np.float32)
    y = np.array(np.random.random((size)), dtype=np.float32)

    print "sdot:       ",
    start = time.clock()
    for i in range(loops):
        tokyo.sdot(x, y)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s " % (rate/1000)

cdef dsdot_speed(int size):
    cdef int i, loops
    loops = speed_base*1500/size
    x = np.array(np.random.random((size)), dtype=np.float32)
    y = np.array(np.random.random((size)), dtype=np.float32)

    print "dsdot:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.dsdot(x, y)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s " % (rate/1000)

cdef sdsdot_speed(int size):
    cdef int i, loops
    loops = speed_base*1500/size
    x = np.array(np.random.random((size)), dtype=np.float32)
    y = np.array(np.random.random((size)), dtype=np.float32)
    alpha = np.float32(np.random.random())

    print "sdsdot:     ",
    start = time.clock()
    for i in range(loops):
        tokyo.sdsdot(alpha, x, y)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s " % (rate/1000)

# Euclidean norm:  ||x||_2

cdef snrm2_speed(int size):
    cdef int i, loops
    loops = speed_base*700/size
    x = np.array(np.random.random((size)), dtype=np.float32)

    print "snrm2:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.snrm2(x)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s " % (rate/1000)

# sum of absolute values: ||x||_1

cdef sasum_speed(int size):
    cdef int i, loops
    loops = speed_base*2000/size
    x = np.array(np.random.random((size)), dtype=np.float32)

    print "sasum:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.sasum(x)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s " % (rate/1000)

# index of maximum absolute value element

cdef isamax_speed(int size):
    cdef int i, loops
    loops = speed_base*2000/size
    x = np.array(np.random.random((size)), dtype=np.float32)

    print "isamax:     ",
    start = time.clock()
    for i in range(loops):
        tokyo.isamax(x)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s " % (rate/1000)


###########################################
#
# BLAS LEVEL 2 (matrix-vector operations)
#
###########################################


# single precision matrix times vector: y = alpha * A   x + beta * y
#                                   or  y = alpha * A.T x + beta * y

cdef sgemv_speed(int size):

    cdef int i, loops

    loops = speed_base*10/(<int>(size**1.2))

    A = np.array(np.random.random((size,size)), dtype=np.float32)
    x = np.array(np.random.random((size)),      dtype=np.float32)
    y = np.array(np.random.random((size)),      dtype=np.float32)

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
        y = tokyo.sgemv(A, x)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    loops *= 5

    print "sgemv3:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.sgemv3(A, x, y)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "sgemv5:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.sgemv5(1.2, A, x, 2.1, y)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "sgemv6:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.sgemv6(tokyo.CblasNoTrans, 1.2, A, x, 2.1, y)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "sgemv_:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.sgemv_(tokyo.CblasRowMajor, tokyo.CblasNoTrans,
                     A_.shape[0], A_.shape[1],
                     1.2, <float*>A_.data, A_.shape[1], <float*>x_.data, 1,
                     2.1, <float*>y_.data, 1)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)


# Single precision symmetric-matrix vector product: y = alpha * A * x + beta * y

cdef ssymv_speed(int size):

    cdef int i, loops

    loops = speed_base*10/(<int>(size**1.2))

    A = np.array(np.random.random((size,size)), dtype=np.float32)
    x = np.array(np.random.random((size)),      dtype=np.float32)
    y = np.array(np.random.random((size)),      dtype=np.float32)
    A = (A + A.T)/2

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

    print "ssymv:       ",
    start = time.clock()
    for i in range(loops):
        y = tokyo.ssymv(A, x)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    loops *= 5

    print "ssymv3:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.ssymv3(A, x, y)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "ssymv5:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.ssymv5(1.2, A, x, 2.1, y)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "ssymv6:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.ssymv6(tokyo.CblasRowMajor, tokyo.CblasLower, 1.2, A, x, 2.1, y)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "ssymv_:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.ssymv_(tokyo.CblasRowMajor, tokyo.CblasLower,
                     A_.shape[1], 1.2, <float*>A_.data, A_.shape[1],
                     <float*>x_.data, 1, 2.1, <float*>y_.data, 1)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)


# Single precision triangular matrix vector product: x <- A * x

cdef strmv_speed(int size):

    cdef int i, loops

    loops = speed_base*10/(<int>(size**1.2))

    A = np.array(np.random.random((size,size)), dtype=np.float32)
    x = np.array(np.random.random((size)),      dtype=np.float32)
    for i in range(size):
        for j in range(size):
            if j > i: A[i,j] = 0

    cdef np.ndarray[float, ndim=2, mode='c'] A_
    cdef np.ndarray[float, ndim=1, mode='c'] x_
    A_ = A; x_ = x

    print "numpy.dot:   ",
    start = time.clock()
    for i in range(loops):
        x = np.dot(A,x)
    np_rate = loops/(time.clock()-start)
    print "%9.0f kc/s" % (np_rate/1000)

    loops *= 3

    print "strmv:       ",
    start = time.clock()
    for i in range(loops):
        tokyo.strmv(A, x)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    loops *= 5

    print "strmv3:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.strmv3(tokyo.CblasNoTrans, A, x)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "strmv6:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.strmv6(tokyo.CblasRowMajor, tokyo.CblasLower, tokyo.CblasNoTrans,
                     tokyo.CblasNonUnit, A, x)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "strmv_:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.strmv_(tokyo.CblasRowMajor, tokyo.CblasLower, tokyo.CblasNoTrans,
                     tokyo.CblasNonUnit, A_.shape[1], <float*>A_.data,
                     A_.shape[1], <float*>x_.data, 1)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)


# Single precision triangular system solve: x <- inv(A) * x

cdef strsv_speed(int size):

    cdef int i, loops

    loops = speed_base*10/(<int>(size**1.2))

    A = np.array(np.random.random((size,size)), dtype=np.float32)
    x = np.array(np.random.random((size)),      dtype=np.float32)
    for i in range(size):
        for j in range(size):
            if j > i: A[i,j] = 0

    cdef np.ndarray[float, ndim=2, mode='c'] A_
    cdef np.ndarray[float, ndim=1, mode='c'] x_
    A_ = A; x_ = x

    loops *= 3

    print "strsv:       ",
    start = time.clock()
    for i in range(loops):
        tokyo.strsv(A, x)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s" % (rate/1000)

    loops *= 5

    print "strsv3:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.strsv3(tokyo.CblasNoTrans, A, x)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s" % (rate/1000)

    print "strsv6:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.strsv6(tokyo.CblasRowMajor, tokyo.CblasLower, tokyo.CblasNoTrans,
                     tokyo.CblasNonUnit, A, x)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s" % (rate/1000)

    print "strsv_:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.strsv_(tokyo.CblasRowMajor, tokyo.CblasLower, tokyo.CblasNoTrans,
                     tokyo.CblasNonUnit, A_.shape[1], <float*>A_.data,
                     A_.shape[1], <float*>x_.data, 1)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s" % (rate/1000)


# single precision vector outer-product: A = alpha * outer_product(x, y.T)

cdef sger_speed(int size):

    cdef int i, loops

    loops = speed_base*10/(<int>(size**1.2))

    x = np.array(np.random.random((size)), dtype=np.float32)
    y = np.array(np.random.random((size)), dtype=np.float32)
    Z = np.array(np.random.random((size,size)), dtype=np.float32)

    cdef np.ndarray[float, ndim=1, mode='c'] x_, y_
    cdef np.ndarray[float, ndim=2, mode='c'] Z_
    x_ = x; y_ = y; Z_ = Z

    print "numpy.outer: ",
    start = time.clock()
    for i in range(loops):
        np.outer(x, y)
    np_rate = loops/(time.clock()-start)
    print "%9.0f kc/s" % (np_rate/1000)

    loops *= 15

    print "sger:        ",
    start = time.clock()
    for i in range(loops):
        tokyo.sger(x, y)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    loops *= 2

    print "sger3:       ",
    start = time.clock()
    for i in range(loops):
        tokyo.sger3(x, y, Z)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "sger4:       ",
    start = time.clock()
    for i in range(loops):
        tokyo.sger4(1.0, x, y, Z)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "sger_:       ",
    start = time.clock()
    for i in range(loops):
        tokyo.sger_(tokyo.CblasRowMajor, x_.shape[0], y_.shape[0],
                    1.0, <float*>x_.data, 1, <float*>y_.data, 1,
                    <float*>Z_.data, Z_.shape[1])
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)


# single precision symmetric rank 1 update: A <- alpha * x * x.T + A

cdef ssyr_speed(int size):

    cdef int i, loops

    loops = speed_base*10/(<int>(size**1.2))

    x = np.array(np.random.random((size)), dtype=np.float32)
    A = np.array(np.random.random((size,size)), dtype=np.float32)

    cdef np.ndarray[float, ndim=1, mode='c'] x_
    cdef np.ndarray[float, ndim=2, mode='c'] A_
    x_ = x; A_ = A

    print "numpy.outer: ",
    start = time.clock()
    for i in range(loops):
        np.outer(x, x)
    np_rate = loops/(time.clock()-start)
    print "%9.0f kc/s" % (np_rate/1000)

    loops *= 15

    print "ssyr:        ",
    start = time.clock()
    for i in range(loops):
        tokyo.ssyr(x)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    loops *= 2

    print "ssyr_2:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.ssyr_2(x, A)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "ssyr_3:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.ssyr_3(1.0, x, A)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "ssyr_:       ",
    start = time.clock()
    for i in range(loops):
        tokyo.ssyr_(tokyo.CblasRowMajor, tokyo.CblasLower,
                    x.shape[0], 1.0, <float*>x_.data, 1,
                    <float*>A_.data, A_.shape[1])
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)


# single precision symmetric rank 2 update:
# A <- alpha * x * y.T + alpha * y * x.T + A

cdef ssyr2_speed(int size):

    cdef int i, loops

    loops = speed_base*10/(<int>(size**1.2))

    x = np.array(np.random.random((size)), dtype=np.float32)
    y = np.array(np.random.random((size)), dtype=np.float32)
    A = np.array(np.random.random((size,size)), dtype=np.float32)

    cdef np.ndarray[float, ndim=1, mode='c'] x_, y_
    cdef np.ndarray[float, ndim=2, mode='c'] A_
    x_ = x; y_ = y; A_ = A

    print "numpy.outer: ",
    start = time.clock()
    for i in range(loops):
        np.outer(x, y) + np.outer(y, x)
    np_rate = loops/(time.clock()-start)
    print "%9.0f kc/s" % (np_rate/1000)

    loops *= 15

    print "ssyr2:       ",
    start = time.clock()
    for i in range(loops):
        tokyo.ssyr2(x, y)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    loops *= 2

    print "ssyr2_3:     ",
    start = time.clock()
    for i in range(loops):
        tokyo.ssyr2_3(x, y, A)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "ssyr2_4:     ",
    start = time.clock()
    for i in range(loops):
        tokyo.ssyr2_4(1.0, x, y, A)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "ssyr2_:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.ssyr2_(tokyo.CblasRowMajor, tokyo.CblasLower,
                    x.shape[0], 1.0, <float*>x_.data, 1,
                    <float*>y_.data, 1,
                    <float*>A_.data, A_.shape[1])
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

cdef sgemm_speed(int size):

    cdef int i, loops

    loops = speed_base*150/(size*size)

    X = np.array(np.random.random((size,size)), dtype=np.float32)
    Y = np.array(np.random.random((size,size)), dtype=np.float32)
    Z = np.array(np.random.random((size,size)), dtype=np.float32)

    cdef np.ndarray[float, ndim=2, mode='c'] X_, Y_, Z_
    X_ = X; Y_ = Y; Z_ = Z

    print "numpy.dot: ",
    start = time.clock()
    for i in range(loops): np.dot(X, Y)
    np_rate = loops/(time.clock()-start)
    print "%9.0f kc/s" % (np_rate/1000)

    print "sgemm:     ",
    start = time.clock()
    for i in range(loops):
        tokyo.sgemm(X, Y)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "sgemm3:    ",
    start = time.clock()
    for i in range(loops):
        tokyo.sgemm3(X, Y, Z)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "sgemm5:    ",
    start = time.clock()
    for i in range(loops):
        tokyo.sgemm5(1.0, X, Y, 0.0, Z)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "sgemm7:    ",
    start = time.clock()
    for i in range(loops):
        tokyo.sgemm7(tokyo.CblasNoTrans, tokyo.CblasNoTrans, 1.0, X, Y, 0.0, Z)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "sgemm_:    ",
    start = time.clock()
    for i in range(loops):
        tokyo.sgemm_(tokyo.CblasRowMajor, tokyo.CblasNoTrans,
                     tokyo.CblasNoTrans, size, size, size, 1.0,
                     <float*>X_.data, size, <float*>Y_.data, size,
                     0.0, <float*>Z_.data, size)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)


# Matrix times matrix: C = alpha * A * B + beta * C
#                  or: C = alpha * B * A + beta * C
# where A = A.T.
#
# single precision

cdef ssymm_speed(int size):

    cdef int i, loops

    loops = speed_base*150/(size*size)

    A = np.array(np.random.random((size,size)), dtype=np.float32)
    B = np.array(np.random.random((size,size)), dtype=np.float32)
    C = np.array(np.random.random((size,size)), dtype=np.float32)
    A = (A + A.T)/2

    cdef np.ndarray[float, ndim=2, mode='c'] A_, B_, C_
    A_ = A; B_ = B; C_ = C

    print "numpy.dot: ",
    start = time.clock()
    for i in range(loops): np.dot(A, B)
    np_rate = loops/(time.clock()-start)
    print "%9.0f kc/s" % (np_rate/1000)

    print "ssymm:     ",
    start = time.clock()
    for i in range(loops):
        tokyo.ssymm(A, B)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "ssymm3:    ",
    start = time.clock()
    for i in range(loops):
        tokyo.ssymm3(A, B, C)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "ssymm5:    ",
    start = time.clock()
    for i in range(loops):
        tokyo.ssymm5(1.0, A, B, 0.0, C)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "ssymm8:    ",
    start = time.clock()
    for i in range(loops):
        tokyo.ssymm8(tokyo.CblasRowMajor, tokyo.CblasLeft, tokyo.CblasLower,
                     1.0, A, B, 0.0, C)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "ssymm_:    ",
    start = time.clock()
    for i in range(loops):
        tokyo.ssymm_(tokyo.CblasRowMajor, tokyo.CblasLeft,
                     tokyo.CblasLower, size, size, 1.0,
                     <float*>A_.data, size, <float*>B_.data, size,
                     0.0, <float*>C_.data, size)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)


# Symmetric rank k update: C <- alpha * A * A.T + beta * C
#                      or: C <- alpha * A.T * A + beta * C
#
# single precision

cdef ssyrk_speed(int size):

    cdef int i, loops

    loops = speed_base*150/(size*size)

    A = np.array(np.random.random((size,size)), dtype=np.float32)
    C = np.array(np.random.random((size,size)), dtype=np.float32)

    cdef np.ndarray[float, ndim=2, mode='c'] A_, C_
    A_ = A; C_ = C

    print "numpy.dot: ",
    start = time.clock()
    for i in range(loops): np.dot(A, A.T)
    np_rate = loops/(time.clock()-start)
    print "%9.0f kc/s" % (np_rate/1000)

    print "ssyrk:     ",
    start = time.clock()
    for i in range(loops):
        tokyo.ssyrk(A)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "ssyrk2:    ",
    start = time.clock()
    for i in range(loops):
        tokyo.ssyrk2(A, C)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "ssyrk5:    ",
    start = time.clock()
    for i in range(loops):
        tokyo.ssyrk5(tokyo.CblasNoTrans, 1.0, A, 0.0, C)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "ssyrk7:    ",
    start = time.clock()
    for i in range(loops):
        tokyo.ssyrk7(tokyo.CblasRowMajor, tokyo.CblasLower,
                     tokyo.CblasNoTrans, 1.0, A, 0.0, C)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "ssyrk_:    ",
    start = time.clock()
    for i in range(loops):
        tokyo.ssyrk_(tokyo.CblasRowMajor, tokyo.CblasLower,
                     tokyo.CblasNoTrans, size, size, 1.0,
                     <float*>A_.data, size,
                     0.0, <float*>C_.data, size)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)


# Symmetric rank-2k update: C <- alpha * A * B.T + alpha * B * A.T + beta * C
#                       or: C <- alpha * A.T * B + alpha * B.T * A + beta * C

cdef ssyr2k_speed(int size):

    cdef int i, loops

    loops = speed_base*150/(size*size)

    A = np.array(np.random.random((size,size)), dtype=np.float32)
    B = np.array(np.random.random((size,size)), dtype=np.float32)
    C = np.array(np.random.random((size,size)), dtype=np.float32)

    cdef np.ndarray[float, ndim=2, mode='c'] A_, B_, C_
    A_ = A; B_ = B ; C_ = C

    print "numpy.dot: ",
    start = time.clock()
    for i in range(loops):
        np.dot(A, B.T) + np.dot(B, A.T)
    np_rate = loops/(time.clock()-start)
    print "%9.0f kc/s" % (np_rate/1000)

    print "ssyr2k:    ",
    start = time.clock()
    for i in range(loops):
        tokyo.ssyr2k(A, B)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "ssyr2k3:   ",
    start = time.clock()
    for i in range(loops):
        tokyo.ssyr2k3(A, B, C)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "ssyr2k6:   ",
    start = time.clock()
    for i in range(loops):
        tokyo.ssyr2k6(tokyo.CblasNoTrans, 1.0, A, B, 0.0, C)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "ssyr2k8:   ",
    start = time.clock()
    for i in range(loops):
        tokyo.ssyr2k8(tokyo.CblasRowMajor, tokyo.CblasLower,
                     tokyo.CblasNoTrans, 1.0, A, B, 0.0, C)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "ssyr2k_:   ",
    start = time.clock()
    for i in range(loops):
        tokyo.ssyr2k_(tokyo.CblasRowMajor, tokyo.CblasLower,
                     tokyo.CblasNoTrans, size, size, 1.0,
                     <float*>A_.data, size, <float*>B_.data, size,
                     0.0, <float*>C_.data, size)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)


#     B = alpha * A * B  or  B = alpha * A.T * B
# or  B = alpha * B * A  or  B = alpha * B * A.T
#
# where A is triangular.

cdef strmm_speed(int size):

    cdef int i, loops

    loops = speed_base*150/(size*size)

    A = np.array(np.random.random((size,size)), dtype=np.float32)
    B = np.array(np.random.random((size,size)), dtype=np.float32)
    ti = np.triu_indices(size, 1)
    A[ti] = 0

    cdef np.ndarray[float, ndim=2, mode='c'] A_, B_
    A_ = A; B_ = B

    print "numpy.dot: ",
    start = time.clock()
    for i in range(loops):
        np.dot(A, B)
    np_rate = loops/(time.clock()-start)
    print "%9.0f kc/s" % (np_rate/1000)

    print "strmm:     ",
    start = time.clock()
    for i in range(loops):
        tokyo.strmm(A, B)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "strmm3:    ",
    start = time.clock()
    for i in range(loops):
        tokyo.strmm3(1.0, A, B)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "strmm5:    ",
    start = time.clock()
    for i in range(loops):
        tokyo.strmm5(tokyo.CblasLeft, tokyo.CblasNoTrans, 1.0, A, B)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "strmm8:    ",
    start = time.clock()
    for i in range(loops):
        tokyo.strmm8(tokyo.CblasRowMajor, tokyo.CblasLeft, tokyo.CblasLower,
                     tokyo.CblasNoTrans, tokyo.CblasNonUnit, 1.0, A, B)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "strmm_:    ",
    start = time.clock()
    for i in range(loops):
        tokyo.strmm_(tokyo.CblasRowMajor, tokyo.CblasLeft, tokyo.CblasLower,
                     tokyo.CblasNoTrans, tokyo.CblasNonUnit, size, size, 1.0,
                     <float*>A_.data, size, <float*>B_.data, size)
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
    A = np.array(np.random.random((size,size)), dtype=np.float32)

    print "smsetzero:  ",
    start = time.clock()
    for i in range(loops):
        tokyo.smsetzero(A)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s " % (rate/1000)


# set a vector of floats to all zeros

cdef svsetzero_speed(int size):
    cdef int i, loops
    loops = speed_base*5000/size
    x = np.array(np.random.random((size)), dtype=np.float32)

    print "svsetzero:  ",
    start = time.clock()
    for i in range(loops):
        tokyo.svsetzero(x)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s " % (rate/1000)


# single precision matrix += scalar * matrix

cdef smaxpy_speed(int size):
    cdef int i, loops
    loops = speed_base*10000/(size*size)
    X = np.array(np.random.random((size,size)), dtype=np.float32)
    Y = np.array(np.random.random((size,size)), dtype=np.float32)

    print "smaxpy:     ",
    start = time.clock()
    for i in range(loops):
        tokyo.smaxpy(1.2, X, Y)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s " % (rate/1000)


