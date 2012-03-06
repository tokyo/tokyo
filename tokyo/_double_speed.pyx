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
    dsymv_speed(size); print
    dtrmv_speed(size); print
    dtrsv_speed(size); print
    dger_speed(size);  print
    dsyr_speed(size);  print
    dsyr2_speed(size); print

print
print "SPEED TEST BLAS 3"
print

for size in test_sizes:

    print "Double precision: Vector size = " + str(size) + \
        "  Matrix size = " + str(size) + "x" + str(size)
    print
    dgemm_speed(size); print
    dsymm_speed(size); print


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


###############################################################################


#####################################
#
# BLAS LEVEL 1 (vector operations)
#
#####################################

# vector swap: x <-> y

cdef dswap_speed(int size):
    cdef int i, loops
    loops = speed_base*1000/size
    x = np.array(np.random.random((size)), dtype=np.float64)
    y = np.array(np.random.random((size)), dtype=np.float64)

    print "dswap:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.dswap(x, y)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s" % (rate/1000)


# scalar vector multiply: x *= alpha

cdef dscal_speed(int size):
    cdef int i, loops
    loops = speed_base*2500/size
    x = np.array(np.random.random((size)), dtype=np.float64)

    print "dscal:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.dscal(1.2, x)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s " % (rate/1000)


# vector copy: y <- x

cdef dcopy_speed(int size):
    cdef int i, loops
    loops = speed_base*1500/size
    x = np.array(np.random.random((size)), dtype=np.float64)
    y = np.array(np.random.random((size)), dtype=np.float64)

    print "dcopy:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.dcopy(x, y)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s " % (rate/1000)


# vector addition: y += alpha * x

cdef daxpy_speed(int size):
    cdef int i, loops
    loops = speed_base*1500/size
    x = np.array(np.random.random((size)), dtype=np.float64)
    y = np.array(np.random.random((size)), dtype=np.float64)

    print "daxpy:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.daxpy(1.2, x, y)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s " % (rate/1000)


# vector dot product: x.T y

cdef ddot_speed(int size):
    cdef int i, loops
    loops = speed_base*1500/size
    x = np.array(np.random.random((size)), dtype=np.float64)
    y = np.array(np.random.random((size)), dtype=np.float64)

    print "ddot:       ",
    start = time.clock()
    for i in range(loops):
        tokyo.ddot(x, y)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s " % (rate/1000)


# Euclidean norm:  ||x||_2

cdef dnrm2_speed(int size):
    cdef int i, loops
    loops = speed_base*700/size
    x = np.array(np.random.random((size)), dtype=np.float64)

    print "dnrm2:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.dnrm2(x)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s " % (rate/1000)


# sum of absolute values: ||x||_1

cdef dasum_speed(int size):
    cdef int i, loops
    loops = speed_base*2000/size
    x = np.array(np.random.random((size)), dtype=np.float64)

    print "dasum:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.dasum(x)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s " % (rate/1000)


# index of maximum absolute value element

cdef idamax_speed(int size):
    cdef int i, loops
    loops = speed_base*2000/size
    x = np.array(np.random.random((size)), dtype=np.float64)

    print "idamax:     ",
    start = time.clock()
    for i in range(loops):
        tokyo.idamax(x)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s " % (rate/1000)


###########################################
#
# BLAS LEVEL 2 (matrix-vector operations)
#
###########################################


# double precision matrix times vector: y = alpha * A   x + beta * y
#                                   or  y = alpha * A.T x + beta * y

cdef dgemv_speed(int size):

    cdef int i, loops

    loops = speed_base*10/(<int>(size**1.2))

    A = np.array(np.random.random((size,size)), dtype=np.float64)
    x = np.array(np.random.random((size)),      dtype=np.float64)
    y = np.array(np.random.random((size)),      dtype=np.float64)

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
        y = tokyo.dgemv(A, x)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    loops *= 5

    print "dgemv3:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.dgemv3(A, x, y)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "dgemv5:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.dgemv5(1.2, A, x, 2.1, y)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "dgemv6:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.dgemv6(tokyo.CblasNoTrans, 1.2, A, x, 2.1, y)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "dgemv_:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.dgemv_(tokyo.CblasRowMajor, tokyo.CblasNoTrans,
                     A_.shape[0], A_.shape[1],
                     1.2, <double*>A_.data, A_.shape[1], <double*>x_.data, 1,
                     2.1, <double*>y_.data, 1)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)


# Double precision symmetric-matrix-vector product: y = alpha * A * x + beta * y

cdef dsymv_speed(int size):

    cdef int i, loops

    loops = speed_base*10/(<int>(size**1.2))

    A = np.array(np.random.random((size,size)), dtype=np.float64)
    x = np.array(np.random.random((size)),      dtype=np.float64)
    y = np.array(np.random.random((size)),      dtype=np.float64)
    A = (A + A.T)/2

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

    print "dsymv:       ",
    start = time.clock()
    for i in range(loops):
        y = tokyo.dsymv(A, x)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    loops *= 5

    print "dsymv3:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.dsymv3(A, x, y)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "dsymv5:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.dsymv5(1.2, A, x, 2.1, y)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "dsymv6:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.dsymv6(tokyo.CblasRowMajor, tokyo.CblasLower, 1.2, A, x, 2.1, y)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "dsymv_:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.dsymv_(tokyo.CblasRowMajor, tokyo.CblasLower,
                     A_.shape[1], 1.2, <double*>A_.data, A_.shape[1],
                     <double*>x_.data, 1, 2.1, <double*>y_.data, 1)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)


# Double precision triangular matrix vector product: x <- A * x

cdef dtrmv_speed(int size):

    cdef int i, loops

    loops = speed_base*10/(<int>(size**1.2))

    A = np.array(np.random.random((size,size)), dtype=np.float64)
    x = np.array(np.random.random((size)),      dtype=np.float64)
    for i in range(size):
        for j in range(size):
            if j > i: A[i,j] = 0

    cdef np.ndarray[double, ndim=2, mode='c'] A_
    cdef np.ndarray[double, ndim=1, mode='c'] x_
    A_ = A; x_ = x

    print "numpy.dot:   ",
    start = time.clock()
    for i in range(loops):
        x = np.dot(A,x)
    np_rate = loops/(time.clock()-start)
    print "%9.0f kc/s" % (np_rate/1000)

    loops *= 3

    print "dtrmv:       ",
    start = time.clock()
    for i in range(loops):
        tokyo.dtrmv(A, x)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    loops *= 5

    print "dtrmv3:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.dtrmv3(tokyo.CblasNoTrans, A, x)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "dtrmv6:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.dtrmv6(tokyo.CblasRowMajor, tokyo.CblasLower, tokyo.CblasNoTrans,
                     tokyo.CblasNonUnit, A, x)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "dtrmv_:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.dtrmv_(tokyo.CblasRowMajor, tokyo.CblasLower, tokyo.CblasNoTrans,
                     tokyo.CblasNonUnit, A_.shape[1], <double*>A_.data,
                     A_.shape[1], <double*>x_.data, 1)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)


# Double precision triangular system solve: x <- inv(A) * x

cdef dtrsv_speed(int size):

    cdef int i, loops

    loops = speed_base*10/(<int>(size**1.2))

    A = np.array(np.random.random((size,size)), dtype=np.float64)
    x = np.array(np.random.random((size)),      dtype=np.float64)
    for i in range(size):
        for j in range(size):
            if j > i: A[i,j] = 0

    cdef np.ndarray[double, ndim=2, mode='c'] A_
    cdef np.ndarray[double, ndim=1, mode='c'] x_
    A_ = A; x_ = x

    loops *= 3

    print "dtrsv:       ",
    start = time.clock()
    for i in range(loops):
        tokyo.dtrsv(A, x)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s" % (rate/1000)

    loops *= 5

    print "dtrsv3:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.dtrsv3(tokyo.CblasNoTrans, A, x)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s" % (rate/1000)

    print "dtrsv6:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.dtrsv6(tokyo.CblasRowMajor, tokyo.CblasLower, tokyo.CblasNoTrans,
                     tokyo.CblasNonUnit, A, x)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s" % (rate/1000)

    print "dtrsv_:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.dtrsv_(tokyo.CblasRowMajor, tokyo.CblasLower, tokyo.CblasNoTrans,
                     tokyo.CblasNonUnit, A_.shape[1], <double*>A_.data,
                     A_.shape[1], <double*>x_.data, 1)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s" % (rate/1000)


# double precision vector outer-product: A = alpha * outer_product(x, y.T)

cdef dger_speed(int size):

    cdef int i, loops

    loops = speed_base*10/(<int>(size**1.2))

    x = np.array(np.random.random((size)),      dtype=np.float64)
    y = np.array(np.random.random((size)),      dtype=np.float64)
    Z = np.array(np.random.random((size,size)), dtype=np.float64)

    cdef np.ndarray[double, ndim=1, mode='c'] x_, y_
    cdef np.ndarray[double, ndim=2, mode='c'] Z_
    x_ = x; y_ = y; Z_ = Z

    print "numpy.outer: ",
    start = time.clock()
    for i in range(loops):
        np.outer(x, y)
    np_rate = loops/(time.clock()-start)
    print "%9.0f kc/s" % (np_rate/1000)

    loops *= 15

    print "dger:        ",
    start = time.clock()
    for i in range(loops):
        tokyo.dger(x, y)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    loops *= 2

    print "dger3:       ",
    start = time.clock()
    for i in range(loops):
        tokyo.dger3(x, y, Z)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "dger4:       ",
    start = time.clock()
    for i in range(loops):
        tokyo.dger4(1.0, x, y, Z)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "dger_:       ",
    start = time.clock()
    for i in range(loops):
        tokyo.dger_(tokyo.CblasRowMajor, x_.shape[0], y_.shape[0],
                    1.0, <double*>x_.data, 1, <double*>y_.data, 1,
                    <double*>Z_.data, Z_.shape[1])
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)


# double precision symmetric rank 1 update: A <- alpha * x * x.T + A

cdef dsyr_speed(int size):

    cdef int i, loops

    loops = speed_base*10/(<int>(size**1.2))

    x = np.array(np.random.random((size)), dtype=np.float64)
    A = np.array(np.random.random((size,size)), dtype=np.float64)

    cdef np.ndarray[double, ndim=1, mode='c'] x_
    cdef np.ndarray[double, ndim=2, mode='c'] A_
    x_ = x; A_ = A

    print "numpy.outer: ",
    start = time.clock()
    for i in range(loops):
        np.outer(x, x)
    np_rate = loops/(time.clock()-start)
    print "%9.0f kc/s" % (np_rate/1000)

    loops *= 15

    print "dsyr:        ",
    start = time.clock()
    for i in range(loops):
        tokyo.dsyr(x)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    loops *= 2

    print "dsyr_2:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.dsyr_2(x, A)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "dsyr_3:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.dsyr_3(1.0, x, A)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "dsyr_:       ",
    start = time.clock()
    for i in range(loops):
        tokyo.dsyr_(tokyo.CblasRowMajor, tokyo.CblasLower,
                    x.shape[0], 1.0, <double*>x_.data, 1,
                    <double*>A_.data, A_.shape[1])
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)


# double precision symmetric rank 2 update:
# A <- alpha * x * y.T + alpha * y * x.T + A

cdef dsyr2_speed(int size):

    cdef int i, loops

    loops = speed_base*10/(<int>(size**1.2))

    x = np.array(np.random.random((size)), dtype=np.float64)
    y = np.array(np.random.random((size)), dtype=np.float64)
    A = np.array(np.random.random((size,size)), dtype=np.float64)

    cdef np.ndarray[double, ndim=1, mode='c'] x_, y_
    cdef np.ndarray[double, ndim=2, mode='c'] A_
    x_ = x; y_ = y; A_ = A

    print "numpy.outer: ",
    start = time.clock()
    for i in range(loops):
        np.outer(x, y) + np.outer(y, x)
    np_rate = loops/(time.clock()-start)
    print "%9.0f kc/s" % (np_rate/1000)

    loops *= 15

    print "dsyr2:       ",
    start = time.clock()
    for i in range(loops):
        tokyo.dsyr2(x, y)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    loops *= 2

    print "dsyr2_3:     ",
    start = time.clock()
    for i in range(loops):
        tokyo.dsyr2_3(x, y, A)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "dsyr2_4:     ",
    start = time.clock()
    for i in range(loops):
        tokyo.dsyr2_4(1.0, x, y, A)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "dsyr2_:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.dsyr2_(tokyo.CblasRowMajor, tokyo.CblasLower,
                    x.shape[0], 1.0, <double*>x_.data, 1,
                    <double*>y_.data, 1,
                    <double*>A_.data, A_.shape[1])
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

cdef dgemm_speed(int size):

    cdef int i, loops

    loops = speed_base*150/(size*size)

    X = np.array(np.random.random((size,size)), dtype=np.float64)
    Y = np.array(np.random.random((size,size)), dtype=np.float64)
    Z = np.array(np.random.random((size,size)), dtype=np.float64)

    cdef np.ndarray[double, ndim=2, mode='c'] X_, Y_, Z_
    X_ = X; Y_ = Y; Z_ = Z

    print "numpy.dot: ",
    start = time.clock()
    for i in range(loops): np.dot(X, Y)
    np_rate = loops/(time.clock()-start)
    print "%9.0f kc/s" % (np_rate/1000)

    print "dgemm:     ",
    start = time.clock()
    for i in range(loops):
        tokyo.dgemm(X, Y)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "dgemm3:    ",
    start = time.clock()
    for i in range(loops):
        tokyo.dgemm3(X, Y, Z)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "dgemm5:    ",
    start = time.clock()
    for i in range(loops):
        tokyo.dgemm5(1.0, X, Y, 0.0, Z)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "dgemm7:    ",
    start = time.clock()
    for i in range(loops):
        tokyo.dgemm7(tokyo.CblasNoTrans, tokyo.CblasNoTrans, 1.0, X, Y, 0.0, Z)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "dgemm_:    ",
    start = time.clock()
    for i in range(loops):
        tokyo.dgemm_(tokyo.CblasRowMajor, tokyo.CblasNoTrans,
                     tokyo.CblasNoTrans, size, size, size, 1.0,
                     <double*>X_.data, size, <double*>Y_.data, size,
                     0.0, <double*>Z_.data, size)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)


# Matrix times matrix: C = alpha * A * B + beta * C
#                  or: C = alpha * B * A + beta * C
# where A = A.T.
#
# double precision

cdef dsymm_speed(int size):

    cdef int i, loops

    loops = speed_base*150/(size*size)

    A = np.array(np.random.random((size,size)), dtype=np.float64)
    B = np.array(np.random.random((size,size)), dtype=np.float64)
    C = np.array(np.random.random((size,size)), dtype=np.float64)
    A = (A + A.T)/2

    cdef np.ndarray[double, ndim=2, mode='c'] A_, B_, C_
    A_ = A; B_ = B; C_ = C

    print "numpy.dot: ",
    start = time.clock()
    for i in range(loops): np.dot(A, B)
    np_rate = loops/(time.clock()-start)
    print "%9.0f kc/s" % (np_rate/1000)

    print "dsymm:     ",
    start = time.clock()
    for i in range(loops):
        tokyo.dsymm(A, B)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "dsymm3:    ",
    start = time.clock()
    for i in range(loops):
        tokyo.dsymm3(A, B, C)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "dsymm5:    ",
    start = time.clock()
    for i in range(loops):
        tokyo.dsymm5(1.0, A, B, 0.0, C)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "dsymm8:    ",
    start = time.clock()
    for i in range(loops):
        tokyo.dsymm8(tokyo.CblasRowMajor, tokyo.CblasLeft, tokyo.CblasLower,
                     1.0, A, B, 0.0, C)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)

    print "dsymm_:    ",
    start = time.clock()
    for i in range(loops):
        tokyo.dsymm_(tokyo.CblasRowMajor, tokyo.CblasLeft,
                     tokyo.CblasLower, size, size, 1.0,
                     <double*>A_.data, size, <double*>B_.data, size,
                     0.0, <double*>C_.data, size)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s %5.1fx" % (rate/1000,rate/np_rate)


####################################################################
#
# Utility function I have put together that aren't in BLAS or LAPACK
#
####################################################################


# set a matrix of double to all zeros

cdef dmsetzero_speed(int size):
    cdef int i, loops
    loops = speed_base*5000/(size*size)
    A = np.array(np.random.random((size,size)), dtype=np.float64)

    print "dmsetzero:  ",
    start = time.clock()
    for i in range(loops):
        tokyo.dmsetzero(A)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s " % (rate/1000)


# set a vector of doubles to all zeros

cdef dvsetzero_speed(int size):
    cdef int i, loops
    loops = speed_base*5000/size
    x = np.array(np.random.random((size)), dtype=np.float64)

    print "dvsetzero:  ",
    start = time.clock()
    for i in range(loops):
        tokyo.dvsetzero(x)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s " % (rate/1000)


# double precision matrix += scalar * matrix

cdef dmaxpy_speed(int size):
    cdef int i, loops
    loops = speed_base*10000/(size*size)
    X = np.array(np.random.random((size,size)), dtype=np.float64)
    Y = np.array(np.random.random((size,size)), dtype=np.float64)

    print "dmaxpy:     ",
    start = time.clock()
    for i in range(loops):
        tokyo.dmaxpy(1.2, X, Y)
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s " % (rate/1000)

