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
print "Tokyo BLAS wrapper verification against scipy/numpy"
print "---------------------------------------------------"
print

print
print "VERIFY CORRECTNESS BLAS Level 1"
print
sswap_verify()
sscal_verify()
scopy_verify()
saxpy_verify()
sdot_verify()
snrm2_verify()
sasum_verify()
isamax_verify()
srotg_verify()
#srot_verify()
print

dswap_verify()
dscal_verify()
dcopy_verify()
daxpy_verify()
ddot_verify()
dnrm2_verify()
dasum_verify()
idamax_verify()
drotg_verify()


print
print "VERIFY CORRECTNESS BLAS Level 2"
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

###############################################################################

# A function for checking that different matrices from different
# computations are in some sense "equal" in the verification tests.
cpdef approx_eq(X, Y): return abs(np.sum(X - Y)) < 1e-5


#####################################
# BLAS LEVEL 1 (vector operations)
#####################################

# vector swap: x <-> y

cdef sswap_verify():
    x = np.array(np.random.random((4)), dtype=np.float32)
    y = np.array(np.random.random((4)), dtype=np.float32)
    temp1 = x.copy()
    temp2 = y.copy()
    tokyo.sswap(x,y)
    print "sswap:  ", (approx_eq(temp1, y) and approx_eq(temp2, x))

cdef dswap_verify():
    x = np.array(np.random.random((4)), dtype=np.float64)
    y = np.array(np.random.random((4)), dtype=np.float64)
    temp1 = x.copy()
    temp2 = y.copy()
    tokyo.dswap(x,y)
    print "dswap:  ", (approx_eq(temp1, y) and approx_eq(temp2, x))

# scalar vector multiply: x *= alpha

cdef sscal_verify():
    x = np.array(np.random.random((4)), dtype=np.float32)
    temp = 1.2 * x
    tokyo.sscal(1.2, x)
    print "sscal:  ", approx_eq(temp, x)

cdef dscal_verify():
    x = np.array(np.random.random((4)), dtype=np.float64)
    temp = 1.2 * x
    tokyo.dscal(1.2, x)
    print "dscal:  ", approx_eq(temp, x)

# vector copy: y <- x

cdef scopy_verify():
    x = np.array(np.random.random((4)), dtype=np.float32)
    y = np.array(np.random.random((4)), dtype=np.float32)
    tokyo.scopy(x,y)
    print "scopy:  ", approx_eq(x, y)

cdef dcopy_verify():
    x = np.array(np.random.random((4)), dtype=np.float64)
    y = np.array(np.random.random((4)), dtype=np.float64)
    tokyo.dcopy(x,y)
    print "dcopy:  ", approx_eq(x, y)

# vector addition: y += alpha * x

cdef saxpy_verify():
    x = np.array(np.random.random((5)),   dtype=np.float32)
    y = np.array(np.random.random((5)),   dtype=np.float32)
    temp = 1.2 * x + y
    tokyo.saxpy(1.2, x, y)
    print "saxpy:  ", approx_eq(temp, y)

cdef daxpy_verify():
    x = np.array(np.random.random((5)),   dtype=np.float64)
    y = np.array(np.random.random((5)),   dtype=np.float64)
    temp = 1.2 * x + y
    tokyo.daxpy(1.2, x, y)
    print "daxpy:  ", approx_eq(temp, y)

# vector dot product: x.T y

cdef sdot_verify():
    x = np.array(np.random.random((5)),   dtype=np.float32)
    y = np.array(np.random.random((5)),   dtype=np.float32)
    print "sdot:   ", approx_eq(np.dot(x,y), tokyo.sdot(x,y))

cdef ddot_verify():
    x = np.array(np.random.random((5)),   dtype=np.float64)
    y = np.array(np.random.random((5)),   dtype=np.float64)
    print "ddot:   ", approx_eq(np.dot(x,y), tokyo.ddot(x,y))

# Euclidean norm:  ||x||_2

cdef snrm2_verify():
    x = np.array(np.random.random((5)),   dtype=np.float32)
    print "snrm2:  ", approx_eq(np.sqrt(np.sum(np.dot(x,x))), tokyo.snrm2(x))

cdef dnrm2_verify():
    x = np.array(np.random.random((5)),   dtype=np.float64)
    print "dnrm2:  ", approx_eq(np.sqrt(np.sum(np.dot(x,x))), tokyo.dnrm2(x))

# sum of absolute values: ||x||_1

cdef sasum_verify():
    x = np.array(np.random.random((5)),   dtype=np.float32)
    print "sasum:  ", approx_eq(np.sum(np.abs(x)), tokyo.sasum(x))

cdef dasum_verify():
    x = np.array(np.random.random((5)),   dtype=np.float64)
    print "dasum:  ", approx_eq(np.sum(np.abs(x)), tokyo.dasum(x))

# index of maximum absolute value element

cdef isamax_verify():
    x = np.array([0.06, -0.1, -0.05, -0.001, 0.07],   dtype=np.float32)
    print "isamax: ", ( 1 == tokyo.isamax(x))

cdef idamax_verify():
    x = np.array([0.06, -0.1, -0.05, -0.001, 0.07],   dtype=np.float64)
    print "idamax: ", ( 1 == tokyo.idamax(x))

cpdef check_negligible(scomp, strue, ssize, sfac):
    """
    Return `True` if difference between `scomp` and `strue` is
    negligible elementwise.
    """
    sd = scomp - strue
    abssize = np.abs(ssize)
    return not np.any((abssize + np.abs(sfac*sd)) - abssize)

cdef srotg_verify():
    da1 = np.array([0.3, 0.4, -0.3, -0.4, -0.3, 0.0, 0.0, 1.0], np.float32)
    db1 = np.array([0.4, 0.3, 0.4, 0.3, -0.4, 0.0, 1.0, 0.0], np.float32)
    dc1 = np.array([0.6, 0.8, -0.6, 0.8, 0.6, 1.0, 0.0, 1.0], np.float32)
    ds1 = np.array([0.8, 0.6, 0.8, -0.6, 0.8, 0.0, 1.0, 0.0], np.float32)
    datrue = np.array([0.5, 0.5, 0.5, -0.5, -0.5, 0.0, 1.0, 1.0], np.float32)
    dbtrue = np.array([1.0/0.6, 0.6, -1.0/0.6, -0.6, 1.0/0.6, 0.0, 1.0, 0.0],
             np.float32)
    a = np.empty(8, np.float32)
    b = np.empty(8, np.float32)
    c = np.empty(8, np.float32)
    s = np.empty(8, np.float32)
    for k in range(len(da1)):
        (a[k],b[k],c[k],s[k]) = tokyo.srotg(da1[k], db1[k])

    a_ok = check_negligible(a, datrue, datrue, 9.765625e-4)
    b_ok = check_negligible(b, dbtrue, dbtrue, 9.765625e-4)
    c_ok = check_negligible(c, dc1, dc1, 9.765625e-4)
    s_ok = check_negligible(s, ds1, ds1, 9.765625e-4)
    if not (a_ok and b_ok and c_ok and s_ok):
        print 'srotg: '
        print 'Got ', (a,b,c,s)
        print 'Expected ', (datrue, dbtrue, dc1, ds1)
    else:
        print 'srotg: All ok'

cdef drotg_verify():
    da1 = np.array([0.3, 0.4, -0.3, -0.4, -0.3, 0.0, 0.0, 1.0], np.float64)
    db1 = np.array([0.4, 0.3, 0.4, 0.3, -0.4, 0.0, 1.0, 0.0], np.float64)
    dc1 = np.array([0.6, 0.8, -0.6, 0.8, 0.6, 1.0, 0.0, 1.0], np.float64)
    ds1 = np.array([0.8, 0.6, 0.8, -0.6, 0.8, 0.0, 1.0, 0.0], np.float64)
    datrue = np.array([0.5, 0.5, 0.5, -0.5, -0.5, 0.0, 1.0, 1.0], np.float64)
    dbtrue = np.array([1.0/0.6, 0.6, -1.0/0.6, -0.6, 1.0/0.6, 0.0, 1.0, 0.0],
             np.float64)
    a = np.empty(8, np.float64)
    b = np.empty(8, np.float64)
    c = np.empty(8, np.float64)
    s = np.empty(8, np.float64)
    for k in range(len(da1)):
        (a[k],b[k],c[k],s[k]) = tokyo.drotg(da1[k], db1[k])

    a_ok = check_negligible(a, datrue, datrue, 9.765625e-4)
    b_ok = check_negligible(b, dbtrue, dbtrue, 9.765625e-4)
    c_ok = check_negligible(c, dc1, dc1, 9.765625e-4)
    s_ok = check_negligible(s, ds1, ds1, 9.765625e-4)
    if not (a_ok and b_ok and c_ok and s_ok):
        print 'drotg: '
        print 'Got ', (a,b,c,s)
        print 'Expected ', (datrue, dbtrue, dc1, ds1)
    else:
        print 'drotg: All ok'


#cdef srot_verify():
#    incxs = np.array([1, 2, -2, -1], np.int)
#    incys = np.array([1, -2, 1, -2], np.int)
#    lens  = np.array([[1, 1], [2, 4], [1, 1], [3, 7]], np.int)
#    ns    = np.array([0, 1, 2, 4], np.int)
#
#    dx1 = np.array([0.6,  0.1, -0.5, 0.8,  0.9, -0.3, -0.4], np.float32)
#    dy1 = np.array([0.5, -0.9,  0.3, 0.7, -0.6,  0.2,  0.8], np.float32)
#    sc  = np.array([0.8], np.float32)
#    ss  = np.array([0.6], np.float32)
#
#    sx    = dx1.copy() ; stx = np.zeros_like(dx1)
#    sy    = dy1.copy() ; sty = np.zeros_like(dy1)
#
#    # Data to be recovered as 7x4x4 arrays.
#    dt9x = np.array([[[ 0.6,  0.0,   0.0,   0.0 ], [ 0.0,  0.0,   0.0,   0.78],
#                      [ 0.0,  0.0,   0.0,   0.0 ], [ 0.0,  0.0,   0.78, -0.46]],
#                     [[ 0.0,  0.0,   0.0,   0.0 ], [ 0.0,  0.78, -0.46, -0.22],
#                      [ 1.06, 0.0,   0.0,   0.0 ], [ 0.6,  0.0,   0.0,   0.0 ]],
#                     [[ 0.0,  0.0,   0.0,   0.78], [ 1.0,  0.0,   0.0,   0.0 ],
#                      [ 0.0,  0.0,   0.66,  0.1 ], [-0.1,  0.0,   0.0,   0.0 ]],
#                     [[ 0.0,  0.96,  0.1,  -0.76], [ 0.8,  0.90, -0.3,  -0.02],
#                      [ 0.6,  0.0,   0.0,   0.0 ], [ 0.0,  0.0,   0.0,   0.78]],
#                     [[ 0.0,  0.0,   0.0,   0.0 ], [ 0.0,  0.0,  -0.06,  0.1 ],
#                      [-0.1,  0.0,   0.0,   0.0 ], [ 0.0,  0.90,  0.1,  -0.22]],
#                     [[ 0.8,  0.18, -0.3,  -0.02], [ 0.6,  0.0,   0.0,   0.0 ],
#                      [ 0.0,  0.0,   0.0,   0.78], [ 0.0,  0.0,   0.0,   0.0 ]],
#                     [[ 0.0,  0.0,   0.78,  0.26], [ 0.0,  0.0,   0.0,   0.0 ],
#                      [ 0.0,  0.78,  0.26, -0.76], [ 1.12, 0.0,   0.0,   0.0]]],
#                     np.float32)
#
#    dt9y = np.array([[[0.5,  0.0,   0.0,   0.0 ], [ 0.0,  0.0,   0.0,   0.04],
#                      [0.0,  0.0,   0.0,   0.0 ], [ 0.0,  0.0,   0.04, -0.78]],
#                     [[0.0,  0.0,   0.0,   0.0 ], [ 0.0,  0.04, -0.78,  0.54],
#                      [0.08, 0.0,   0.0,   0.0 ], [ 0.5,  0.0,   0.0,   0.0 ]],
#                     [[0.0,  0.0,   0.0,   0.04], [ 0.0,  0.0,   0.0,   0.0 ],
#                      [0.0,  0.0,   0.7,  -0.9 ], [-0.12, 0.0,   0.0,   0.0 ]],
#                     [[0.0,  0.64, -0.9,  -0.30], [ 0.7, -0.18,  0.2,   0.28],
#                      [0.5,  0.0,   0.0,   0.0 ], [ 0.0,  0.0,   0.0,   0.04]],
#                     [[0.0,  0.0,   0.0,   0.0 ], [ 0.0,  0.0,   0.7,  -1.08],
#                      [0.0,  0.0,   0.0,   0.0 ], [ 0.0,  0.64, -1.26,  0.54]],
#                     [[0.20, 0.0,   0.0,   0.0 ], [ 0.5,  0.0,   0.0,   0.0 ],
#                      [0.0,  0.0,   0.0,   0.04], [ 0.0,  0.0,   0.0,   0.0 ]],
#                     [[0.0,  0.0,   0.04, -0.9 ], [ 0.18, 0.0,   0.0,   0.0 ],
#                      [0.0,  0.04, -0.9,   0.18], [ 0.7, -0.18,  0.2,   0.16]]],
#                     np.float32)
#
#    ssize2 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
#                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
#                       0.0, 1.17, 1.17, 1.17, 1.17, 1.17,
#                       1.17, 1.17, 1.17, 1.17, 1.17, 1.17,
#                       1.17, 1.17, 1.17], np.float32)
#
#    for j in range(len(incxs)):
#        incx = incxs[j]
#        incy = incys[j]
#        mx = abs(incx)
#        my = abs(incy)
#
#        for k in range(len(ns)):
#            n = ns[k] ; ksize = min(1,k-1)
#            lenx = lens[k,mx-1]
#            leny = lens[k,my-1]
#
#            for i in range(len(dx1)):
#                stx[i] = dt9x[i,k,j]
#                sty[i] = dt9y[i,k,j]
#
#            tokyo.srot(sx, sy, sc[0], ss[0], dx=incx, dy=incy)
#            x_ok = check_negligible(sx, stx, ssize2[ksize], 9.765625e-4)
#            y_ok = check_negligible(sy, sty, ssize2[ksize], 9.765625e-4)
#
#    if not (x_ok and y_ok):
#        print 'srot: '
#        print 'Got:' ; print 'sx = ', sx ; print 'sy = ', sy
#        print 'Expected:' ; print 'sx = ', stx ; print 'sy = ', sty
#    else:
#        print 'srot: All ok'

###########################################
# BLAS LEVEL 2 (matrix-vector operations)
###########################################


# single precision matrix times vector: y = alpha * A   x + beta * y
#                                   or  y = alpha * A.T x + beta * y

cdef sgemv_verify():

    A = np.array(np.random.random((4,5)), dtype=np.float32)
    x = np.array(np.random.random((5)),   dtype=np.float32)
    y = np.array(np.random.random((4)),   dtype=np.float32)

    cdef np.ndarray[float, ndim=2, mode='c'] A_
    cdef np.ndarray[float, ndim=1, mode='c'] x_, y_
    A_ = A; x_ = x; y_ = y

    temp = np.dot(A,x)
    temp2 = tokyo.sgemv(A, x)
    print "sgemv:  ", approx_eq(temp, temp2)

    temp = np.dot(A,x)
    tokyo.sgemv3(A, x, y)
    print "sgemv3: ", approx_eq(temp, y)

    temp = 1.2*np.dot(A,x) + 2.1*y
    tokyo.sgemv5(1.2, A, x, 2.1, y)
    print "sgemv5: ", approx_eq(temp, y)

    temp = 1.2*np.dot(A,x) + 2.1*y
    tokyo.sgemv6(tokyo.CblasNoTrans, 1.2, A, x, 2.1, y)
    print "sgemv6: ", approx_eq(temp, y)

    temp = 1.2*np.dot(A,x) + 2.1*y
    tokyo.sgemv_(tokyo.CblasRowMajor, tokyo.CblasNoTrans, 4, 5,
                       1.2, <float*>A_.data, 5, <float*>x_.data, 1,
                       2.1, <float*>y_.data, 1)
    print "sgemv_: ", approx_eq(temp, y)


# double precision matrix times vector: y = alpha * A   x + beta * y
#                                   or  y = alpha * A.T x + beta * y


cdef dgemv_verify():

    A = np.array(np.random.random((4,5)), dtype=np.float64)
    x = np.array(np.random.random((5)),   dtype=np.float64)
    y = np.array(np.random.random((4)),   dtype=np.float64)

    cdef np.ndarray[double, ndim=2, mode='c'] A_
    cdef np.ndarray[double, ndim=1, mode='c'] x_, y_
    A_ = A; x_ = x; y_ = y

    temp = np.dot(A,x)
    temp2 = tokyo.dgemv(A, x)
    print "dgemv:  ", approx_eq(temp, temp2)

    temp = np.dot(A,x)
    tokyo.dgemv3(A, x, y)
    print "dgemv3: ", approx_eq(temp, y)

    temp = 1.2*np.dot(A,x) + 2.1*y
    tokyo.dgemv5(1.2, A, x, 2.1, y)
    print "dgemv5: ", approx_eq(temp, y)

    temp = 1.2*np.dot(A,x) + 2.1*y
    tokyo.dgemv6(tokyo.CblasNoTrans, 1.2, A, x, 2.1, y)
    print "dgemv6: ", approx_eq(temp, y)

    temp = 1.2*np.dot(A,x) + 2.1*y
    tokyo.dgemv_(tokyo.CblasRowMajor, tokyo.CblasNoTrans, 4, 5,
                       1.2, <double*>A_.data, 5, <double*>x_.data, 1,
                       2.1, <double*>y_.data, 1)
    print "dgemv_: ", approx_eq(temp, y)


# single precision vector outer-product: A = alpha * outer_product(x, y.T)

cdef sger_verify():

    x = np.array(np.random.random((4)),   dtype=np.float32)
    y = np.array(np.random.random((5)),   dtype=np.float32)
    A = np.array(np.random.random((4,5)), dtype=np.float32)

    result = np.outer(x, y)
    print "sger:   ", approx_eq(result, tokyo.sger(x, y))

    result = A + np.outer(x, y)
    tokyo.sger3(x, y, A)
    print "sger3:  ", approx_eq(result, A)

    result = A + 1.2*np.outer(x, y)
    tokyo.sger4(1.2, x, y, A)
    print "sger4:  ", approx_eq(result, A)


# double precision vector outer-product: A = alpha * outer_product(x, y.T)

cdef dger_verify():

    x = np.array(np.random.random((4)),   dtype=np.float64)
    y = np.array(np.random.random((5)),   dtype=np.float64)
    A = np.array(np.random.random((4,5)), dtype=np.float64)

    result = np.outer(x, y)
    print "dger:   ", approx_eq(result, tokyo.dger(x, y))

    result = A + np.outer(x, y)
    tokyo.dger3(x, y, A)
    print "dger3:  ", approx_eq(result, A)

    result = A + 1.2*np.outer(x, y)
    tokyo.dger4(1.2, x, y, A)
    print "dger4:  ", approx_eq(result, A)



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

cdef sgemm_verify():

    X = np.array(np.random.random((3,4)), dtype=np.float32)
    Y = np.array(np.random.random((4,5)), dtype=np.float32)

    print "sgemm:  ", approx_eq(np.dot(X, Y), tokyo.sgemm(X, Y))

    Z = np.array(np.random.random((3,5)), dtype=np.float32)
    tokyo.sgemm3(X, Y, Z)
    print "sgemm3: ", approx_eq(np.dot(X, Y), Z)

    Z = np.array(np.random.random((3,5)), dtype=np.float32)
    result = 2.3*np.dot(X, Y) + 1.2*Z
    tokyo.sgemm5(2.3, X, Y, 1.2, Z)
    print "sgemm5: ", approx_eq(result, Z)

    Z = np.array(np.random.random((3,5)), dtype=np.float32)
    result = 2.3*np.dot(X, Y) + 1.2*Z
    tokyo.sgemm7(tokyo.CblasNoTrans, tokyo.CblasNoTrans, 2.3, X, Y, 1.2, Z)
    print "sgemm7: ", approx_eq(result, Z)


# matrix times matrix: C = alpha * A   B   + beta * C
#                  or  C = alpha * A.T B   + beta * C
#                  or  C = alpha * A   B.T + beta * C
#                  or  C = alpha * A.T B.T + beta * C
#
# double precision

cdef dgemm_verify():

    X = np.array(np.random.random((3,4)), dtype=np.float64)
    Y = np.array(np.random.random((4,5)), dtype=np.float64)

    print "dgemm:  ", approx_eq(np.dot(X, Y), tokyo.dgemm(X, Y))

    Z = np.array(np.random.random((3,5)), dtype=np.float64)
    tokyo.dgemm3(X, Y, Z)
    print "dgemm3: ", approx_eq(np.dot(X, Y), Z)

    Z = np.array(np.random.random((3,5)), dtype=np.float64)
    result = 2.3*np.dot(X, Y) + 1.2*Z
    tokyo.dgemm5(2.3, X, Y, 1.2, Z)
    print "dgemm5: ", approx_eq(result, Z)

    Z = np.array(np.random.random((3,5)), dtype=np.float64)
    result = 2.3*np.dot(X, Y) + 1.2*Z
    tokyo.dgemm7(tokyo.CblasNoTrans, tokyo.CblasNoTrans, 2.3, X, Y, 1.2, Z)
    print "dgemm7: ", approx_eq(result, Z)



####################################################################
#
# Utility function I have put together that aren't in BLAS or LAPACK
#
####################################################################

# set a matrix of floats to all zeros

cdef smsetzero_verify():
    A = np.array(np.random.random((4,4)), dtype=np.float32)
    B = np.zeros((4,4))
    tokyo.smsetzero(A)
    print "smsetzero:  ", approx_eq(A, B)

# set a matrix of floats to all zeros

cdef dmsetzero_verify():
    A = np.array(np.random.random((4,4)), dtype=np.float64)
    B = np.zeros((4,4))
    tokyo.dmsetzero(A)
    print "dmsetzero:  ", approx_eq(A, B)

# set a vector of doubles to all zeros

cdef svsetzero_verify():
    x = np.array(np.random.random((4)), dtype=np.float32)
    y = np.zeros((4))
    tokyo.svsetzero(x)
    print "svsetzero:  ", approx_eq(x, y)

# set a vector of doubles to all zeros

cdef dvsetzero_verify():
    x = np.array(np.random.random((4)), dtype=np.float64)
    y = np.zeros((4))
    tokyo.dvsetzero(x)
    print "dvsetzero:  ", approx_eq(x, y)

# double precision matrix += scalar * matrix

cdef smaxpy_verify():
    X = np.array(np.random.random((4,4)), dtype=np.float32)
    Y = np.array(np.random.random((4,4)), dtype=np.float32)
    temp = 1.2 * X + Y
    tokyo.smaxpy(1.2, X, Y)
    print "smaxpy:     ", approx_eq(temp, Y)

# double precision matrix += scalar * matrix

cdef dmaxpy_verify():
    X = np.array(np.random.random((4,4)), dtype=np.float64)
    Y = np.array(np.random.random((4,4)), dtype=np.float64)
    temp = 1.2 * X + Y
    tokyo.dmaxpy(1.2, X, Y)
    print "dmaxpy:     ", approx_eq(temp, Y)

