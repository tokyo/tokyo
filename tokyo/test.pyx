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
print "Tokyo BLAS wrapper"
print "------------------"
print
print "Make sure your CPU isn't doing frequency scaling, otherwise"
print "the speed results here might be all messed up.   A few percent"
print "variation in speed results from run to run is normal."
print
print "Correctness is verified against scipy/numpy equivalent calls."
print
print "Speed is given in thousands of calls per second (kc/s), and in"
print "some cases how many times faster than scipy/numpy the call is."
print "Naturally the advantage is greatest on small vectors/matrices"
print "because that's when the numpy/scipy overhead is high relative"
print "to the total computation cost."

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
    snrm2_speed(size)
    sasum_speed(size)
    isamax_speed(size)
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
print "VERIFY CORRECTNESS BLAS 2"
print

sgemv_verify(); print
sger_verify(); print

dgemv_verify(); print
dger_verify(); print


print
print "SPEED TEST BLAS 2"
print

for size in test_sizes:
    
    print "Single Precision: Vector size = " + str(size) + \
        "  Matrix size = " + str(size) + "x" + str(size)
    print
    sgemv_speed(size); print
    sger_speed(size); print


for size in test_sizes:
    
    print "Double precision: Vector size = " + str(size) + \
        "  Matrix size = " + str(size) + "x" + str(size)
    print
    dgemv_speed(size); print
    dger_speed(size); print

print
print "VERIFY CORRECTNESS BLAS 3"
print

sgemm_verify(); print

dgemm_verify(); print

print
print "SPEED TEST BLAS 3"
print

for size in test_sizes:
    
    print "Single precision: Vector size = " + str(size) + \
        "  Matrix size = " + str(size) + "x" + str(size)
    print
    sgemm_speed(size); print

for size in test_sizes:
    
    print "Double precision: Vector size = " + str(size) + \
        "  Matrix size = " + str(size) + "x" + str(size)
    print
    dgemm_speed(size); print

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

for size in test_sizes:  
    
    print "Double precision: Vector size = " + str(size) + \
        "  Matrix size = " + str(size) + "x" + str(size) 
    print
    dmsetzero_speed(size)
    dvsetzero_speed(size)
    dmaxpy_speed(size)
    print


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
    
def sswap_speed(int size):
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

def dswap_verify():
    x = np.array( np.random.random( (4) ), dtype=np.float64 )
    y = np.array( np.random.random( (4) ), dtype=np.float64 )
    temp1 = x.copy()
    temp2 = y.copy()
    tokyo.dswap(x,y)
    print "dswap:  ", (approx_eq( temp1, y ) and approx_eq( temp2, x ))
    
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

def sscal_verify():
    x = np.array( np.random.random( (4) ), dtype=np.float32 )
    temp = 1.2 * x
    tokyo.sscal( 1.2, x)
    print "sscal:  ", approx_eq( temp, x )

def sscal_speed(int size):
    cdef int i, loops
    loops = speed_base*2500/size
    x = np.array( np.random.random( (size) ), dtype=np.float32 )

    print "sscal:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.sscal( 1.2, x )
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s " % (rate/1000)

def dscal_verify():
    x = np.array( np.random.random( (4) ), dtype=np.float64 )
    temp = 1.2 * x
    tokyo.dscal( 1.2, x)
    print "dscal:  ", approx_eq( temp, x )

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

def scopy_verify():
    x = np.array( np.random.random( (4) ), dtype=np.float32 )
    y = np.array( np.random.random( (4) ), dtype=np.float32 )
    tokyo.scopy(x,y)
    print "scopy:  ", approx_eq( x, y )

def scopy_speed(int size):
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

def dcopy_verify():
    x = np.array( np.random.random( (4) ), dtype=np.float64 )
    y = np.array( np.random.random( (4) ), dtype=np.float64 )
    tokyo.dcopy(x,y)
    print "dcopy:  ", approx_eq( x, y )

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

def saxpy_verify():
    x = np.array( np.random.random( (5) ),   dtype=np.float32 )
    y = np.array( np.random.random( (5) ),   dtype=np.float32 )
    temp = 1.2 * x + y
    tokyo.saxpy( 1.2, x, y )
    print "saxpy:  ", approx_eq( temp, y )

def saxpy_speed( int size ):
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

def daxpy_verify():
    x = np.array( np.random.random( (5) ),   dtype=np.float64 )
    y = np.array( np.random.random( (5) ),   dtype=np.float64 )
    temp = 1.2 * x + y
    tokyo.daxpy( 1.2, x, y )
    print "daxpy:  ", approx_eq( temp, y )

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

def sdot_verify():
    x = np.array( np.random.random( (5) ),   dtype=np.float32 )
    y = np.array( np.random.random( (5) ),   dtype=np.float32 )
    print "sdot:   ", approx_eq( np.dot(x,y), tokyo.sdot(x,y) )

def sdot_speed(int size):
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

def ddot_verify():
    x = np.array( np.random.random( (5) ),   dtype=np.float64 )
    y = np.array( np.random.random( (5) ),   dtype=np.float64 )
    print "ddot:   ", approx_eq( np.dot(x,y), tokyo.ddot(x,y) )

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

def snrm2_verify():
    x = np.array( np.random.random( (5) ),   dtype=np.float32 )
    print "snrm2:  ", approx_eq( np.sqrt(np.sum(np.dot(x,x))), tokyo.snrm2(x) )

def snrm2_speed(int size):
    cdef int i, loops
    loops = speed_base*700/size
    x = np.array( np.random.random( (size) ), dtype=np.float32 )

    print "snrm2:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.snrm2( x )
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s " % (rate/1000)


def dnrm2_verify():
    x = np.array( np.random.random( (5) ),   dtype=np.float64 )
    print "dnrm2:  ", approx_eq( np.sqrt(np.sum(np.dot(x,x))), tokyo.dnrm2(x) )

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

def sasum_verify():
    x = np.array( np.random.random( (5) ),   dtype=np.float32 )
    print "sasum:  ", approx_eq( np.sum(np.abs(x)), tokyo.sasum(x) )

def sasum_speed(int size):
    cdef int i, loops
    loops = speed_base*2000/size
    x = np.array( np.random.random( (size) ), dtype=np.float32 )

    print "sasum:      ",
    start = time.clock()
    for i in range(loops):
        tokyo.sasum( x )
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s " % (rate/1000)

def dasum_verify():
    x = np.array( np.random.random( (5) ),   dtype=np.float64 )
    print "dasum:  ", approx_eq( np.sum(np.abs(x)), tokyo.dasum(x) )

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

def isamax_verify():
    x = np.array( [0.06, -0.1, -0.05, -0.001, 0.07],   dtype=np.float32 )
    print "isamax: ", (  1 == tokyo.isamax(x) )

def isamax_speed(int size):
    cdef int i, loops
    loops = speed_base*2000/size
    x = np.array( np.random.random( (size) ), dtype=np.float32 )

    print "isamax:     ",
    start = time.clock()
    for i in range(loops):
        tokyo.isamax( x )
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s " % (rate/1000)

def idamax_verify():
    x = np.array( [0.06, -0.1, -0.05, -0.001, 0.07],   dtype=np.float64 )
    print "idamax: ", (  1 == tokyo.idamax(x) )

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



def sgemv_speed( int size ):

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


    
def sger_speed( int size ):

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

    
def sgemm_speed( int size ):

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

# set a matrix of floats to all zeros

def smsetzero_verify():
    A = np.array( np.random.random( (4,4) ), dtype=np.float32 )
    B = np.zeros( (4,4) )
    tokyo.smsetzero(A)
    print "smsetzero:  ", approx_eq( A, B )

def smsetzero_speed(int size):
    cdef int i, loops
    loops = speed_base*5000/(size*size)
    A = np.array( np.random.random( (size,size) ), dtype=np.float32 )

    print "smsetzero:  ",
    start = time.clock()
    for i in range(loops):
        tokyo.smsetzero( A )
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s " % (rate/1000)


# set a matrix of floats to all zeros

def dmsetzero_verify():
    A = np.array( np.random.random( (4,4) ), dtype=np.float64 )
    B = np.zeros( (4,4) )
    tokyo.dmsetzero(A)
    print "dmsetzero:  ", approx_eq( A, B )

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

def svsetzero_verify():
    x = np.array( np.random.random( (4) ), dtype=np.float32 )
    y = np.zeros( (4) )
    tokyo.svsetzero(x)
    print "svsetzero:  ", approx_eq( x, y )

def svsetzero_speed(int size):
    cdef int i, loops
    loops = speed_base*5000/size
    x = np.array( np.random.random( (size) ), dtype=np.float32 )

    print "svsetzero:  ",
    start = time.clock()
    for i in range(loops):
        tokyo.svsetzero( x )
    rate = loops/(time.clock()-start)
    print "%9.0f kc/s " % (rate/1000)

# set a vector of doubles to all zeros

def dvsetzero_verify():
    x = np.array( np.random.random( (4) ), dtype=np.float64 )
    y = np.zeros( (4) )
    tokyo.dvsetzero(x)
    print "dvsetzero:  ", approx_eq( x, y )

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

def smaxpy_verify():
    X = np.array( np.random.random( (4,4) ), dtype=np.float32 )
    Y = np.array( np.random.random( (4,4) ), dtype=np.float32 )
    temp = 1.2 * X + Y
    tokyo.smaxpy( 1.2, X, Y )
    print "smaxpy:     ", approx_eq( temp, Y )

def smaxpy_speed( int size ):
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


# double precision matrix += scalar * matrix

def dmaxpy_verify():
    X = np.array( np.random.random( (4,4) ), dtype=np.float64 )
    Y = np.array( np.random.random( (4,4) ), dtype=np.float64 )
    temp = 1.2 * X + Y
    tokyo.dmaxpy( 1.2, X, Y )
    print "dmaxpy:     ", approx_eq( temp, Y )

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


