
import numpy as np
cimport numpy as np

import tokyo
cimport tokyo

cdef np.ndarray x = np.array( [ 1.0, 2.0, 3.0, 4.0 ] )
cdef np.ndarray y = np.array( [ 7.0, 8.0, 9.0, 0.0 ] )

print tokyo.dger( x, y )
