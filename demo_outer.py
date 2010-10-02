#!/usr/bin/env python

import numpy as np

x = np.array( [ 1, 2, 3, 4 ], dtype=np.float64 )
y = np.array( [ 7, 8, 9, 0 ], dtype=np.float64 )

print np.outer( x, y )
