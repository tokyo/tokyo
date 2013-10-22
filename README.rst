=====
Tokyo
=====

This is a fork of Tokyo that fixes a few issues related to Cython syntax. It
was tested on OSX 10.6 with Python 2.7 and Cython 0.14. The original contents
of the README file can be found below. The intent is to complete the interface
and interface LAPACK subroutines. Tokyo was started by Shane Legg
<shane@vetta.org>. See his
`blog post
<http://www.vetta.org/2009/09/tokyo-a-cython-blas-wrapper-for-fast-matrix-math>`_.

Enjoy!


The Plan
========

The current plan is to cover the BLAS Level 1, 2 and 3 entirely in real single
and double precision. This will be done gradually.

The next step will be to add support for LAPACK subroutines as we need them.
Here again, real single and double precision are the priority.


Contributing
============

If you are interested in contributing to other aspects of Tokyo, here is a list
of ideas of medium to low priority aspects in which we would welcome initial
contributions:

* Automatic detection of local (optimized) BLAS installation
* Proper unit tests with `nose
  <http://somethingaboutorange.com/mrl/projects/nose>`_
* Interface to complex single and double precision BLAS subroutines

Other ideas welcome!


Original README
===============

Tokyo is a partial BLAS wrapper written in Cython.  It allows you
to call various BLAS matrix and vector routines in double precision
without using the numpy/scipy wrapper -- which is in slow interpreted
Python.  Speed ups range from nothing to over 100 times depending on
the size of the matrices and vectors and which operations you need to
perform.

There isn't really any installation.  However, you need to do these
things:

First of all make sure you have Python and Cython!  I use Python 2.6
and Cython 0.12

Now make sure you have cblas installed.  On a unix system try::

    locate cblas.h

If that finds something for you then you have it and you also now
know where it is!  I installed cblas on my Ubuntu system using the
synaptic package manager.  I think it came with BLAS or ATLAS or
perhaps Sage.  I don't remember exactly which.  There is a cblas
that comes with GSL, but I haven't tried hooking Tokyo up to that.
I'm sure it's possible but it might require renaming some of the
functions in the wrapper code.  My suggestion, at least to start
with, just use plain cblas.  Later on you can hook up an optimsed
ATLAS for your machine or Goto BLAS etc.

Once you have cblas and know where cblas.h is, you need to go to
the setup.py file used in the Cython compilation and make sure the
directories are pointing somewhere where they will see this file.
If you're on Ubuntu the directories should already be correct.

Now build it in the usual Cython way::

    python setup.py build_ext --inplace

If you get no errors, now run the verification test to check
it's computing the right things::

    python verify.py

Next try the single and double precision performance tests::

    python single_speed.py
    python double_speed.py

These will tell you whether it's worth switching your code
to use Tokyo or not for your machine and matrix sizes.

I've named the functions like they are named in BLAS.  It's
a bit strange looking to start with but it does make sense.
The first character is s for single precsion and d for double.
I've put an _ at the end of a name to indicate that it's a
raw BLAS call will all the options exposed.  You probably
won't want to use these.  When I have multiple different
functions for a single BLAS function to expose varying sets of
options, I put the number of parameters as the last character,
unless it's the simplest call possible in which case I leave
this off.  You probably want to start with the latter calls
before moving on to the more complex ones as the simpler ones
are closer to the existing numpy/scipy ones in that they create
and return a matrix as a result and make sure that it's set to
zero etc. before the computation begins.  This incurs some
overhead, so once your code is working with that, you might want
to move to some of the other versions of the functions, where
they are provided, that do things like writing results into
preexisting matrices etc. rather than creating new objects to
return the results on each call.

The best place to see what all the functions provided are is
cheat_sheet.txt  If something there confuses you have a look
at the proper definition in tokyo.pxd

In case you look at the implementation and decide to remove
all the size and type checks that are performed, don't!
I've made sure that these have been done in an efficient way and
so they have almost no impact on the performance and may well
save you a lot of debugging time and strange crashes.

Finally, I should point out that you should give your matrices
and vectors a cdef ndarray type.  As per usual in Cython, giving
variables a cdef type isn't necessary for the code to work but
it allows Cython to produce faster C code.  In this case it
saves Cython having to perform a bunch of Python checks to see if
your matrix really is of type ndarray before each function call.


.. image:: https://d2weczhvl823v0.cloudfront.net/tokyo/tokyo/trend.png
   :alt: Bitdeli badge
   :target: https://bitdeli.com/free

