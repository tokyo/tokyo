# Much simplified system tools for now.
import os
import inspect

def get_include():
    thisfile = inspect.getabsfile(inspect.currentframe())
    thispath = os.path.dirname(thisfile)
    toppath  = os.path.dirname(thispath)
    return toppath
