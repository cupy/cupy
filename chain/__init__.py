def _init():
    import pycuda.autoinit
    import scikits.cuda.linalg as culinalg
    culinalg.init()
_init()

from variable import Variable
from function import Function, Layer
from optimizer import Optimizer

import functions.basic_math  # Install variable operators
