# make the top-level functions and classes available for the module
from . probability import  *
from . estimators import *
from . import utils
from . import generators
try:
    from . import md
except ModuleNotFoundError:
    print('informant.md requires pyemma installation.')

try:
    from . import plots
except ModuleNotFoundError:
    print('informant.plots require matplotlib installation')