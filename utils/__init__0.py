# disabled (by name) to load only separate modules like:
# import utils.ak as ak  /  from utils import ak  /  etc.
# but  import utils  doesn't work now
from .ak import *
from .copies import *
from .ts import *
from .graphs import *
# each module depends on number of specific packages
# and we do not want to load everything always, like: pmdarima, statsmodels, networkx
# only to use some simple utility

# if this file is enabled then to run
# import utils.ak as ak
# all packages needed for other modules: copies, ts, graphs, must be installed...
# it's nonsense!
