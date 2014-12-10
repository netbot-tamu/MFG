# Look for csparse; note the difference in the directory specifications!
# The version of csparse this module looks for is the one included with G2o
FIND_PATH(CSPARSE_INCLUDE_DIR NAMES cs.h
   PATHS
   /usr/include
   /usr/include/suitesparse   
   /usr/include/ufsparse
   /opt/local/include   
   /sw/include
   /sw/include/ufsparse
   /usr/local/include   
   /opt/local/include/ufsparse
   /usr/local/include/ufsparse
# G2o installs its csparse headers in the following locations
   /opt/local/include/EXTERNAL/csparse
   /usr/local/include/EXTERNAL/csparse	
)

FIND_LIBRARY(CSPARSE_LIBRARY NAMES cxsparse
   PATHS
   /usr/lib
   /usr/local/lib
   /opt/local/lib
   /sw/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CSPARSE DEFAULT_MSG
   CSPARSE_INCLUDE_DIR
   CSPARSE_LIBRARY
)
