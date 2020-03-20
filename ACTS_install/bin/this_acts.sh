# set up environment variables
# the ${VAR:+:} part adds a double colon only if VAR is not empty
export PATH="/home/bomki/Projects/ACTS/ACTS_install/bin${PATH:+:}${PATH}"
export LD_LIBRARY_PATH="/home/bomki/Projects/ACTS/ACTS_install/lib64${LD_LIBRARY_PATH:+:}${LD_LIBRARY_PATH}"
export DYLD_LIBRARY_PATH="/home/bomki/Projects/ACTS/ACTS_install/lib64${DYLD_LIBRARY_PATH:+:}${DYLD_LIBRARY_PATH}"
