#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "ActsFatras" for configuration ""
set_property(TARGET ActsFatras APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(ActsFatras PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib64/libActsFatras.so"
  IMPORTED_SONAME_NOCONFIG "libActsFatras.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS ActsFatras )
list(APPEND _IMPORT_CHECK_FILES_FOR_ActsFatras "${_IMPORT_PREFIX}/lib64/libActsFatras.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
