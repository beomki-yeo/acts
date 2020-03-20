#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "ActsCore" for configuration ""
set_property(TARGET ActsCore APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(ActsCore PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib64/libActsCore.so"
  IMPORTED_SONAME_NOCONFIG "libActsCore.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS ActsCore )
list(APPEND _IMPORT_CHECK_FILES_FOR_ActsCore "${_IMPORT_PREFIX}/lib64/libActsCore.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
