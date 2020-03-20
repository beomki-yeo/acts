# CMake config for the Acts package
#
# Only defines CMake targets for requested and available components.
# No addititional variables are defined. All additional information, e.g.
# include directories and dependencies, are defined as target-specific
# properties and are automatically propagated when linking to the target.


####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was ActsConfig.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

set(Acts_SUPPORTED_COMPONENTS Core;Fatras)
set(Acts_COMMIT_HASH "26c5717684d46bf36c071ba2c89e5272f17cf814-dirty")
set(Acts_COMMIT_HASH_SHORT "26c571768-dirty")

# print version and components information
if(NOT Acts_FIND_QUIETLY)
  message(STATUS "found Acts version ${Acts_VERSION} - ${Acts_COMMIT_HASH_SHORT}")
  message(STATUS "supported components:")
  foreach(_component ${_supported_components})
    message(STATUS "  - ${_component}")
  endforeach()
endif()

# check that requested components are available
foreach(_component ${Acts_FIND_COMPONENTS})
  # check if this component is supported
  if(NOT _component IN_LIST Acts_SUPPORTED_COMPONENTS)
    if (${Acts_FIND_REQUIRED_${_component}})
      # not supported, but required -> fail
      set(Acts_FOUND False)
      message(FATAL_ERROR "required component \"${_component}\" not found")
    else()
      # not supported and optional -> skip
      list(REMOVE_ITEM Acts_FIND_COMPONENTS ${_component})
      if(NOT Acts_FIND_QUIETLY)
        message(STATUS "optional component \"${_component}\" not found")
      endif()
    endif()
  endif()
endforeach()

# load requested and available components
if(NOT Acts_FIND_QUIETLY)
  message(STATUS "loading components:")
endif()
foreach(_component ${Acts_FIND_COMPONENTS})
  if(NOT Acts_FIND_QUIETLY)
    message(STATUS "  - ${_component}")
  endif()
  # include the targets file to create the imported targets for the user
  include(${CMAKE_CURRENT_LIST_DIR}/Acts${_component}Targets.cmake)
endforeach()
