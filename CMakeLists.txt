cmake_minimum_required(VERSION 3.18)

project(vta LANGUAGES CXX)

set(TVM_PATH "" CACHE PATH "Path to the TVM source tree")
set(VTA_PATH "${CMAKE_CURRENT_SOURCE_DIR}" CACHE PATH "Path to the VTA source tree")

if(NOT TVM_PATH)
  message(FATAL_ERROR "TVM_PATH is required and must point to the TVM source tree")
endif()

file(REAL_PATH "${TVM_PATH}" TVM_PATH)
file(REAL_PATH "${VTA_PATH}" VTA_PATH)
if(NOT EXISTS "${VTA_PATH}/config/vta_config.py")
  message(FATAL_ERROR "VTA_PATH must point to the VTA source tree: ${VTA_PATH}")
endif()

set(TVM_INCLUDE_DIR "${TVM_PATH}/include")
set(TVM_WORKSPACE_POOL_HEADER
    "${TVM_INCLUDE_DIR}/../src/runtime/workspace_pool.h")

foreach(required_path
        "${TVM_INCLUDE_DIR}/tvm/runtime/registry.h"
        "${TVM_WORKSPACE_POOL_HEADER}"
        "${TVM_PATH}/3rdparty/dlpack/include/dlpack/dlpack.h"
        "${TVM_PATH}/3rdparty/dmlc-core/include/dmlc/logging.h")
  if(NOT EXISTS "${required_path}")
    message(FATAL_ERROR "Required TVM dependency was not found: ${required_path}")
  endif()
endforeach()

find_library(
  TVM_LIBRARY
  NAMES tvm
  PATHS "${TVM_PATH}/build"
  NO_DEFAULT_PATH
  REQUIRED)

find_package(Python3 REQUIRED COMPONENTS Interpreter)
execute_process(
  COMMAND "${Python3_EXECUTABLE}" "${VTA_PATH}/config/vta_config.py" --target
  RESULT_VARIABLE VTA_CONFIG_TARGET_RESULT
  OUTPUT_VARIABLE VTA_TARGET
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT VTA_CONFIG_TARGET_RESULT EQUAL 0)
  message(FATAL_ERROR "Failed to read the VTA target configuration")
endif()
if(NOT VTA_TARGET STREQUAL "sim")
  message(FATAL_ERROR
          "Only the FSIM target is currently supported; configured target is '${VTA_TARGET}'")
endif()

execute_process(
  COMMAND "${Python3_EXECUTABLE}" "${VTA_PATH}/config/vta_config.py" --defs
  RESULT_VARIABLE VTA_CONFIG_DEFS_RESULT
  OUTPUT_VARIABLE VTA_DEFINITIONS_RAW
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT VTA_CONFIG_DEFS_RESULT EQUAL 0)
  message(FATAL_ERROR "Failed to read the VTA compile definitions")
endif()
separate_arguments(VTA_DEFINITIONS NATIVE_COMMAND "${VTA_DEFINITIONS_RAW}")
list(TRANSFORM VTA_DEFINITIONS REPLACE "^-D" "")

add_library(
  vta_fsim SHARED
  "${VTA_PATH}/src/runtime/device_api.cc"
  "${VTA_PATH}/src/runtime/runtime.cc"
  "${VTA_PATH}/src/sim/sim_driver.cc"
  "${VTA_PATH}/src/sim/sim_tlpp.cc"
  "${VTA_PATH}/src/vmem/virtual_memory.cc")

target_compile_features(vta_fsim PRIVATE cxx_std_17)
target_compile_definitions(
  vta_fsim
  PRIVATE
    DMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>
    USE_FSIM_TLPP
    ${VTA_DEFINITIONS})
target_include_directories(
  vta_fsim
  PRIVATE
    "${VTA_PATH}/include"
    "${TVM_INCLUDE_DIR}"
    "${TVM_PATH}/3rdparty/dlpack/include"
    "${TVM_PATH}/3rdparty/dmlc-core/include")
target_link_libraries(vta_fsim PRIVATE "${TVM_LIBRARY}")

set_target_properties(
  vta_fsim
  PROPERTIES
    CXX_EXTENSIONS OFF
    BUILD_RPATH "${TVM_PATH}/build")
