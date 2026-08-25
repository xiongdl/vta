cmake_minimum_required(VERSION 3.18)
project(tvm_vta LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(TVM_SOURCE_DIR "" CACHE PATH "Path to the Apache TVM source tree")
set(TVM_BUILD_DIR "" CACHE PATH "Path containing libtvm_runtime")
set(VTA_CONFIG "${CMAKE_CURRENT_SOURCE_DIR}/config/vta_config.json" CACHE FILEPATH
    "VTA hardware configuration")
set(VTA_RUNTIME "sim" CACHE STRING "Runtime to build: sim or tsim")
set(VTA_OUTPUT_NAME "" CACHE STRING "Optional runtime library basename")
set_property(CACHE VTA_RUNTIME PROPERTY STRINGS sim tsim)

if(NOT EXISTS "${TVM_SOURCE_DIR}/include/tvm/runtime/c_runtime_api.h")
  message(FATAL_ERROR "TVM_SOURCE_DIR must point to an Apache TVM source tree")
endif()
if(NOT EXISTS "${VTA_CONFIG}")
  message(FATAL_ERROR "VTA_CONFIG does not exist: ${VTA_CONFIG}")
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter)
execute_process(
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/config/vta_config.py"
          "--use-cfg=${VTA_CONFIG}" --defs
  OUTPUT_VARIABLE VTA_DEFS_RAW
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
separate_arguments(VTA_DEFS NATIVE_COMMAND "${VTA_DEFS_RAW}")

find_library(TVM_RUNTIME_LIBRARY
  NAMES tvm_runtime
  PATHS "${TVM_BUILD_DIR}"
  NO_DEFAULT_PATH REQUIRED)

set(VTA_COMMON_SOURCES
  runtime/runtime.cc
  runtime/device_api.cc
  src/vmem/virtual_memory.cc)

if(VTA_RUNTIME STREQUAL "sim")
  set(VTA_TARGET_NAME vta_fsim)
  list(APPEND VTA_COMMON_SOURCES src/sim/sim_driver.cc src/sim/sim_tlpp.cc)
elseif(VTA_RUNTIME STREQUAL "tsim")
  set(VTA_TARGET_NAME vta_tsim)
  list(APPEND VTA_COMMON_SOURCES src/tsim/tsim_driver.cc src/dpi/module.cc)
  set(VERILATOR_INC_DIR "" CACHE PATH "Path to Verilator runtime headers")
  if(NOT EXISTS "${VERILATOR_INC_DIR}/vltstd/svdpi.h")
    message(FATAL_ERROR "VERILATOR_INC_DIR must contain vltstd/svdpi.h for TSIM")
  endif()
else()
  message(FATAL_ERROR "Unsupported VTA_RUNTIME=${VTA_RUNTIME}")
endif()

add_library(${VTA_TARGET_NAME} SHARED ${VTA_COMMON_SOURCES})
if(VTA_OUTPUT_NAME)
  set_target_properties(${VTA_TARGET_NAME} PROPERTIES OUTPUT_NAME "${VTA_OUTPUT_NAME}")
endif()
target_include_directories(${VTA_TARGET_NAME} PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/runtime"
  "${TVM_SOURCE_DIR}/include"
  "${TVM_SOURCE_DIR}/3rdparty/dlpack/include"
  "${TVM_SOURCE_DIR}/3rdparty/dmlc-core/include")
if(VTA_RUNTIME STREQUAL "tsim")
  target_include_directories(${VTA_TARGET_NAME} PRIVATE
    "${VERILATOR_INC_DIR}" "${VERILATOR_INC_DIR}/vltstd")
  if(EXISTS "${VERILATOR_INC_DIR}/verilated_threads.cpp")
    target_sources(${VTA_TARGET_NAME} PRIVATE "${VERILATOR_INC_DIR}/verilated_threads.cpp")
  endif()
endif()
if(APPLE)
  # Load as a TVM plugin. Linking another libtvm_runtime instance would create
  # a second global Registry invisible to the host Python process.
  target_link_options(${VTA_TARGET_NAME} PRIVATE "-Wl,-undefined,dynamic_lookup")
else()
  target_link_libraries(${VTA_TARGET_NAME} PRIVATE "${TVM_RUNTIME_LIBRARY}")
endif()
target_compile_definitions(${VTA_TARGET_NAME} PRIVATE
  DMLC_USE_FOPEN64=0
  DMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>
  NDEBUG=1
  TVM_INDEX_DEFAULT_I64=1
  TVM_KALLOC_ALIGNMENT=64
  __STDC_CONSTANT_MACROS
  __STDC_FORMAT_MACROS
  __STDC_LIMIT_MACROS)
if(VTA_RUNTIME STREQUAL "sim")
  target_compile_definitions(${VTA_TARGET_NAME} PRIVATE USE_FSIM_TLPP)
endif()
foreach(VTA_DEF IN LISTS VTA_DEFS)
  string(REGEX REPLACE "^-D" "" VTA_DEF_STRIPPED "${VTA_DEF}")
  # vta_config.py emits VTA_TARGET=tsim. A bare token target value becomes a
  # preprocessor macro and corrupts C++ namespaces with the same identifier.
  # Runtime selection is expressed by the source set, so this macro is not
  # needed by the VTA ABI or instruction definitions.
  if(NOT VTA_DEF_STRIPPED MATCHES "^(VTA_TARGET|VTA_HW_VER)=")
    target_compile_definitions(${VTA_TARGET_NAME} PRIVATE "${VTA_DEF_STRIPPED}")
  endif()
endforeach()

install(TARGETS ${VTA_TARGET_NAME} LIBRARY DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
