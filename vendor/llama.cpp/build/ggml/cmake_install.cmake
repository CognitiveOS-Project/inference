# Install script for directory: /workspace/inference/vendor/llama.cpp/ggml

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "0")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/workspace/inference/vendor/llama.cpp/build/ggml/src/cmake_install.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/workspace/inference/vendor/llama.cpp/build/libggml.a")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES
    "/workspace/inference/vendor/llama.cpp/ggml/include/ggml.h"
    "/workspace/inference/vendor/llama.cpp/ggml/include/ggml-cpu.h"
    "/workspace/inference/vendor/llama.cpp/ggml/include/ggml-alloc.h"
    "/workspace/inference/vendor/llama.cpp/ggml/include/ggml-backend.h"
    "/workspace/inference/vendor/llama.cpp/ggml/include/ggml-blas.h"
    "/workspace/inference/vendor/llama.cpp/ggml/include/ggml-cann.h"
    "/workspace/inference/vendor/llama.cpp/ggml/include/ggml-cpp.h"
    "/workspace/inference/vendor/llama.cpp/ggml/include/ggml-cuda.h"
    "/workspace/inference/vendor/llama.cpp/ggml/include/ggml-opt.h"
    "/workspace/inference/vendor/llama.cpp/ggml/include/ggml-metal.h"
    "/workspace/inference/vendor/llama.cpp/ggml/include/ggml-rpc.h"
    "/workspace/inference/vendor/llama.cpp/ggml/include/ggml-virtgpu.h"
    "/workspace/inference/vendor/llama.cpp/ggml/include/ggml-sycl.h"
    "/workspace/inference/vendor/llama.cpp/ggml/include/ggml-vulkan.h"
    "/workspace/inference/vendor/llama.cpp/ggml/include/ggml-webgpu.h"
    "/workspace/inference/vendor/llama.cpp/ggml/include/ggml-zendnn.h"
    "/workspace/inference/vendor/llama.cpp/ggml/include/ggml-openvino.h"
    "/workspace/inference/vendor/llama.cpp/ggml/include/gguf.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/workspace/inference/vendor/llama.cpp/build/libggml-base.a")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/ggml" TYPE FILE FILES
    "/workspace/inference/vendor/llama.cpp/build/ggml/ggml-config.cmake"
    "/workspace/inference/vendor/llama.cpp/build/ggml/ggml-version.cmake"
    )
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/workspace/inference/vendor/llama.cpp/build/ggml/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
