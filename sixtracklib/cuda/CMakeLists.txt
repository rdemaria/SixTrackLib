# ==============================================================================
# CUDA sixtracklib architecture

message( STATUS "---- processing sixtracklib/cuda/CMakeLists.txt" )

if( SIXTRACKL_CUDA_COMPUTE_ARCH_LIST )

    foreach( A IN LISTS SIXTRACKL_CUDA_COMPUTE_ARCH_LIST )
        string( APPEND CMAKE_CUDA_FLAGS
                " -gencode=arch=compute_${A},code=compute_${A}" )
    endforeach()
endif()

string( APPEND CMAKE_CUDA_FLAGS " -D_FORCE_INLINES" )
string( APPEND CMAKE_CUDA_FLAGS " -cudart shared" )

if( CMAKE_BUILD_TYPE MATCHES "Debug" )
    string( APPEND CMAKE_CUDA_FLAGS " -g -G" )
else()
    string( APPEND CMAKE_CUDA_FLAGS " -fmad=false" )
    string( APPEND CMAKE_CUDA_FLAGS " -lineinfo" )
endif()

if( SIXTRACKL_ENABLE_APERATURE_CHECK )
    string( APPEND CMAKE_CUDA_FLAGS " -DSIXTRL_ENABLE_APERATURE_CHECK=1" )
else()
    string( APPEND CMAKE_CUDA_FLAGS " -DSIXTRL_ENABLE_APERATURE_CHECK=0" )
endif()

set( SIXTRACKL_CUDA_HOST_DEFINITIONS )
list( APPEND SIXTRACKL_CUDA_HOST_DEFINITIONS ${SIXTRACKLIB_NAMESPACE_FLAGS} )

# ------------------------------------------------------------------------------

set( SIXTRACKLIB_CUDA_C99_INSTALL_PREFIX
     "${SIXTRACKLIB_C99_INSTALL_PREFIX}/cuda" )

set( SIXTRACKLIB_CUDA_CXX_INSTALL_PREFIX
     "${SIXTRACKLIB_CXX_INSTALL_PREFIX}/cuda" )

# -----------------------------------------------------------------------------
# submodules

add_subdirectory( control )
add_subdirectory( track )

set( SIXTRACKLIB_CUDA_PART_HOST_HEADERS
     wrappers/controller_wrappers.h
     wrappers/track_job_wrappers.h
     cuda_tools.h
)

set( SIXTRACKLIB_CUDA_PART_HOST_SOURCES
     wrappers/controller_wrappers.cu
     wrappers/track_job_wrappers.cu
)

add_library( sixtrack_cuda_host OBJECT
             ${SIXTRACKLIB_CUDA_PART_HOST_HEADERS}
             ${SIXTRACKLIB_CUDA_PART_HOST_SOURCES}
)

target_compile_definitions( sixtrack_cuda_host
    PUBLIC ${SIXTRACKL_CUDA_HOST_DEFINITIONS} )

target_include_directories( sixtrack_cuda_host
    PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    PUBLIC  $<BUILD_INTERFACE:${SIXTRACKL_CUDA_INCLUDE_DIRS}> )

set_target_properties( sixtrack_cuda_host PROPERTIES
    POSITION_INDEPENDENT_CODE ON LINKER_LANGUAGE C CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON )

# target_compile_options( sixtrack_cuda_host BEFORE PUBLIC
#         ${SIXTRACKLIB_CXX_FLAGS} )

# ------------------------------------------------------------------------------
# pass on sixtrack_cuda_host as a module for sixtracklib:

set(  SIXTRACKL_LIBRARY_MODULES ${SIXTRACKL_LIBRARY_MODULES}
    $<TARGET_OBJECTS:sixtrack_cuda_host> CACHE INTERNAL "" FORCE )

find_library( CUDA_LIBRARY cuda ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES} )
find_library( CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES} )

if( SIXTRACKL_ENABLE_CUDA_RTC )
    find_library( CUDA_NVRTC_LIBRARY nvrtc ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES} )
else()
    set( CUDA_NVRTC_LIBRARY )
endif()

set( SIXTRACKL_LINK_LIBRARIES ${SIXTRACKL_LINK_LIBRARIES} ${CUDA_NVRTC_LIBRARY}
    ${CUDA_LIBRARY} ${CUDART_LIBRARY} CACHE INTERNAL "" FORCE )

# ------------------------------------------------------------------------------
# --- CUDA Host C++ object library

set( SIXTRACKL_CUDA_CXX_HEADERS argument.hpp controller.hpp )
set( SIXTRACKL_CUDA_C99_HEADERS definitions.h control/argument_base.h
     argument.h controller.h )

# ------------------------------------------------------------------------------
# --- CUDA Kernel object library:

set( SIXTRACKL_CUDA_DEVICE_DEFINITIONS )
list( APPEND SIXTRACKL_CUDA_DEVICE_DEFINITIONS "_GPUCODE=1" )
list( APPEND SIXTRACKL_CUDA_DEVICE_DEFINITIONS ${SIXTRACKLIB_NAMESPACE_FLAGS} )

set( SIXTRACKLIB_CUDA_PART_DEVICE_HEADERS
     cuda_tools.h
     kernels/be_monitors_assign_out_buffer.cuh
     kernels/elem_by_elem_assign_out_buffer.cuh
     kernels/extract_particles_addr.cuh
     kernels/managed_buffer_remap.cuh
     kernels/track_particles.cuh
     kernels/assign_address_item.cuh
)

set( SIXTRACKLIB_CUDA_PART_DEVICE_SOURCES
     kernels/be_monitors_assign_out_buffer.cu
     kernels/elem_by_elem_assign_out_buffer.cu
     kernels/extract_particles_addr.cu
     kernels/managed_buffer_remap.cu
     kernels/track_particles.cu
     kernels/assign_address_item.cu
)

add_library( sixtrack_cuda_device OBJECT
    ${SIXTRACKLIB_CUDA_PART_DEVICE_HEADERS}
    ${SIXTRACKLIB_CUDA_PART_DEVICE_SOURCES} )

target_compile_definitions( sixtrack_cuda_device
    PUBLIC ${SIXTRACKL_CUDA_DEVICE_DEFINITIONS} )

set_target_properties( sixtrack_cuda_device PROPERTIES
    POSITION_INDEPENDENT_CODE ON LINKER_LANGUAGE C CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON CUDA_SEPARABLE_COMPILATION  ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON CUDA_STANDARD 11 CUDA_STANDARD_REQUIRED 11 )

target_include_directories( sixtrack_cuda_device PRIVATE ${CMAKE_SOURCE_DIR} )

# ------------------------------------------------------------------------------
# pass on sixtrack_cuda_device as a module for sixtracklib:

set( SIXTRACKL_LIBRARY_MODULES ${SIXTRACKL_LIBRARY_MODULES}
    $<TARGET_OBJECTS:sixtrack_cuda_device> CACHE INTERNAL "" FORCE )

# ------------------------------------------------------------------------------
# install:

set( SIXTRACKLIB_CUDA_INSTALL_PATH
     "${CMAKE_INSTALL_PREFIX}/include/sixtracklib/cuda" )

if( SIXTRACKLIB_CUDA_PART_HOST_HEADERS )
    install( FILES ${SIXTRACKLIB_CUDA_PART_HOST_HEADERS}
             DESTINATION ${SIXTRACKLIB_CUDA_INSTALL_PATH} )
endif()

if( SIXTRACKLIB_CUDA_PART_DEVICE_HEADERS OR
    SIXTRACKLIB_CUDA_PART_DEVICE_SOURCES )

    set( SIXTRACKLIB_CUDA_DEVICE_INSTALL_PATH
         "${SIXTRACKLIB_CUDA_INSTALL_PATH}/kernels" )

    install( FILES ${SIXTRACKLIB_CUDA_PART_DEVICE_HEADERS}
                   ${SIXTRACKLIB_CUDA_PART_DEVICE_SOURCES}
             DESTINATION ${SIXTRACKLIB_CUDA_DEVICE_INSTALL_PATH} )
endif()

if( SIXTRACKL_CUDA_C99_HEADERS )
    install( FILES ${SIXTRACKL_CUDA_C99_HEADERS}
             DESTINATION ${SIXTRACKLIB_CUDA_INSTALL_PATH} )
endif()

if( SIXTRACKL_ENABLE_CXX )
    if( SIXTRACKL_CUDA_CXX_HEADERS )
        install( FILES ${SIXTRACKL_CUDA_CXX_HEADERS}
                 DESTINATION ${SIXTRACKLIB_CUDA_INSTALL_PATH} )
    endif()
endif()
