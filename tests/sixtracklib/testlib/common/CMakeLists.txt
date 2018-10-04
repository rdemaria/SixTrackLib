# tests/sixtracklib/testlib/common/CMakeLists.txt

set( TESTLIB_COMMON_C99_HEADERS
     beam_elements.h
     generic_buffer_obj.h
     gpu_kernel.h
     particles.h
     random.h
     time.h
)

set( TESTLIB_COMMON_C99_SOURCES
     beam_elements.c
     gpu_kernel.c
     particles.c
     random.c
     time.c
)

set( TESTLIB_COMMON_CXX_HEADERS )
set( TESTLIB_COMMON_CXX_SOURCES )


add_library( sixtrack_test_common OBJECT
            ${TESTLIB_COMMON_C99_HEADERS}
            ${TESTLIB_COMMON_C99_SOURCES}
            ${TESTLIB_COMMON_CXX_HEADERS}
            ${TESTLIB_COMMON_CXX_SOURCES}
)

target_include_directories( sixtrack_common_buffer PRIVATE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/tests>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include/tests> )

set_target_properties( sixtrack_test_common PROPERTIES LINKER_LANGUAGE C )
set_target_properties( sixtrack_test_common PROPERTIES POSITION_INDEPENDENT_CODE ON )
set_target_properties( sixtrack_test_common PROPERTIES C_STANDARD 99 )
set_target_properties( sixtrack_test_common PROPERTIES C_STANDARD_REQUIRED ON )

target_compile_options( sixtrack_test_common PRIVATE
                        -Wall -Werror -pedantic -ansi
                        ${SIXTRACKLIB_CPU_FLAGS} )

# ------------------------------------------------------------------------------
# pass on sixtrack_test_common as a module for sixtrack_test:

set(  SIXTRACKL_TESTLIB_LIBRARY_MODULES
    ${SIXTRACKL_TESTLIB_LIBRARY_MODULES}
    $<TARGET_OBJECTS:sixtrack_test_common> CACHE INTERNAL "" FORCE )


# ------------------------------------------------------------------------------
# Install :

if( TESTLIB_COMMON_C99_HEADERS )
    set( TESTLIB_COMMON_C99_INSTALL_PATH include/sixtracklib/testlib/common )

    install( FILES ${TESTLIB_COMMON_C99_HEADERS}
             DESTINATION ${TESTLIB_COMMON_C99_INSTALL_PATH} )

endif()


if( TESTLIB_COMMON_CXX_HEADERS )
    set( TESTLIB_COMMON_CXX_INSTALL_PATH include/sixtracklib/testlib/common )

    install( FILES ${TESTLIB_COMMON_CXX_HEADERS}
             DESTINATION ${TESTLIB_COMMON_CXX_INSTALL_PATH} )

endif()

#end: tests/sixtracklib/testlib/common/CMakeLists.txt