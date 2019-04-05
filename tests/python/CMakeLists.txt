# tests/python/CMakeLists.txt

message( STATUS "---- processing tests/python/CMakeLists.txt" )

if( SIXTRACKL_ENABLE_PROGRAMM_TESTS )

    # -------------------------------------------------------------------------
    # test_cbuffer_st_buffer:

    add_test( NAME Python_CBufferStBufferTests
        COMMAND ${PYTHON_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/test_cbuffer_st_buffer.py
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/python )

    # -------------------------------------------------------------------------
    # test_output_buffer:

    add_test( NAME Python_OutputBufferTests
        COMMAND ${PYTHON_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/test_output_buffer.py
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/python )
endif()


#end: tests/python/CMakeLists.txt