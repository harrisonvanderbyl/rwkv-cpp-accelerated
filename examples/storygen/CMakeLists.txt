cmake_minimum_required(VERSION 3.17)

# ensure CXX at least 11.3.0

# if cuda is not found, throw error
if(NOT CMAKE_CUDA_COMPILER)
        
    # ensure CXX at least 11.3.0
    list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)

    set(CMAKE_CXX_STANDARD 17)

    # set variable for include directory of ../../include
    set(RWKV_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../include CACHE PATH "Path to rwkv include directory")


    # add parent directory + ../include to include path
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/.. ${RWKV_INCLUDE_DIR} /opt/rocm/include)

    #print the hip include dirs

    # add hip file to project executables from include/rwkv/hip/rwkv.cpp


    enable_language(CXX HIP)
    project("rwkv.hip" CXX HIP)

    find_package(HIP)

    add_executable(rwkv
        storygen.cpp
        ../../include/rwkv/cuda/rwkv.hip
    )
    # force all files use hipcc
    set_source_files_properties(storygen.cpp ../../include/rwkv/cuda/rwkv.hip PROPERTIES LANGUAGE HIP)

    # add ./samplers to include path
    target_include_directories(rwkv PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/samplers)

    # static link against hip
    # target_link_libraries(rwkv hip::device)

    # add hip runtime
    


else()
    if(WIN32)
    else()
        set(CMAKE_CXX_COMPILER "g++")
        set(CMAKE_C_COMPILER "gcc")
    endif()

    set(CMAKE_CXX_STANDARD 17)

    # set variable for include directory of ../../include
    set(RWKV_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../include CACHE PATH "Path to rwkv include directory")


    # add parent directory + ../include to include path
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/.. ${RWKV_INCLUDE_DIR})

    # add cuda file to project executables from include/rwkv/cuda/rwkv.cu

    project("rwkv.cuda" CXX CUDA)


    find_package(CUDAToolkit)

    add_executable(rwkv
        storygen.cpp
        ../../include/rwkv/cuda/rwkv.cu
    )

    # add ./samplers to include path
    target_include_directories(rwkv PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/samplers)

    # link cuda lcudart_static
    target_link_libraries(rwkv PRIVATE CUDA::cudart_static)

endif()