cmake_minimum_required(VERSION 3.12.4)

# *******************************************
# ************* CMake Content ***************
# *******************************************
# This CMake create a workspace containing the following projects
# 
# Programs
#  - image_denoising

set (PROJECT_NAME image_denoising)

project(${PROJECT_NAME})

# Disable C and C++ compiler extensions.
# C/CXX_EXTENSIONS are ON by default to allow the compilers to use extended
# variants of the C/CXX language.
# However, this could expose cross-platform bugs in user code or in the headers
# of third-party dependencies and thus it is strongly suggested to turn
# extensions off.
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT ${CMAKE_GENERATOR} MATCHES "Visual Studio.*")

	# Link with pthread
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

	# Set to 1 to activate debug compilation (0 for release)
	set(DEBUG 0)

	if(${DEBUG})
			MESSAGE("Generate Debug project")
			set(CMAKE_BUILD_TYPE Debug)
			set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/Debug)
			set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -pg -Wall")
	else()
			MESSAGE("Generate Release project")
			set(CMAKE_BUILD_TYPE Release)
			set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/Release)
			set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall")
	endif()
	#add libmath during non visual studio builds
	set(CMAKE_EXTRA_LIB m)
else()
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
	add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

# Add definitions for testing purposes
if(${TESTING})
	MESSAGE("Testing mode")
	add_definitions(-DNO_CONSOLE_CONTROL -DNB_GENERATIONS=10)
endif()


# *******************************************
# ************ CIFAR database ***************
# *******************************************
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/dat/cifar-10/cifar-10-binary.tar.gz")
	message(STATUS "Download CIFAR-10 training database (may take a few seconds).")
	file(DOWNLOAD https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz ${CMAKE_CURRENT_SOURCE_DIR}/dat/cifar-10/cifar-10-binary.tar.gz)

else()
	message(STATUS "CIFAR-10 data already downloaded.")
endif()	

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/dat/cifar-10/cifar-10-batches-bin/data_batch_1.bin")
	message(STATUS "Extract CIFAR-10 database.")
	execute_process(
		COMMAND tar -xf ${CMAKE_SOURCE_DIR}/dat/cifar-10/cifar-10-binary.tar.gz
		WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/dat/cifar-10
	)

else()
	message(STATUS "CIFAR-10 data already extracted")
endif()

if(NOT EXISTS "${CMAKE_SOURCE_DIR}/dat/cifar-10/denoised-cifar-10")
	execute_process(COMMAND mkdir ${CMAKE_SOURCE_DIR}/dat/cifar-10/denoised-cifar-10
			WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/dat/cifar-10
			)
endif()

set(CIFAR_10_DATA_DIR ${CMAKE_SOURCE_DIR}/dat/cifar-10/cifar-10-batches-bin)
set(NOISY_CIFAR_10_DATA_DIR ${CMAKE_SOURCE_DIR}/dat/cifar-10/noisy-cifar-10-batches-bin)
set(RESULT_DENOISING_CIFAR_10_DIR ${CMAKE_SOURCE_DIR}/dat/cifar-10/denoised-cifar-10)

# *******************************************
# *********** GEGELATI LIBRARY **************
# *******************************************

if(WIN32)
	set(LIBS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib)
    # find the gegelatilib-x.y.z folder in the lib directory.
	file(GLOB GEGELATI_ROOT_DIR "${LIBS_DIR}/gegelatilib-[\\.|0-9]*")
	set(ENV{GEGELATI_DIR} ${GEGELATI_ROOT_DIR})
endif()
find_package(GEGELATI)


if (WIN32)
	file(GLOB
		GEGELATI_DLL
		${GEGELATI_ROOT_DIR}/bin/*.dll
	)

	MESSAGE("Copy GEGELATI DLLs into ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
	if(NOT ${CMAKE_GENERATOR} MATCHES "Visual Studio.*")
		file(COPY ${GEGELATI_DLL} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
	else()
		file(COPY ${GEGELATI_DLL} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Debug)
		file(COPY ${GEGELATI_DLL} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Release)
	endif()
endif()


# *******************************************
# ************ Opencv LIBRARY  **************
# *******************************************

find_package(OpenCV REQUIRED)

# *******************************************
# ************** Executable  ****************
# *******************************************
#Noising the dataset
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/dat/cifar-10/noisy-cifar-10-batches-bin/data_batch_1.bin")
	message(STATUS "Compiling noise_dataset program and noisng of the dataset")
	execute_process(COMMAND ${CMAKE_SOURCE_DIR}/dat/noise.sh
			WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/dat/
			)
else()
	message(STATUS "CIFAR-10 data already noised")
endif()


#Executable for the toolchain
file(GLOB_RECURSE
	image_denoising_files
	./src/*.cpp
	./src/*.h
	./src/cifar/cifar10_reader.hpp
	./params.json
)

include_directories(${GEGELATI_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
add_executable(${PROJECT_NAME} ${image_denoising_files})
target_link_libraries(${PROJECT_NAME} ${GEGELATI_LIBRARIES} ${OpenCV_LIBRARIES})
target_compile_definitions(${PROJECT_NAME} PRIVATE RESULT_DENOISING_CIFAR_10_LOCATION="${RESULT_DENOISING_CIFAR_10_DIR}" CIFAR_10_DATA_LOCATION="${CIFAR_10_DATA_DIR}" NOISY_CIFAR_10_DATA_LOCATION="${NOISY_CIFAR_10_DATA_DIR}" ROOT_DIR="${CMAKE_SOURCE_DIR}")