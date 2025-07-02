find_package(OpenCV REQUIRED)

# INCLUDE HEADERS
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories("/usr/local/cuda/include")

# collect source files
file( GLOB  hdr *.hpp *.h )
file( GLOB  cu  *.cu)
SET (HW2_files main.cpp reference_calc.cpp compare.cpp)

add_executable(HW2 ${HW2_files} ${hdr} ${cu})
set_target_properties(HW2 PROPERTIES CUDA_ARCHITECTURES 86)
target_link_libraries(HW2 ${OpenCV_LIBS})

