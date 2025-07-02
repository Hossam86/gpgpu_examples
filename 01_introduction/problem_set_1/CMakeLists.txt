find_package(OpenCV REQUIRED)

# INCLUDE HEADERS
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories("/usr/local/cuda/include")

file( GLOB  hdr *.hpp *.h )
file( GLOB  cu  *.cu)
SET (HW1_files main.cpp reference_calc.cpp compare.cpp)

add_executable(HW1 ${HW1_files} ${hdr} ${cu})
set_target_properties(HW1 PROPERTIES CUDA_ARCHITECTURES 86)
target_link_libraries(HW1 ${OpenCV_LIBS})