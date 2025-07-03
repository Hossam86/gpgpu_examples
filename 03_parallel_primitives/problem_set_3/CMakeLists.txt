find_package(OpenCV REQUIRED)

# INCLUDE HEADERS
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories("/usr/local/cuda/include")

# collect source files
file( GLOB  hdr *.hpp *.h )
file( GLOB  cu  *.cu)
SET (HW3_files main.cpp loadSaveImage.cpp reference_calc.cpp compare.cpp)

add_executable(HW3 ${HW3_files} ${hdr} ${cu})
set_target_properties(HW3 PROPERTIES CUDA_ARCHITECTURES 86)
target_link_libraries(HW3 ${OpenCV_LIBS})

