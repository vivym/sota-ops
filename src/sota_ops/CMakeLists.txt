set(MODULE_NAME sota_ops)

add_library(${MODULE_NAME} SHARED)

file(GLOB_RECURSE HEADER_LIST CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/include/*.h")
file(GLOB_RECURSE CPP_SRC CONFIGURE_DEPENDS "*.cpp")
file(GLOB_RECURSE CUDA_SRC CONFIGURE_DEPENDS "*.cu")

target_sources(${MODULE_NAME}
  PRIVATE
    ${CPP_SRC}
    ${CUDA_SRC}
  PUBLIC
    ${HEADER_LIST}
)

target_include_directories(${MODULE_NAME} PRIVATE
  ${TORCH_INCLUDE_DIRS}
  "${PROJECT_SOURCE_DIR}/include"
)

target_link_libraries(${MODULE_NAME}
  ${TORCH_LIBRARIES}
  ${TORCH_PYTHON_LIBRARY}
  CUDA::cublas
  CUDA::cusparse
  MODERNGPU
)

target_compile_options(${MODULE_NAME} PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
)
set_property(TARGET ${MODULE_NAME} PROPERTY CXX_STANDARD 17)

install(TARGETS ${MODULE_NAME} DESTINATION sota_ops)
