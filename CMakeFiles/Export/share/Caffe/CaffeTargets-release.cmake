#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "caffe" for configuration "Release"
set_property(TARGET caffe APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(caffe PROPERTIES
  IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE "proto;proto;/usr/lib64/libboost_system-mt.so;/usr/lib64/libboost_thread-mt.so;/usr/lib64/libboost_filesystem-mt.so;-lpthread;/usr/lib64/libglog.so;/usr/lib64/libgflags.so;/usr/lib64/libprotobuf.so;-lpthread;/usr/lib64/libhdf5_hl.so;/usr/lib64/libhdf5.so;/usr/lib64/libhdf5_hl.so;/usr/lib64/libhdf5.so;/usr/lib64/liblmdb.so;/usr/lib64/libleveldb.so;/usr/local/lib/libsnappy.so;/usr/local/cuda-8.0/lib64/libcudart.so;/usr/local/cuda-8.0/lib64/libcurand.so;/usr/local/cuda-8.0/lib64/libcublas.so;/usr/local/cuda-8.0/lib64/libcudnn.so;opencv_core;opencv_highgui;opencv_imgproc;opencv_imgcodecs;/opt/OpenBLAS/lib/libopenblas.so"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libcaffe.so.1.0.0-rc4"
  IMPORTED_SONAME_RELEASE "libcaffe.so.1.0.0-rc4"
  )

list(APPEND _IMPORT_CHECK_TARGETS caffe )
list(APPEND _IMPORT_CHECK_FILES_FOR_caffe "${_IMPORT_PREFIX}/lib/libcaffe.so.1.0.0-rc4" )

# Import target "proto" for configuration "Release"
set_property(TARGET proto APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(proto PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libproto.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS proto )
list(APPEND _IMPORT_CHECK_FILES_FOR_proto "${_IMPORT_PREFIX}/lib/libproto.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
