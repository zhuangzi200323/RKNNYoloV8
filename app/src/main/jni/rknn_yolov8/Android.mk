LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

CFLAGS := -Werror

LOCAL_C_INCLUDES := \
		$(LOCAL_PATH)/ \
		$(LOCAL_PATH)/../ \

LOCAL_LDLIBS := -L$(SYSROOT)/usr/lib -ldl
LOCAL_LDLIBS += -llog
LOCAL_LDLIBS += -landroid
LOCAL_LDLIBS += -ljnigraphics

LOCAL_SHARED_LIBRARIES += rknnrt-shared

LOCAL_STATIC_LIBRARIES += rga-static
LOCAL_STATIC_LIBRARIES += turbojpeg-static

LOCAL_MODULE    := rknn_yolov8
LOCAL_SRC_FILES :=  \
	utils/file_utils.c \
	utils/image_drawing.c \
	utils/image_utils.c \
	main.cc \
	postprocess.cc \
	rknn_yolov8_jni.cc \
	yolov8.cc \
	yolov8_zero_copy.cc \

include $(BUILD_SHARED_LIBRARY)