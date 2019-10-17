all: libcommon.so

libcommon.so: common.c common.h
	gcc -fPIC -L/opt/amdgpu-pro/lib/x86_64-linux-gnu -shared common.c -o libcommon.so -lOpenCL
