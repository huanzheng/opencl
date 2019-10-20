all: tiled

tiled: main.c
	gcc -I../lib -L../lib -L/opt/amdgpu-pro/lib/x86_64-linux-gnu main.c  -lcommon -o tiled -lOpenCL
