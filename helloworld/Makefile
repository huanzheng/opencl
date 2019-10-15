all: helloworld

helloworld: helloworld.c
	gcc -L/opt/amdgpu-pro/lib/x86_64-linux-gnu helloworld.c -lOpenCL -o helloworld

clean:
	rm helloworld
