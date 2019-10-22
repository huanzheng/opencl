// Increased the amount of work-per-thread by a factor WPT
//#define WPT 8                        // The amount of work-per-thread, i.e. the thread-coarsening factor
//## 1 devices, running on 0: 'Ellesmere', MAX_WORK_GROUP_SIZE is 256
//Matrix 1024x1024 * 1024x1024 -> 1024x1024 takes 0.015716 seconds to complete on GPU
//## 1 devices, running on 0: 'Ellesmere', MAX_WORK_GROUP_SIZE is 256
//Matrix 2048x2048 * 2048x2048 -> 2048x2048 takes 0.134260 seconds to complete on GPU
//## 1 devices, running on 0: 'Ellesmere', MAX_WORK_GROUP_SIZE is 256
//Matrix 4096x4096 * 4096x4096 -> 4096x4096 takes 1.077868 seconds to complete on GPU
//WPT是8时，workgroupsize 是 16*(16/8)=32; 首先是一个workgroup只有一个wavefront了，并且本身1个wavefront是可以包含64个workitem的，而现在只有32个。所以反而慢了

//#define WPT 4                        // The amount of work-per-thread, i.e. the thread-coarsening factor
//## 1 devices, running on 0: 'Ellesmere', MAX_WORK_GROUP_SIZE is 256
//Matrix 1024x1024 * 1024x1024 -> 1024x1024 takes 0.007695 seconds to complete on GPU
//## 1 devices, running on 0: 'Ellesmere', MAX_WORK_GROUP_SIZE is 256
//Matrix 2048x2048 * 2048x2048 -> 2048x2048 takes 0.058056 seconds to complete on GPU
//## 1 devices, running on 0: 'Ellesmere', MAX_WORK_GROUP_SIZE is 256
//Matrix 4096x4096 * 4096x4096 -> 4096x4096 takes 0.463742 seconds to complete on GPU
//WPT是4时，workgroupsize 是 16*(16/4)=64; 首先是一个workgroup只有一个wavefront了，1个wavefront现在是包含住了64个workitem，比上面那个快了


//#define WPT 2                        // The amount of work-per-thread, i.e. the thread-coarsening factor
//## 1 devices, running on 0: 'Ellesmere', MAX_WORK_GROUP_SIZE is 256
//Matrix 1024x1024 * 1024x1024 -> 1024x1024 takes 0.005606 seconds to complete on GPU
//## 1 devices, running on 0: 'Ellesmere', MAX_WORK_GROUP_SIZE is 256
//Matrix 2048x2048 * 2048x2048 -> 2048x2048 takes 0.033337 seconds to complete on GPU
//## 1 devices, running on 0: 'Ellesmere', MAX_WORK_GROUP_SIZE is 256
//Matrix 4096x4096 * 4096x4096 -> 4096x4096 takes 0.281573 seconds to complete on GPU
//WPT是2时，workgroupsize 是 16*(16/2)=128; workgroup有2个wavefront了，因此可以有memory latency hiding了。并且1个wavefront现在是包含住了64个workitem，比上面那个快了

//#define WPT 1                        // The amount of work-per-thread, i.e. the thread-coarsening factor
//## 1 devices, running on 0: 'Ellesmere', MAX_WORK_GROUP_SIZE is 256
//Matrix 1024x1024 * 1024x1024 -> 1024x1024 takes 0.003852 seconds to complete on GPU
//## 1 devices, running on 0: 'Ellesmere', MAX_WORK_GROUP_SIZE is 256
//Matrix 2048x2048 * 2048x2048 -> 2048x2048 takes 0.016858 seconds to complete on GPU
//## 1 devices, running on 0: 'Ellesmere', MAX_WORK_GROUP_SIZE is 256
//Matrix 4096x4096 * 4096x4096 -> 4096x4096 takes 0.146555 seconds to complete on GPU
//WPT是1时，workgroupsize 是 16*(16/1)=256; workgroup有4个wavefront了，因此可以有memory latency hiding了。并且1个wavefront现在是包含住了64个workitem，比上面那个快了
//WPT是1时，跟单纯titled的那个版本，性能基本一致
//这个case，memory latency hiding比more computation inside work item更加影响性能

#define RTS (TS/WPT)                 // The reduced tile-size in one dimension


__kernel void tiled_morework(const int M, const int N, const int K,
                      const __global float* A,
                      const __global float* B,
                      __global float* C) {
    
    // Thread identifiers
    const int row = get_local_id(0); // Local row ID (max: TS)
    const int col = get_local_id(1); // Local col ID (max: TS/WPT == RTS)
    const int globalRow = TS*get_group_id(0) + row; // Row ID of C (0..M)
    const int globalCol = TS*get_group_id(1) + col; // Col ID of C (0..N)
 
    // Local memory to fit a tile of TS*TS elements of A and B
    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];
 
    // Initialise the accumulation registers
    float acc[WPT];
    for (int w=0; w<WPT; w++) {
        acc[w] = 0.0f;
    }
    
    // Loop over all tiles
    const int numTiles = K/TS;
    for (int t=0; t<numTiles; t++) {
 
        // Load one tile of A and B into local memory
        for (int w=0; w<WPT; w++) {
            const int tiledRow = TS*t + row;
            const int tiledCol = TS*t + col;
            Asub[col + w*RTS][row] = A[(tiledCol + w*RTS)*M + globalRow];
            Bsub[col + w*RTS][row] = B[(globalCol + w*RTS)*K + tiledRow];
        }
        
        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);
 
        // Perform the computation for a single tile
        for (int k=0; k<TS; k++) {
            for (int w=0; w<WPT; w++) {
                acc[w] += Asub[k][row] * Bsub[col + w*RTS][k];
            }
        }
 
        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
 
    // Store the final results in C
    for (int w=0; w<WPT; w++) {
        C[(globalCol + w*RTS)*M + globalRow] = acc[w];
    }
}
