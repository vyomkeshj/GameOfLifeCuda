#include <iostream>
#include <cstdio>
#define CUDACHECK(err) { cuda_check((err), __FILE__, __LINE__); }

inline void cuda_check(cudaError_t error_code, const char *file, int line)
{
    if (error_code != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error %d: %s. In '%s' on line %d\n", error_code, cudaGetErrorString(error_code), file, line);
        exit(error_code);
    }
}

#define CUDAMEASURE(command) do { cudaEvent_t b,e; \
    CUDACHECK(cudaEventCreate(&b)); CUDACHECK(cudaEventCreate(&e)); \
    CUDACHECK(cudaEventRecord(b)); command ; CUDACHECK(cudaEventRecord(e)); \
    CUDACHECK(cudaEventSynchronize(e)); \
    float time; CUDACHECK(cudaEventElapsedTime(&time, b, e)); \
    printf("Execution time: %f ms\n", time); \
    CUDACHECK(cudaEventDestroy(b)); CUDACHECK(cudaEventDestroy(e)); } while(false)


// Draws a glider on the grid skipping col_offst number of cols from left and ...
void create_glider(int col_offst, int row_offst, int grid_width, int grid_height, int *grid) {
    int glider_coords[3][3] = {{0, 0, 1}, {1, 0, 1}, {0, 1, 1}};
    for(int x_pos = 0; x_pos < 3; x_pos++) {
        for(int y_pos = 0; y_pos < 3; y_pos++) {
            if(col_offst<=grid_width-3 && row_offst<=grid_height-3) {
                grid[(x_pos+row_offst)*grid_width + (y_pos+col_offst)] = glider_coords[x_pos][y_pos];
            } else {
                std::cout<<"glider out of grid"<<std::endl;
            }
        }
    }
}

void print_grid(int grid_width, int grid_height, int *grid) {
    for (int i = 0; i< grid_width; i++) {
        for (int j = 0; j< grid_height; j++) {
            if (grid[i*grid_width + j] == 0) {
                std::cout<<" - ";
            } else {
                std::cout<<" X ";
            }
        }
        std::cout<<std::endl;
    }
}

__global__ void run_life_one_step(const int* world_matrix, int grid_width,
                                  int grid_height, int* result_world_matrix) {
    int worldSize = grid_width * grid_height;

    for (int cellId = (blockIdx.x * blockDim.x) + threadIdx.x; cellId < worldSize; cellId += blockDim.x * gridDim.x) {
        int x_pos = cellId % grid_width;
        int y_pos = cellId - x_pos;

        int upper_cell_pos = (y_pos + worldSize - grid_width) % worldSize;
        int lower_cell_pos = (y_pos + grid_width) % worldSize;
        int left_cell_pos = (x_pos + grid_width - 1) % grid_width;
        int right_cell_pos = (x_pos + 1) % grid_width;

        int neighbour_sum = world_matrix[left_cell_pos + upper_cell_pos] + world_matrix[x_pos + upper_cell_pos]
                          + world_matrix[right_cell_pos + upper_cell_pos] + world_matrix[left_cell_pos + y_pos] + world_matrix[right_cell_pos + y_pos]
                          + world_matrix[left_cell_pos + lower_cell_pos] + world_matrix[x_pos + lower_cell_pos] + world_matrix[right_cell_pos + lower_cell_pos];

        result_world_matrix[x_pos + y_pos] = neighbour_sum == 3 || (neighbour_sum == 2 && world_matrix[x_pos + y_pos]) ? 1 : 0;
    }
}

void run_life(int*& d_world_matrix, int*& d_world_matrixBuffer, size_t grid_width,
              size_t grid_height, size_t iterationsCount, short threadsCount) {

    int reqBlocksCount = (grid_width * grid_height) / threadsCount;
    int blocksCount = (int) std::min(65535, reqBlocksCount);

    for (int i = 0; i < iterationsCount; ++i) {
        run_life_one_step<<<blocksCount, threadsCount>>>(d_world_matrix, grid_width, grid_height, d_world_matrixBuffer);
        std::swap(d_world_matrix, d_world_matrixBuffer);
    }
}

int main(int argc, char **argv) {
    int grid_width = 30;
    int grid_height = 30;
    int generations = 1000;
    int tpb = 32;

    for (int i = 0; i < argc; ++i) {
        if (!strcmp("-n", argv[i])) {
            grid_width = atoi(argv[i + 1]);
        } else if (!strcmp("-m", argv[i])) {
            grid_height = atoi(argv[i + 1]);
        } else if (!strcmp("-max", argv[i])) {
            generations = atoi(argv[i + 1]);
        } else if (!strcmp("-tpb", argv[i])) {
            tpb = atoi(argv[i + 1]);
        }
    }
    size_t size = sizeof(float)*grid_height*grid_width;

    int* life_data_host;
    int* life_data_buffer_host;

    CUDACHECK(cudaMallocHost(&life_data_host, size));
    CUDACHECK(cudaMallocHost(&life_data_buffer_host, size));

    create_glider(3, 4, grid_width, grid_height, life_data_host);
    create_glider(0, 12, grid_width, grid_height, life_data_host);
    print_grid(grid_width, grid_height, life_data_host);

    int* life_data_device;
    int* life_data_buffer_device;

    CUDACHECK(cudaMalloc(&life_data_device, size));
    CUDACHECK(cudaMalloc(&life_data_buffer_device, size));


    CUDACHECK(cudaMemcpy(life_data_device, life_data_host, size, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(life_data_buffer_device, life_data_buffer_host, size, cudaMemcpyHostToDevice));

    CUDAMEASURE((run_life(life_data_device, life_data_buffer_device, grid_width, grid_height, generations, tpb)));
    CUDACHECK(cudaMemcpy(life_data_host, life_data_device, size, cudaMemcpyDeviceToHost));

    CUDACHECK(cudaDeviceSynchronize());

    print_grid(grid_width, grid_height, life_data_host);

    return 0;
}

