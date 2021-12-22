import numpy as np
import pycuda.autoinit
import pycuda.driver as cu
import pycuda.gpuarray as gpu
import pycuda.compiler as nvcc
import matplotlib.image as img
from pycuda.elementwise import ElementwiseKernel
import matplotlib
from PIL import Image, ImageDraw
import copy
import scipy
import time

pycuda_seaming = \
"""
__global__ void convolution(int* red, int* green, int* blue, int* filter, int* energy_map, int height, int width){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int index_pixel = row * width + col;
    int size = height * width;
    
    if(col < width - 1 && row < height - 1 && col > 0 && row > 0){
        int r = (red[index_pixel] * filter[4]) + (red[index_pixel - 1] * filter[3]) + (red[index_pixel + 1] * filter[5]) + (red[index_pixel - width] * filter[1]) + (red[index_pixel + width] * filter[7]) + (red[index_pixel - width - 1] * filter[0]) + (red[index_pixel - width + 1] * filter[2]) + (red[index_pixel + width - 1] * filter[6]) + (red[index_pixel + width + 1] * filter[8]);
        int g = (green[index_pixel] * filter[4]) + (green[index_pixel - 1] * filter[3]) + (green[index_pixel + 1] * filter[5]) + (green[index_pixel - width] * filter[1]) + (green[index_pixel + width] * filter[7]) + (green[index_pixel - width - 1] * filter[0]) + (green[index_pixel - width + 1] * filter[2]) + (green[index_pixel + width - 1] * filter[6]) + (green[index_pixel + width + 1] * filter[8]);
        int b = (blue[index_pixel] * filter[4]) + (blue[index_pixel - 1] * filter[3]) + (blue[index_pixel + 1] * filter[5]) + (blue[index_pixel - width] * filter[1]) + (blue[index_pixel + width] * filter[7]) + (blue[index_pixel - width - 1] * filter[0]) + (blue[index_pixel - width + 1] * filter[2]) + (blue[index_pixel + width - 1] * filter[6]) + (blue[index_pixel + width + 1] * filter[8]);
    
        energy_map[index_pixel] = r;
        energy_map[size+index_pixel] = g;
        energy_map[size*2+index_pixel] = b;
    }
}

__global__ void suma(int* x, int* y, int height, int width){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int index_pixel = row * width + col;
    int new_index_pixel = (row-1)*(width-2) + (col-1);
    int size = height * width;
    if(col<width-1 && row<height-1 && col>0 && row>0){
        y[new_index_pixel]=x[index_pixel]+x[size+index_pixel]+x[2*size + index_pixel];
    }
}

__global__ void find_min(int* energy_map, int height, int width, int* backtrack){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int index_pixel = row * width + col;
    int size = height * width;
    int val = 9999999;
    int idx;
    if(col<width && row<height-1){
        if(col == 0){
            for(int i = 0; i < 2; i = i + 1){
                if(val > energy_map[index_pixel + width + i]){
                    idx = i;
                    val = energy_map[index_pixel + i +  width];
                }
            }
            backtrack[index_pixel] = index_pixel+width+idx;
        }
        else if(col==width-1){
            for(int i = 0; i < 2; i = i + 1){
                if(val > energy_map[index_pixel + width + i - 1]){
                    idx = i;
                    val = energy_map[index_pixel + i +  width - 1];
                }
            }
            backtrack[index_pixel] =  index_pixel+width+idx-1;
        }
        else{
            for(int i = 0; i < 3; i = i + 1){
                if(val > energy_map[index_pixel + width + i - 1]){
                    idx = i;
                    val = energy_map[index_pixel + i +  width - 1];
                }
            }
            backtrack[index_pixel] =  index_pixel+width+idx-1;
        }
    }
}

__global__ void get_sum_map(int* sum_map, int* next_sum_map, int* offset_map, int* next_offset,int height, int width, int n){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int index_pixel = row * width + col;
    int size=height*width;
    if(col < width && row < height && row%(2*n)==0 && offset_map[index_pixel]>=0 && offset_map[index_pixel]<=width*height){
        next_sum_map[index_pixel] = sum_map[index_pixel]+sum_map[offset_map[index_pixel]];
        __syncthreads();
        next_offset[index_pixel]=offset_map[offset_map[index_pixel]];
        __syncthreads();
    }
}

__global__ void extract_seam_path(int* backtrack, int index,int height, int width, int* path){
    int tid=blockIdx.x * blockDim.x + threadIdx.x;
    if(tid==0){
        path[0]=index;
        path[1]=index;
        for(int i=1;i<height-1;++i){
            path[i+1]=backtrack[index]%(width-2);
            index=backtrack[index];
        }
        path[height-1]=path[height-2];
    }
}

__global__ void remove_seam(int* red, int* green, int* blue, int* new_red, int* new_green, int* new_blue,int* path, int height, int width){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int new_index_pixel = row *(width-1) + col;
    int index_pixel = row*width+col;
    if(col<width-1 && row<height){
        if(col>=path[row]){
            new_red[new_index_pixel]=red[index_pixel+1];
            new_green[new_index_pixel]=green[index_pixel+1];
            new_blue[new_index_pixel]=blue[index_pixel+1];
        }
        else{
            new_red[new_index_pixel] = red[index_pixel];
            new_green[new_index_pixel] = green[index_pixel];
            new_blue[new_index_pixel] = blue[index_pixel];
        }
    }
}
"""

if __name__ == '__main__':

    start = time.time()
    # Kernels de CUDA
    module = nvcc.SourceModule(pycuda_seaming)
    convolution_kernel = module.get_function("convolution")
    suma=module.get_function("suma")
    find_min = module.get_function("find_min")
    get_sum_map = module.get_function("get_sum_map")
    extract_seam_path = module.get_function("extract_seam_path")
    remove_seam = module.get_function("remove_seam")
    
    # Leer imagen
    h_img_in = np.array(img.imread('img.jpg'), dtype=np.int32)
    height, width, channels = np.int32(h_img_in.shape)

    block = (32, 32, 1)
    grid = (int(np.ceil(width/block[0])),
            int(np.ceil(height/block[1])))

    #Partiendo la imagen en sus canales y comunicandolos (esto simplificara el algoritmo)
    
    red_cpu = np.array(copy.deepcopy(h_img_in[:,:,0]), dtype=np.int32)
    green_cpu = np.array(copy.deepcopy(h_img_in[:,:,1]), dtype=np.int32)
    blue_cpu = np.array(copy.deepcopy(h_img_in[:,:,2]), dtype=np.int32)
    red_gpu = gpu.to_gpu(red_cpu)
    green_gpu = gpu.to_gpu(green_cpu)
    blue_gpu = gpu.to_gpu(blue_cpu)
    
    filterdu_cpu = np.array([
        [1.0, 2.0, 1.0],
        [0.0, 0.0, 0.0],
        [-1.0, -2.0, -1.0],
    ], dtype = np.int32)
        
    filterdv_cpu = np.array([
        [1.0, 0.0, -1.0],
        [2.0, 0.0, -2.0],
        [1.0, 0.0, -1.0]
    ], dtype = np.int32)

    filterdu_gpu = gpu.to_gpu(filterdu_cpu)
    filterdv_gpu = gpu.to_gpu(filterdv_cpu)

    # Transferir al GPU
    d_img_in = gpu.to_gpu(h_img_in)
    seams=600
    for i in range(seams):
        #print(i)
        # Aplicar convolucion para determinar las energias
        
        width2=np.int32(width-2)
        height2=np.int32(height-2)
        energy_mapdu = gpu.empty((3,height, width), dtype=np.int32)
        energy_mapdv = gpu.empty((3,height, width), dtype=np.int32)
        convolution_kernel(red_gpu, green_gpu, blue_gpu, filterdu_gpu, energy_mapdu, height, width, block = block, grid = grid)
        convolution_kernel(red_gpu, green_gpu, blue_gpu, filterdv_gpu, energy_mapdv, height, width, block = block, grid = grid)
        
        energy_map = energy_mapdu.__abs__() + energy_mapdv.__abs__()
        
        # Los bordes se eliminan con la convolucion, asi que el energy map es de estas dimensiones
        
        energy_map2 = gpu.empty((height2, width2), dtype=np.int32)
        suma(energy_map, energy_map2, height, width, block = block, grid = grid)
        
        # Guardar los posibles caminos en backtrack que podriamos seguir con un enfoque greedy (nos vamos por la menor energia cada vez)
        
        backtrack = gpu.empty((height2, width2), dtype = np.int32)
        
        find_min(energy_map2, height2, width2, backtrack, block = block, grid = grid)
        offset_map = backtrack.copy()
        sum_map = energy_map2.copy()
        
        # Calcular el sum_top de las energias cumulativas de todos los caminos por reduccion
        
        n=np.int32(1)
        while n<height2/2:
            next_sum_map = gpu.empty_like(sum_map)
            next_offset = gpu.empty_like(offset_map)
            get_sum_map(sum_map, next_sum_map,offset_map, next_offset,height2, width2, n,block = block, grid = grid)
            #print(next_offset)
            sum_map = next_sum_map.copy()
            offset_map = next_offset.copy()
            n=n*2

        # Indice minimo de sum_top para iniciar el corte, este sera nuestro seam optimo

        indexs=gpu.arange(0,width2, dtype=np.int32)
        sum_top = gpu.take(sum_map, indexs)
        sum_cpu = np.zeros(width2, dtype=np.int32)
        cu.memcpy_dtoh(sum_cpu, sum_top.gpudata)
        indice=np.argmin(sum_cpu)

        # Extrayendo y removiendo el seam optimo encontrado
        
        path = gpu.empty(height, dtype=np.int32)
        extract_seam_path(backtrack, np.int32(indice), height, width, path, block = (1,1,1), grid = (1,1))
        new_red_gpu=gpu.empty((height,width-1), dtype=np.int32)
        new_green_gpu=gpu.empty((height,width-1), dtype=np.int32)
        new_blue_gpu=gpu.empty((height,width-1), dtype=np.int32)
        remove_seam(red_gpu, green_gpu, blue_gpu, new_red_gpu, new_green_gpu, new_blue_gpu, path, height, width, block = block, grid = grid)
        
        # El width se decrementa en uno tras haber removido el seam, guardando los resultados para volver a iterar
        width=np.int32(width-1)
        red_gpu, green_gpu, blue_gpu = new_red_gpu, new_green_gpu, new_blue_gpu

    # Comunicando canales
    
    red_cpu=np.zeros((height,width), dtype=np.int32)
    green_cpu=np.zeros((height,width), dtype=np.int32)
    blue_cpu=np.zeros((height,width), dtype=np.int32)
    cu.memcpy_dtoh(red_cpu, red_gpu.gpudata)
    cu.memcpy_dtoh(green_cpu, green_gpu.gpudata)
    cu.memcpy_dtoh(blue_cpu, blue_gpu.gpudata)
    
    # Fusionando canales y guardando imagen
    
    red_cpu=red_cpu.astype(np.uint8)
    green_cpu=green_cpu.astype(np.uint8)
    blue_cpu=blue_cpu.astype(np.uint8)
    rgb=np.array([red_cpu.transpose(),green_cpu.transpose(),blue_cpu.transpose()])
    rgb=rgb.transpose()
    print(rgb.shape)
    image=Image.fromarray(rgb, mode='RGB')
    image.save("img_paralelo.png")
    stop = time.time()
    tiempo = stop - start
    print(tiempo)
