#include<iostream>
#include<cuda_runtime.h>

//note that we are using C to store the results
//so no const for that
__global__ void vectorAdd(const float * A, const float * B, float * C, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < N){
        C[idx] = A[idx] + B[idx];
    }
}

int main(){
    const int N = 1024;
    const int size = N * sizeof(int);

    float *h_A = new float[N];
    float *h_B = new float[N];
    float *n_C = new float[N];

    cudaMemcpy(d_A,h_A,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,size,cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = ( N + threadsPerBlock -1) / threadsPerBlock;

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A,d_B,d_C,N);

    cudaMemcpy(h_C,d_C,size,cudaMemcpyDeviceToHost);
    for(int i =N-10;i<N;i++){
        std::cout << "C[" << i << "] = " << h_C[i] << std::endl;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] h_A; 
    delete[] h_B; 
    delete[] h_C; 

    return 0; 
}