#include <stdint.h>
#include<cufft.h>
#include<cuda.h>

#define IDX(i,j,ld) (((i)*(ld))+(j))


__global__ void zeroPadKernel(float *dx, int m, int n, float *dy, int p, int q, int s, int t){
  int i = blockIdx.y*blockDim.y + threadIdx.y;
  int j = blockIdx.x*blockDim.x + threadIdx.x;

  if ( i<m && j<n ){
    if( (s-1)<i && i<(p+s) && (t-1)<j && j<(q+t) ){
      dx[IDX( i, j, n )] = dy[IDX( i-s, j-t, q )];
    }
    else{
      dx[IDX( i, j, n )] = 0.0f;
    }
  }
}


__global__ void zeroPadComplexKernel(cufftComplex *dx, int m, int n, float *dy, int p, int q, int s, int t){
  int i = blockIdx.y*blockDim.y + threadIdx.y;
  int j = blockIdx.x*blockDim.x + threadIdx.x;

  if ( i<m && j<n ){
    if( (s-1)<i && i<(p+s) && (t-1)<j && j<(q+t) ){
      dx[IDX( i, j, n )].x = dy[IDX( i-s, j-t, q )];
      dx[IDX( i, j, n )].y = 0.0f;
    }
    else{
    dx[IDX( i, j, n )].x = 0.0f;
    dx[IDX( i, j, n )].y = 0.0f;
    }
  }
}


__global__ void crop_Kernel(float *dx, int m, int n, float *dy, int p, int q, int s, int t){
  int i = blockIdx.y*blockDim.y + threadIdx.y;
  int j = blockIdx.x*blockDim.x + threadIdx.x;

  if ( (s-1)<i && i<m+s ){
    if( (t-1)<j && j<n+t ){
      dx[IDX( i-s, j-t, n )] = dy[IDX( i, j, q )];
    }
  }
}


__global__ void crop_ComplexKernel(float *dx, int m, int n, cufftComplex *dy, int p, int q, int s, int t){
  int i = blockIdx.y*blockDim.y + threadIdx.y;
  int j = blockIdx.x*blockDim.x + threadIdx.x;

  if ( (s-1)<i && i<m+s ){
    if( (t-1)<j && j<n+t ){
      dx[IDX( i-s, j-t, n )] = dy[IDX( i, j, q )].x;
    }
  }
}
