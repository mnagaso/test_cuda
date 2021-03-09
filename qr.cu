#include "cuda_runtime.h"

#include<iostream>
#include<stdlib.h>
#include<stdio.h>
#include<assert.h>
#include<vector>
#include "cublas_v2.h"
#include "Eigen/Core"
#define IDX2C(i,j,ld) (((j)*(ld))+(i))
#define cuda_error_check(x) do { if((x) != 0) { printf("cuda error in %s at line: %d\n", __FILE__, __LINE__); exit(1);} } while(false)

const int nbatch=100;
const int nrows = 3;
const int ncols = 3;
const int ltau  = 3;

int main(){
   // init n eigen matrices
    Eigen::MatrixXd a_mat(nrows, ncols);
    //a_mat << 12, -51, 4, \
    //        6, 167, -68, \
    //        -4, 24, -41;
    a_mat << 2.5,1.1,0.3, \
             2.2,1.9,0.4, \
             1.8,0.1,0.3;


    std::vector<Eigen::MatrixXd> e_mats;
    for (int i = 0; i < nbatch; i++){
        e_mats.push_back(a_mat);
    }
 
    // cublas init
    cublasHandle_t handle;
    cuda_error_check(cublasCreate(&handle));

    // array on host
    // use Eigen::Matrix::data() method to access the pointer array inside of Eigen obj
    // arrays on device
    double*  devPtrMat[nbatch];  
    double*  devPtrMat_TAU[nbatch];
    double** d_devPtrMat;
    double** d_devPtrMat_TAU;
    int info;


    for (int i = 0; i < nbatch; i++){
      cuda_error_check(cudaMalloc((void**)&devPtrMat[i], nrows*ncols*sizeof(double)));
      cuda_error_check(cudaMalloc((void**)&devPtrMat_TAU[i], ltau*sizeof(double)));
    }

    cuda_error_check(cudaMalloc((void**)&d_devPtrMat,     sizeof(double*)*nbatch));
    cuda_error_check(cudaMalloc((void**)&d_devPtrMat_TAU, sizeof(double*)*nbatch));
    
    // download memory
    for (int i = 0; i < nbatch; i++){
        double* mtmp = e_mats[i].data();
        cuda_error_check(cublasSetMatrix (nrows, ncols, sizeof(double), mtmp, nrows, devPtrMat[i], nrows));
    }

    cuda_error_check(cudaMemcpy(d_devPtrMat,     devPtrMat,     sizeof(devPtrMat),     cudaMemcpyHostToDevice));
    cuda_error_check(cudaMemcpy(d_devPtrMat_TAU, devPtrMat_TAU, sizeof(devPtrMat_TAU), cudaMemcpyHostToDevice));

    // do qr
    cuda_error_check(cudaDeviceSynchronize());
    cuda_error_check(cublasDgeqrfBatched(handle, nrows, ncols, d_devPtrMat, ltau, d_devPtrMat_TAU, &info, nbatch));
    cuda_error_check(cudaDeviceSynchronize());

    // upload memory
    double* tmp;
    double* tmp2;
    tmp =(double*)malloc(nrows*ncols*sizeof(double));
    tmp2=(double*)malloc(ltau*sizeof(double));


    for (int i = 0; i < nbatch; i++){
        cuda_error_check(cudaMemcpy(tmp, devPtrMat[i], nrows*ncols*sizeof(double), cudaMemcpyDeviceToHost));
        Eigen::Map<Eigen::MatrixXd> tmpm(tmp,nrows,ncols);            
        e_mats[i] = tmpm;

        cuda_error_check(cudaMemcpy(tmp2,devPtrMat_TAU[i],ltau*sizeof(double), cudaMemcpyDeviceToHost));
    }
    for (int i = 0; i < nbatch; i++){
        if (i == 0){
            std::cout << "i- " << i << std::endl;
            std::cout << e_mats[i] << std::endl;
            std::cout << "tau: ";
            for (int j = 0; j < ltau; j++) std::cout << tmp2[j] << "   ";
        }
    }

    free(tmp);
    free(tmp2);

    if (d_devPtrMat)     cudaFree(d_devPtrMat);
    if (d_devPtrMat_TAU) cudaFree(d_devPtrMat_TAU);
    if (handle)          cublasDestroy(handle);
        

    return 0;
}
