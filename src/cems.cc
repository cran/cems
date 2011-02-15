#ifndef NULL
#define NULL 0
#endif

#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>
#include <stdio.h>

#include "CEM.h"

extern "C" {
     


SEXP cem_create(SEXP Ry, SEXP Rz, SEXP Rn, SEXP Rmy, SEXP Rmz, SEXP Ri, SEXP
    Rknn, SEXP Rs, SEXP Rf, SEXP Rverbose) {
  
  int verbose = *INTEGER(Rverbose);
  int knn = *INTEGER(Rknn);
  int iter = *INTEGER(Ri);
  int n = *INTEGER(Rn);
  int mz = *INTEGER(Rmz);
  int my = *INTEGER(Rmy);
  double *y = REAL(Ry);
  double *z = REAL(Rz);
  double s = *REAL(Rs);
  double fudge = *REAL(Rf);
  
  if(knn > n){
    knn = n;
  }
  DenseMatrix<double> Z(mz, n, z);
  DenseMatrix<double> Y(my, n, y);
  Z = Linalg<double>::Copy(Z);
  Y = Linalg<double>::Copy(Y);
  

  CEM<double> cem(Y, Z, fudge, knn);
  cem.gradDescent(iter,s, verbose);



  SEXP Zopt;
  PROTECT(Zopt = Rf_allocMatrix(REALSXP, mz, n));
  memcpy( REAL(Zopt), cem.getZ().data(), mz*n*sizeof(double) );

  SEXP sigma;
  PROTECT(sigma = Rf_allocVector(REALSXP, 1));
  double *sigmap = REAL(sigma);
  sigmap[0] = cem.getSigmaX();
  
  
  SEXP list;
  PROTECT( list = Rf_allocVector(VECSXP, 2));
  SET_VECTOR_ELT(list, 0, Zopt);
  SET_VECTOR_ELT(list, 1, sigma);
  
  UNPROTECT(3);
   
  cem.cleanup();

  return list;  
}



SEXP cem_optimize(SEXP Ry, SEXP Rz, SEXP Rn, SEXP Rmy, SEXP Rmz, SEXP Rknn, SEXP
    Rsigma, SEXP Ri, SEXP Rs, SEXP Rverbose) {
    
  int verbose = *INTEGER(Rverbose); 
  int knn = *INTEGER(Rknn);
  int n = *INTEGER(Rn);
  int mz = *INTEGER(Rmz);
  int my = *INTEGER(Rmy);
  double *y = REAL(Ry);
  double *z = REAL(Rz);
  double sigma = *REAL(Rsigma);
  int iter = *INTEGER(Ri);
  double s = *REAL(Rs);

  DenseMatrix<double> Z(mz, n, z);
  DenseMatrix<double> Y(my, n, y);
  Z = Linalg<double>::Copy(Z);
  Y = Linalg<double>::Copy(Y);
  CEM<double> cem(Y, Z, knn, sigma);
  cem.gradDescent(iter,s, verbose);


  SEXP Zopt;
  PROTECT(Zopt = Rf_allocMatrix(REALSXP, mz, n));
  memcpy( REAL(Zopt), cem.getZ().data(), mz*n*sizeof(double) );
  UNPROTECT(1);
   
  cem.cleanup();

  return Zopt;  
}




SEXP cem_parametrize(SEXP Rdata, SEXP Rnd, SEXP Ry, SEXP Rz, SEXP Rn, SEXP Rmy,
		SEXP Rmz, SEXP Rknn, SEXP Rsigma) {
   
  int knn = *INTEGER(Rknn);
  int n = *INTEGER(Rn);
  int nd = *INTEGER(Rnd);
  int mz = *INTEGER(Rmz);
  int my = *INTEGER(Rmy);
  double *y = REAL(Ry);
  double *data = REAL(Rdata);
  double *z = REAL(Rz);
  double sigma = *REAL(Rsigma);
 
  
  DenseMatrix<double> Z(mz, n, z);
  DenseMatrix<double> Y(my, n, y);
  Z = Linalg<double>::Copy(Z);
  Y = Linalg<double>::Copy(Y);
  DenseMatrix<double> Ynew(my, nd, data);
  CEM<double> cem(Y, Z, knn, sigma);

  DenseMatrix<double> Xt = cem.parametrize(Ynew);
  
  SEXP Xnew;
  PROTECT(Xnew = Rf_allocMatrix(REALSXP, mz, nd));
  memcpy( REAL(Xnew), Xt.data(), mz*nd*sizeof(double) );
  UNPROTECT(1);
  
  Xt.deallocate();
  cem.cleanup();
  
  return Xnew;  
}




SEXP cem_reconstruct(SEXP Rdata, SEXP Rnd, SEXP Ry, SEXP Rz, SEXP Rn, SEXP Rmy,
    SEXP Rmz, SEXP Rknn, SEXP Rsigma) {
   
  int knn = *INTEGER(Rknn);
  int n = *INTEGER(Rn);
  int nd = *INTEGER(Rnd);
  int mz = *INTEGER(Rmz);
  int my = *INTEGER(Rmy);
  double *y = REAL(Ry);
  double *data = REAL(Rdata);
  double *z = REAL(Rz);
  double sigma = *REAL(Rsigma);
  
  
  DenseMatrix<double> Z(mz, n, z);
  Z = Linalg<double>::Copy(Z);
  DenseMatrix<double> Y(my, n, y);
  Y = Linalg<double>::Copy(Y);
  DenseMatrix<double> Xnew(mz, nd, data);
  CEM<double> cem(Y, Z, knn, sigma);


  DenseMatrix<double> Yt = cem.reconstruct(Xnew);

  SEXP Ynew;
  PROTECT(Ynew = Rf_allocMatrix(REALSXP, my, nd));
  memcpy( REAL(Ynew), Yt.data(), my*nd*sizeof(double) );
  UNPROTECT(1);
  
  cem.cleanup();
  Yt.deallocate();
  
  return Ynew;  
}



}//end extern C
