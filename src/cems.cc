#ifndef NULL
#define NULL 0
#endif

#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>
#include <stdio.h>

#include "CEM.h"
#include "CEMRegression.h"

extern "C" {
  




//CEM methods

SEXP cem_create(SEXP Ry, SEXP Rz, SEXP Rn, SEXP Rmy, SEXP Rmz,
    SEXP RknnY, SEXP RknnX, SEXP RsigmaY, SEXP RsigmaX, SEXP Ri, 
    SEXP RsZ, SEXP RsBW, SEXP Rverbose, SEXP Rtype, SEXP RsigmaAsFactor, SEXP
    RoptimalSigmaX) {
  
  int verbose = *INTEGER(Rverbose);
  int knnY = *INTEGER(RknnY);
  int knnX = *INTEGER(RknnX);
  int iter = *INTEGER(Ri);
  int n = *INTEGER(Rn);
  int mz = *INTEGER(Rmz);
  int my = *INTEGER(Rmy);
  int type = *INTEGER(Rtype);
  double *y = REAL(Ry);
  double *z = REAL(Rz);
  double sZ = *REAL(RsZ);
  double sBW = *REAL(RsBW);
  double sigmaY = *REAL(RsigmaY);
  double sigmaX = *REAL(RsigmaX);
  bool sigmaAsFactor = *INTEGER(RsigmaAsFactor);
  bool optimalSigmaX = *INTEGER(RoptimalSigmaX);
  
  DenseMatrix<double> Z(mz, n, z);
  DenseMatrix<double> Y(my, n, y);
  Z = Linalg<double>::Copy(Z);
  Y = Linalg<double>::Copy(Y);
  

  CEM<double> cem(Y, Z, knnY, knnX, sigmaY, sigmaX, sigmaAsFactor);
  cem.gradDescent(iter, sZ, sBW, verbose, type, optimalSigmaX);
  
  

    
  SEXP list;
  PROTECT( list = Rf_allocVector(VECSXP, 3));

  SEXP Zopt;
  PROTECT(Zopt = Rf_allocMatrix(REALSXP, mz, n));
  memcpy( REAL(Zopt), cem.getZ().data(), mz*n*sizeof(double) );
  SET_VECTOR_ELT(list, 0, Zopt);

  SEXP sigmaXn;
  PROTECT(sigmaXn = Rf_allocVector(REALSXP, 1));
  double *sigmaXp = REAL(sigmaXn);
  sigmaXp[0] = cem.getSigmaX();
  SET_VECTOR_ELT(list, 1, sigmaXn);

  SEXP sigmaYn;
  PROTECT(sigmaYn = Rf_allocVector(REALSXP, 1));
  double *sigmaYp = REAL(sigmaYn);
  sigmaYp[0] = cem.getSigmaY();
  SET_VECTOR_ELT(list, 2, sigmaYn);

  
  UNPROTECT(4);
   
  cem.cleanup();

  return list;  
}



SEXP cem_optimize(SEXP Ry, SEXP Rz, SEXP Rn, SEXP Rmy, SEXP Rmz, SEXP RknnY,
    SEXP RknnX, SEXP RsigmaY, SEXP RsigmaX, SEXP Ri, SEXP RsZ, SEXP RsBW, SEXP
    Rverbose, SEXP Rtype,  SEXP RoptimalSigmaX) {
    
  int verbose = *INTEGER(Rverbose); 
  int knnY = *INTEGER(RknnY);
  int knnX = *INTEGER(RknnX);
  int n = *INTEGER(Rn);
  int mz = *INTEGER(Rmz);
  int my = *INTEGER(Rmy);
  int type = *INTEGER(Rtype);
  double *y = REAL(Ry);
  double *z = REAL(Rz);
  double sigmaY = *REAL(RsigmaY);
  double sigmaX = *REAL(RsigmaX);
  int iter = *INTEGER(Ri);
  double sZ = *REAL(RsZ);
  double sBW = *REAL(RsBW);
  bool optimalSigmaX = *INTEGER(RoptimalSigmaX);


  DenseMatrix<double> Z(mz, n, z);
  DenseMatrix<double> Y(my, n, y);
  Z = Linalg<double>::Copy(Z);
  Y = Linalg<double>::Copy(Y);

  CEM<double> cem(Y, Z, knnY, knnX, sigmaY, sigmaX, false);
  cem.gradDescent(iter, sZ, sBW, verbose, type, optimalSigmaX);


  


    
  SEXP list;
  PROTECT( list = Rf_allocVector(VECSXP, 3));

  SEXP Zopt;
  PROTECT(Zopt = Rf_allocMatrix(REALSXP, mz, n));
  memcpy( REAL(Zopt), cem.getZ().data(), mz*n*sizeof(double) );
  SET_VECTOR_ELT(list, 0, Zopt);

  SEXP sigmaxn;
  PROTECT(sigmaxn = Rf_allocVector(REALSXP, 1));
  double *sigmaxp = REAL(sigmaxn);
  sigmaxp[0] = cem.getSigmaX();
  SET_VECTOR_ELT(list, 1, sigmaxn);

  SEXP sigmayn;
  PROTECT(sigmayn = Rf_allocVector(REALSXP, 1));
  double *sigmayp = REAL(sigmayn);
  sigmayp[0] = cem.getSigmaY();
  SET_VECTOR_ELT(list, 2, sigmayn);
  
  UNPROTECT(4);


  cem.cleanup();

  return list;  
}







SEXP cem_parametrize(SEXP Rdata, SEXP Rnd, SEXP Ry, SEXP Rz, SEXP Rn, SEXP Rmy,
		SEXP Rmz, SEXP RknnY, SEXP RknnX, SEXP RsigmaY, SEXP RsigmaX) {

  
  int knnY = *INTEGER(RknnY);
  int knnX = *INTEGER(RknnX);
  int n = *INTEGER(Rn);
  int mz = *INTEGER(Rmz);
  int my = *INTEGER(Rmy);
  double *y = REAL(Ry);
  double *z = REAL(Rz);
  double sigmaY = *REAL(RsigmaY);
  double sigmaX = *REAL(RsigmaX);
  


  
  DenseMatrix<double> Z(mz, n, z);
  DenseMatrix<double> Y(my, n, y);
  Z = Linalg<double>::Copy(Z);
  Y = Linalg<double>::Copy(Y);
     
  CEM<double> cem(Y, Z, knnY, knnX, sigmaY, sigmaX, false);
  
  double *data = REAL(Rdata);
  int nd = *INTEGER(Rnd);
  DenseMatrix<double> Ynew(my, nd, data);
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
    SEXP Rmz, SEXP RknnY, SEXP RknnX, SEXP RsigmaY, SEXP RsigmaX) {
   
  int knnY = *INTEGER(RknnY);
  int knnX = *INTEGER(RknnX);
  int n = *INTEGER(Rn);
  int nd = *INTEGER(Rnd);
  int mz = *INTEGER(Rmz);
  int my = *INTEGER(Rmy);
  double *y = REAL(Ry);
  double *data = REAL(Rdata);
  double *z = REAL(Rz);
  double sigmaY = *REAL(RsigmaY);
  double sigmaX = *REAL(RsigmaX);


  
  DenseMatrix<double> Z(mz, n, z);
  Z = Linalg<double>::Copy(Z);
  DenseMatrix<double> Y(my, n, y);
  Y = Linalg<double>::Copy(Y);
  DenseMatrix<double> Xnew(mz, nd, data);
  CEM<double> cem(Y, Z, knnY, knnX, sigmaY, sigmaX, false);

  DenseMatrix<double> *Ty = new DenseMatrix<double>[Z.M()];
  for(int k=0; k<Z.M(); k++){
    Ty[k] = DenseMatrix<double>(Y.M(), Xnew.N());
  }
  DenseMatrix<double> Yt(Y.M(), Xnew.N());

  DenseVector<double> yp(Y.M());
  DenseVector<double> xp(Z.M());
  DenseMatrix<double> J(Y.M(), Z.M());
  for(int i=0; i<Xnew.N(); i++){
	  Linalg<double>::ExtractColumn(Xnew, i, xp);
	  cem.g(xp, yp, J);
	  Linalg<double>::SetColumn(Yt, i, yp);
	  //Linalg<double>::QR_inplace(J);
	  for(int j=0; j<J.N(); j++){
	    Linalg<double>::SetColumn(Ty[j], i, J, j);
	  }
  }
  yp.deallocate();
  xp.deallocate();
  J.deallocate();

  SEXP list;
  PROTECT( list = Rf_allocVector(VECSXP, 1+Z.M()));

  SEXP Ynew;
  PROTECT(Ynew = Rf_allocMatrix(REALSXP, my, nd));
  memcpy( REAL(Ynew), Yt.data(), my*nd*sizeof(double) );
  SET_VECTOR_ELT(list, 0, Ynew);
  Yt.deallocate();

  for(int i=0; i<Z.M(); i++){
    SEXP Tnew;
    PROTECT( Tnew = Rf_allocMatrix(REALSXP, Y.M(), nd));
    memcpy( REAL(Tnew), Ty[i].data(), Y.M()*nd*sizeof(double) );
    SET_VECTOR_ELT(list, i+1, Tnew);
    Ty[i].deallocate();
  }
  delete[] Ty;
  

  
  UNPROTECT(2+Z.M());


  cem.cleanup();
  
  return list;  
}





//CEM regression methods
SEXP cemr_create(SEXP Ry, SEXP Rz, SEXP Rl, SEXP Rn, SEXP Rmy, SEXP Rmz, SEXP
    Rml, SEXP RknnY, SEXP RknnX, SEXP Rf, SEXP Rlambda, SEXP Ri, SEXP Rs, SEXP Rverbose) {
  
  int verbose = *INTEGER(Rverbose);
  int knnY = *INTEGER(RknnY);
  int knnX = *INTEGER(RknnX);
  int iter = *INTEGER(Ri);
  int n = *INTEGER(Rn);
  int mz = *INTEGER(Rmz);
  int ml = *INTEGER(Rml);
  int my = *INTEGER(Rmy);
  double *y = REAL(Ry);
  double *z = REAL(Rz);
  double *l = REAL(Rl);
  double s = *REAL(Rs);
  double fudge = *REAL(Rf);
  double lambda = *REAL(Rlambda);
  
  DenseMatrix<double> Z(mz, n, z);
  DenseMatrix<double> L(ml, n, l);
  DenseMatrix<double> Y(my, n, y);
  Z = Linalg<double>::Copy(Z);
  L = Linalg<double>::Copy(L);
  Y = Linalg<double>::Copy(Y);
  
  
  CEMRegression<double> cemr(Y, Z, L, fudge, knnY, knnX, lambda);
  cemr.gradDescent(iter,s, verbose);

  //Store kernels
  MahalanobisKernel<double> *kY = cemr.getKernelsY();
  DenseMatrix<double> T(mz*my, n);
  DenseMatrix<double> var(mz+1, n);
  DenseMatrix<double> mean(my, n);
  for(int i=0; i<n; i++){
    MahalanobisKernelParam<double> &p = kY[i].getKernelParam();
    for(int k = 0; k<mz; k++){
      for(int j=0; j<my; j++){
        T(k*my+j, i) = p.ev(j, k);
      } 
      var(k, i) = p.var(k);
    }
    var(mz, i) = p.varOrtho;
    Linalg<double>::SetColumn(mean, i, p.mean);
  }

    
  SEXP list;
  PROTECT( list = Rf_allocVector(VECSXP, 5));

  SEXP Zopt;
  PROTECT(Zopt = Rf_allocMatrix(REALSXP, mz, n));
  memcpy( REAL(Zopt), cemr.getZ().data(), mz*n*sizeof(double) );
  SET_VECTOR_ELT(list, 0, Zopt);

  SEXP sigma;
  PROTECT(sigma = Rf_allocVector(REALSXP, 1));
  double *sigmap = REAL(sigma);
  sigmap[0] = cemr.getSigmaX();
  SET_VECTOR_ELT(list, 1, sigma);

  SEXP va;
  PROTECT(va = Rf_allocMatrix(REALSXP, mz+1, n));
  memcpy( REAL(va), var.data(), (mz+1)*n*sizeof(double) );
  SET_VECTOR_ELT(list, 2, va);
  var.deallocate();

  SEXP me;
  PROTECT(me = Rf_allocMatrix(REALSXP, my, n));
  memcpy( REAL(me), mean.data(), my*n*sizeof(double) );
  SET_VECTOR_ELT(list, 3, me);
  mean.deallocate();
  
  SEXP Tnew;
  PROTECT( Tnew= Rf_allocMatrix(REALSXP, Y.M()*mz, n));
  memcpy( REAL(Tnew), T.data(), Y.M()*mz*n*sizeof(double) );
  SET_VECTOR_ELT(list, 4, Tnew);
  T.deallocate();
  
  
  
  UNPROTECT(6);
   
  cemr.cleanup();

  return list;  
 
}



//CEM regression methods
SEXP cemr_optimize(SEXP Ry, SEXP Rz, SEXP Rl, SEXP Rn, SEXP Rmy, SEXP Rmz, SEXP
    Rml, SEXP RknnY, SEXP RknnX, SEXP Rsigma, SEXP Rlambda, SEXP Ri, SEXP Rs,
    SEXP Rverbose, SEXP RT, SEXP Rvar, SEXP Rmean) {
  
  int verbose = *INTEGER(Rverbose);
  int knnY = *INTEGER(RknnY);
  int knnX = *INTEGER(RknnX);
  int iter = *INTEGER(Ri);
  int n = *INTEGER(Rn);
  int mz = *INTEGER(Rmz);
  int ml = *INTEGER(Rml);
  int my = *INTEGER(Rmy);
  double *y = REAL(Ry);
  double *z = REAL(Rz);
  double *l = REAL(Rl);
  double s = *REAL(Rs);
  double sigma = *REAL(Rsigma);
  double lambda = *REAL(Rlambda);
  
  DenseMatrix<double> Z(mz, n, z);
  DenseMatrix<double> L(ml, n, l);
  DenseMatrix<double> Y(my, n, y);
  Z = Linalg<double>::Copy(Z);
  L = Linalg<double>::Copy(L);
  Y = Linalg<double>::Copy(Y);
  
 
  double *t = REAL(RT);
  double *var = REAL(Rvar);
  double *mean = REAL(Rmean);
  DenseMatrix<double> T(mz*my, n, t);
  DenseMatrix<double> M(my, n, mean);
  DenseMatrix<double> V(mz+1, n, var);
  MahalanobisKernel<double> *kY = new MahalanobisKernel<double>[n];
  for(int i=0; i<n; i++){
    MahalanobisKernelParam<double> p;
    p.mean = Linalg<double>::ExtractColumn(M, i);
    p.varOrtho = V(mz, i);
    p.ev = DenseMatrix<double>(my, mz);
    p.var = DenseVector<double>(mz);
    for(int k=0; k<mz; k++){ 
      for(int j=0; j<my; j++){
        p.ev(j, k) = T(k*my+j, i);
      }
      p.var(k) = V(k, i);
    }
    kY[i].setKernelParam(p);
  }

 
  CEMRegression<double> cemr(Y, Z, L, knnY, knnX, sigma, lambda, kY);
  cemr.gradDescent(iter, s, verbose);


  //Store kernels
  kY = cemr.getKernelsY();
  DenseMatrix<double> Tn(mz*my, n);
  DenseMatrix<double> varn(mz+1, n);
  DenseMatrix<double> meann(my, n);
  for(int i=0; i<n; i++){
    MahalanobisKernelParam<double> &p = kY[i].getKernelParam();
    for(int k = 0; k<mz; k++){
      for(int j=0; j<my; j++){
        Tn(k*my+j, i) = p.ev(j, k);
      } 
      varn(k, i) = p.var(k);
    }
    varn(mz, i) = p.varOrtho;
    Linalg<double>::SetColumn(meann, i, p.mean);
  }
  



    SEXP list;
  PROTECT( list = Rf_allocVector(VECSXP, 5));

  SEXP Zopt;
  PROTECT(Zopt = Rf_allocMatrix(REALSXP, mz, n));
  memcpy( REAL(Zopt), cemr.getZ().data(), mz*n*sizeof(double) );
  SET_VECTOR_ELT(list, 0, Zopt);

  SEXP sigman;
  PROTECT(sigman = Rf_allocVector(REALSXP, 1));
  double *sigmap = REAL(sigman);
  sigmap[0] = cemr.getSigmaX();
  SET_VECTOR_ELT(list, 1, sigman);

  SEXP va;
  PROTECT(va = Rf_allocMatrix(REALSXP, mz+1, n));
  memcpy( REAL(va), varn.data(), (mz+1)*n*sizeof(double) );
  SET_VECTOR_ELT(list, 2, va);
  varn.deallocate();

  SEXP me;
  PROTECT(me = Rf_allocMatrix(REALSXP, my, n));
  memcpy( REAL(me), meann.data(), my*n*sizeof(double) );
  SET_VECTOR_ELT(list, 3, me);
  meann.deallocate();
  
  SEXP Tnew;
  PROTECT( Tnew= Rf_allocMatrix(REALSXP, Y.M()*mz, n));
  memcpy( REAL(Tnew), Tn.data(), Y.M()*mz*n*sizeof(double) );
  SET_VECTOR_ELT(list, 4, Tnew);
  Tn.deallocate();
  
  
  
  UNPROTECT(6);


  cemr.cleanup();


  return list;  
}


}//end extern C
