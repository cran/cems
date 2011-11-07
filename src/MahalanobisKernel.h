#ifndef MAHALANOBISKERNEL_H
#define MAHALANOBISKERNEL_H

#include "Kernel.h"
#include "EuclideanMetric.h"
#include "Linalg.h"

#include <cmath>




template<typename TPrecision>
class MahalanobisKernelParam{
  public:
	  DenseMatrix<TPrecision> ev;
	  DenseVector<TPrecision> var;
	  TPrecision varOrtho;
	  DenseVector<TPrecision> mean;
	 
	  MahalanobisKernelParam(){};

	  MahalanobisKernelParam(DenseMatrix<TPrecision> dirs,
			  DenseVector<TPrecision> vars, TPrecision ortho,
			  DenseVector<TPrecision> m){ 
	     ev = dirs;
	     var = vars;
	     varOrtho = ortho;
	     mean = m;
	  };

	  void deallocate(){
	    mean.deallocate();
	    var.deallocate();
	    ev.deallocate();
	  };
};





template <typename TPrecision>
class MahalanobisKernel{

  private:
    MahalanobisKernelParam<TPrecision> m;
    DenseVector<TPrecision> lPlane;   
    DenseVector<TPrecision> diff;   
    int c;
    
  public:
 
    MahalanobisKernel(){
    };

    MahalanobisKernel(MahalanobisKernelParam<TPrecision> params){
      setKernelParam(params); 
    };

    virtual ~MahalanobisKernel(){
      lPlane.deallocate();
      diff.deallocate();
      m.deallocate();
    };




    TPrecision f(Vector<TPrecision> &x){
      Linalg<TPrecision>::Subtract(m.mean, x, diff);
      return f( );
    };



  
    TPrecision f(Matrix<TPrecision> &X, int i){
      Linalg<TPrecision>::Subtract(X, i, m.mean, diff);
      return f( );
    };
  
  




 
    MahalanobisKernelParam<TPrecision> &getKernelParam(){
	  return m;
    };

    void setKernelParam(MahalanobisKernelParam<TPrecision> &param){
      m = param;
      int d = m.mean.N();
      int dOrtho = d - param.ev.M();
      c = 1;
      lPlane.deallocate();
      diff.deallocate();
      diff = DenseVector<TPrecision>(param.ev.M());
      lPlane.deallocate();
      lPlane = DenseVector<TPrecision>(m.var.N());
      /*c = pow(2* M_PI, d/2.0) * pow(m.varOrtho, dOrtho/2.0);
      for(int i=0; i<m.var.N(); i++){
        c *= sqrt(m.var(i));
      }
      std::cout << c << std::endl;
      c = 1/c;
      */
    };

  
    

  private:
    TPrecision f(){
      TPrecision l = Linalg<TPrecision>::SquaredLength(diff);
      

      Linalg<TPrecision>::Multiply(m.ev, diff, lPlane, true);
      double lOrtho = l;
      TPrecision dist = 0;
      for(int i=0; i<lPlane.N(); i++){
	TPrecision tmp = lPlane(i);
        tmp = tmp*tmp;
	lOrtho -= tmp;
        dist += tmp/m.var(i);
      }
      dist += lOrtho / m.varOrtho;
      return c * exp(-dist);
      
      //return exp(-l/m.varOrtho);
    };


};
#endif
