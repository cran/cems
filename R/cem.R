#type 1 indiacte to use persistence for merging, type 2 (not 1 actually) uses
#R^2 based merging criterion
cem <- function (y, z, knn = 15, iter = 100, step = 0.1, fudge = 2.5, verbose=1) 
{
    this.call <- match.call()
  
    if(is.null(nrow(y))){
      y <- as.matrix(y, ncol=1)
    }
    else{ 
      y <- as.matrix(y)
    }    
    if(is.null(nrow(z))){
      z <- as.matrix(z, ncol=1)
    }
    else{ 
      z <- as.matrix(z)
    }

    nry <- nrow(y)
    nrz <- nrow(z)
    if(nry != nry){
      stop("z and y don't have the same number of observations") 
    }
    ncy <- ncol(y)
    ncz <- ncol(z)
    res <- .Call("cem_create", as.double(t(y)), as.double(t(z)), nry, ncy, ncz, as.integer(iter), 
        as.integer(knn), as.double(step), as.double(fudge), as.integer(verbose) )  
    
    obj <- structure( list( y=y, z=t(as.matrix(res[[1]])), knn = knn,
                            sigma=res[[2]] ), class="cem") 
    obj

}



#Optimize existing CEM further
cem.optimize <- function(object, iter = 50, step=0.1, verbose=1){

  y <- object$y
  z <- object$z
  knn <- object$knn
  sigma <- object$sigma
  nry <- nrow(y)
  nrz <- nrow(z)
  ncy <- ncol(y)
  ncz <- ncol(z)
  nrd <- nrow(data)
  res <- .Call("cem_optimize", as.double(t(y)), t(z), nry, ncy, ncz, as.integer(knn), 
       as.double(sigma), as.integer(iter), as.double(step), as.integer(verbose))  
    
  object$z = t(as.matrix(res))
  object

}




predict.cem <- function(object, newdata = object$y, ...){

  if( is.null( nrow(newdata) ) ){
    data <- as.matrix(newdata, ncol=1)
  }
  else{
    data <- as.matrix(newdata)
  }
  y <- object$y
  z <- object$z
  knn <- object$knn
  sigma <- object$sigma
  nry <- as.integer(nrow(y))
  nrz <- as.integer(nrow(z))
  ncy <- as.integer(ncol(y))
  ncz <- as.integer(ncol(z))
  nrd <- as.integer(nrow(data))
  if(ncol(data)  == ncol(object$y) ){
    res <- .Call("cem_parametrize", as.double(t(data)), nrd, as.double(t(y)),
                 t(z), nry, ncy, ncz, as.integer(knn), as.double(sigma) )  
  }
  else{    
    res <- .Call("cem_reconstruct", as.double(t(data)), nrd, as.double(t(y)),
                 t(z), nry, ncy, ncz, as.integer(knn), as.double(sigma) )  
  }

  res <- t(as.matrix(res))
  res
}

