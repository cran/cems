#cem <- function (y, z, knnY = 25, knnX = nrow(y)*0.1, iter = 100, step = 0.1, 
# 	 	 fudge = 1, verbose=1, scales = nrow(y), cv = 0 ){
# if(cv <= 0 || cv >= 1){
#   ps <- cem.scales(y=y, z=z, knnX=knnX, knnY=knnY,
#             iter=iter, step=step, fudge=fudge, verbose=verbose, scales=scales)
# }
# else{
#}
#}



cem <- function (y, z, knnY = 0.5*nrow(y), knnX = 5*ncol(z), sigmaY = 1/3,
                 sigmaX=1/3, iter = 100, stepZ = 1, stepBW = 0.1, verbose=1, 
                 scales = nrow(y), type = 2, sigmaAsFactor = T, optimalSigmaX = T ){
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
  

  scales <- sort(scales)
  r <- order(runif(nrow(y)));
  #initial cem at coarsest scale
  scale = scales[1];
  ps <- cem.single(y=y[r[1:scale], ], z=z[r[1:scale], ], knnX=knnX, knnY=knnY,
             sigmaY = sigmaY, sigmaX = sigmaX, iter=iter, stepZ=stepZ, stepBW=stepBW, 
             verbose=verbose, type=type, sigmaAsFactor = sigmaAsFactor,
             optimalSigmaX =  optimalSigmaX );
  
  if(length(scales) > 1){
    #update ps with additional points
    for( i in 2:length(scales) ){

        scale <- scales[i];
        scaleP <- nrow(ps$z)+1

        #yadd = y[r[scaleP:scale], ]
        yadd = y[r[1:scale], ]
        x <- predict(ps, yadd);
        #ky <- cem.estimate.kernel(ps, x);
        #ps$kV <- cbind(ps$kV, ky$kV)
        #ps$kT <- cbind(ps$kT, ky$kT)
        #ps$kM <- cbind(ps$kM, ky$kM)
        #ps$y <- rbind(ps$y, yadd) 
        #ps$z <- rbind(ps$z, x )
        
        #ps$kV <- ky$kV
        #ps$kT <- ky$kT
        #ps$kM <- ky$kM
        ps$y  <- yadd 
        ps$z  <- x
        
        ps$knnX <- floor(ps$knnX/scaleP*scale)
        ps <- cem.optimize(ps, iter=iter, stepZ=stepZ, stepBW=stepBW,
                           verbose=verbose, optimalSigmaX =  optimalSigmaX )

	#ps <- cem.single(y=yadd, z=x, knnX=knnX/scaleP*scale, knnY=knnY,
        #     iter=iter, step=step, fudge=fudge, verbose=verbose);
    }
  }
  #reverse random order
  ps$y[r,] = ps$y
  ps$z[r,] = ps$z
  #ps$fy[r,] = ps$fy
  #ps$kV[,r] <- ps$kV
  #ps$kT[,r] <- ps$kT
  #ps$kM[,r] <- ps$kM


  ps
}

   



cem.single <- function (y, z, knnY=0.5*nrow(y), knnX = 5*ncol(z), sigmaY=1/3, sigmaX=1/3, 
                         iter = 100, stepZ = 1, stepBW = 0.1, verbose=1, type=2,
                          sigmaAsFactor=T, optimalSigmaX =  T ) 
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
    res <- .Call("cem_create", as.double(t(y)), as.double(t(z)), nry, ncy, ncz, 
	as.integer(knnY), as.integer(knnX), as.double(sigmaY),
	as.double(sigmaX), as.integer(iter), as.double(stepZ),
        as.double(stepBW), as.integer(verbose), as.integer(type),
        as.integer(sigmaAsFactor), as.integer(optimalSigmaX))  
    
    obj <- structure( list( y=y, z=t(as.matrix(res[[1]])), knnY= knnY, knnX = knnX,
                            sigmaX=res[[2]], sigmaY = res[[3]], type=type ), class="cem") 
    obj

}



#Optimize existing CEM further
cem.optimize <- function(object, iter = 50, stepZ=1, stepBW=0.1, verbose=1, 
                         optimalSigmaX =  T ){

  y <- object$y
  z <- object$z
  knnX <- object$knnX
  knnY <- object$knnY
  sigmaX <- object$sigmaX
  sigmaY <- object$sigmaY
  nry <- nrow(y)
  nrz <- nrow(z)
  ncy <- ncol(y)
  ncz <- ncol(z)
  res <- .Call("cem_optimize", as.double(t(y)), as.double(t(z)), nry, ncy, ncz,
      as.integer(knnY), as.integer(knnX), as.double(sigmaY),as.double(sigmaX), 
      as.integer(iter), as.double(stepZ), as.double(stepBW),
      as.integer(verbose), as.integer(object$type), as.integer(optimalSigmaX) ) 
    
  object$z = t(as.matrix(res[[1]]))
  object$sigmaX = res[[2]] 
  object$sigmaY = res[[3]] 
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
  knnX <- object$knnX
  knnY <- object$knnY
  sigmaY <- object$sigmaY
  sigmaX <- object$sigmaX
  nry <- as.integer(nrow(y))
  nrz <- as.integer(nrow(z))
  ncy <- as.integer(ncol(y))
  ncz <- as.integer(ncol(z))
  nrd <- as.integer(nrow(data))
  if(ncol(data)  == ncol(object$y) ){
    res <- .Call("cem_parametrize", as.double(t(data)), nrd, as.double(t(y)),
        as.double(t(z)), nry, ncy, ncz, as.integer(knnY), as.integer(knnX),
        as.double(sigmaY), as.double(sigmaX) )  
    res <- t(as.matrix(res))
  }
  else{    
    res <- .Call("cem_reconstruct", as.double(t(data)), nrd, as.double(t(y)),
        as.double(t(z)), nry, ncy, ncz, as.integer(knnY), as.integer(knnX),
        as.double(sigmaY), as.double(sigmaX))  
    T = c()
    for(i in 1:ncz){
      T[[i]] = t(as.matrix(res[[1+i]]))
    }
    res <- list(y = t(as.matrix(res[[1]])), tangents=T)
  }
  res
}

