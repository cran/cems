
cem.example.arc <- function(n=150, noise=0.2, type=2, sigmaX= 0.1, sigmaY=0.1,
init = 0, plotEach = 10){


### Create arc data set

#create arc
phi <- runif(n)*pi 
arc <- cbind(cos(phi), sin(phi)) * (1+rnorm(n) * noise)
arc0 <- cbind(cos(phi), sin(phi))

#order by angle for plotting curves    
o0 <- order(phi) 

#test data
phit <- runif(n*10)*pi 
arct <- cbind(cos(phit), sin(phit)) * (1+rnorm(10*n) * noise)





### Initialization

if(init == 0){
 #ground truth initialization
 z <- phi
}else if(init == 1){
 #random initialization
 z <- runif(n)
}
else{
 #close to principal component initialization
 z = arc[,2]
}

#compute initial principal curve
#do not optimize sigmaX after each itertion to show optimization path
pc <-  cem(y=arc, z=z, knnX=n, knnY=50, iter=0, optimalSigmaX=F)

#set type
pc$type=type

#set intial bandwidth
pc$sigmaX=sigmaX; #smoothness of initial curve 
pc$sigmaY=sigmaY;


#initial prediction (curve)
xi <- predict(pc)
yi <- predict(pc, xi)
oi <- order(xi)

#create sample from initial curve for plotting
r = range(xi)
xpi = seq(r[1], r[2], length.out=500)
ypi = predict(pc, xpi)


if( pc$type > 0 ){
 colName <- "dodgerblue2"
}else{
 colName <- "gold2"
}

tmp <- col2rgb(colName)
col <- rgb(tmp[1], tmp[2], tmp[3], 160, maxColorValue=255)
col2 <- rgb(tmp[1], tmp[2], tmp[3], 255, maxColorValue=255)

lty <- 1
lwd <- 8
pcs <- list()
o <- list()
x <- list()
sigmaX <- list()
sigmaY <- list()
iter <- list()
y <- list()
yt <- list()
mse = NULL
mset = NULL
ortho = NULL
orthot = NULL

lwi = 6
lwg = 6
lwp = 4



### Optimization

#run nIter iterations between plotting
nIter = plotEach


#plot ground truth and initial curve
par(mar=c(5,5,4,2))
plot(arc, xlab=expression(y[1]), ylab=expression(y[2]), col = "#00000010",
       pch=19, asp=1, cex.lab=1.75, cex.axis=1.75, cex=2, bty="n")
lines(arc0[o0,], lwd=lwg, col="black", lty=6)
lines(ypi$y, col="darkgray", lwd=lwi, lty=1)


#cross validation flag (selected curve)
selected = -1


#run a 100 iterations, one at a time 
#for running the whole optimization in one go run either
# pc <-  cem(y=arc, z=z, knnX=n, knnY=50, iter=100)
# pc <- cem.optimize(pc, stepZ=1, stepBW=0.1, iter=100)
# here one iterations is run at the time and corss-validation is performed on a
# test set. For minimizing orthogonality cross-validation appears to be not
# necessary.

for(k in 1:100){
  #run one iterations  
  pc <- cem.optimize(pc, stepZ=1, stepBW=0.1, iter=1, verbose=2, optimalSigmaX=F)
  
  #store cem values at iteration k
  
  sigmaX[k] <- pc$sigmaX
  sigmaY[k] <- pc$sigmaY
  

  x[[k]] <-predict(pc)
  
  #train data
  r = range(x[[k]])
  xp = seq(r[1], r[2], length.out=500)
  yp = predict(pc, xp);
  o[[k]] <- order(x[[k]]);
  y[[k]] <- predict(pc, x[[k]])

  #test data
  xt <-predict(pc, arct)
  yt[[k]] <-predict(pc, xt)
  
  #compute mean squared errors
  dt <- (yt[[k]]$y - arct)
  lt <- rowSums(dt^2)
  d <- (y[[k]]$y - arc)
  l <- rowSums(d^2)
  mset[k] <- mean( lt )
  mse[k] <- mean( l )
  
  #compute orthogonaility
  if(pc$type == 3 || pc$type == 0){
    d = d / cbind(sqrt(l), sqrt(l))
    dt = dt / cbind(sqrt(lt), sqrt(lt))
  }
  if(pc$type != 1){
    tl = rowSums( y[[k]]$tangents[[1]]^2)
    tlt = rowSums( yt[[k]]$tangents[[1]]^2)
    y[[k]]$tangents[[1]] = y[[k]]$tangents[[1]] / cbind(sqrt(tl), sqrt(tl))
    yt[[k]]$tangents[[1]] = yt[[k]]$tangents[[1]] / cbind(sqrt(tlt), sqrt(tlt))
  } 
  ortho[k]  = mean( rowSums( y[[k]]$tangents[[1]]  * d  )^2 )
  orthot[k] = mean( rowSums( yt[[k]]$tangents[[1]] * dt )^2 )
 
  #print orthogoanilty and mse values
  print( sprintf("ortho: %f", ortho[k]) ) 
  print( sprintf("mse: %f", mse[k]) )


  #cross-validation 
  if(k>1){
    if(pc$type == 0){
      if(mset[k] > mset[k-1]){
        selected = selected+1
      }
    }
    else{
      if(orthot[k] > orthot[k-1]){
        selected = selected+1
      }
    }
  }

  #print curve every 10 iteration until selected curve based on cross-validation
  if(selected < 0){
    if(k %% nIter == 0){
      lines(yp$y, col=col, lty=1, lwd=lwp)
    }
  }
  if(selected == 0){
     lines(yp$y, col=col2, lty=1, lwd=lwd)
     selected=1;
  }
  
}


#cross-validtion did not select curve - plot curve at end of  optimization
if(selected < 0 ){
  lines(yp$y, col=col2, lty=1, lwd=lwd)
  selected=1;
}

#pretty plot legend
legend("topright", col = c("black", "darkgray", col, col2), lty=c(6, 1, 1, 1), lwd=c(lwg, lwi, lwp, lwd),
legend = c("ground truth", "initialization", "intermediates", "selected"), cex=1.75, bty="n", seg.len=4 )



#plot test and train error
dev.new()

par(mar=c(5,6,4,2))
if(pc$type==0){
  mset[mset > 5*max(mse)] = 5*max(mse)
  
  plot((1:length(mse)), mse, type="l", lwd=8, cex.lab=1.75, cex.axis=1.75,
       col="darkgray", lty=1, xlab="iteration", ylab=expression(hat(d)(lambda,
       Y)^2), bty="n", ylim = range(mse, mset)) 
  mset = mset -0.005
  lines((1:length(mset)), mset, lwd=8, lty=2, col="black")

  i1 <- which.min(mse)
  i2 <- which.min(mset)
  points(i1, mse[[i1]], col="darkgray", pch=19, cex=3) 
  points(i2, mset[[i2]], col="black", pch=19, cex=3) 
  
  legend("topright", col = c("darkgray", "black"), lty=c(1,2), lwd=c(8,8), legend = c("train", "test"), cex=1.75, bty="n", seg.len=4 )

}
if(pc$type != 0){

  plot((1:length(ortho)), ortho, type="l", lwd=8, cex.lab=1.75, cex.axis=1.75, col="darkgray", lty=1, xlab="iteration", ylab=expression(hat(q)(lambda, Y)^2), bty="n", ylim = range(ortho, orthot))
  lines((1:length(orthot)),orthot, lwd=8, lty=2, col="black")

  i1 <- which.min(ortho)
  i2 <- which.min(orthot)
  points(i1, ortho[[i1]], col="darkgray", pch=19, cex=3) 
  points(i2, orthot[[i2]], col="black", pch=19, cex=3) 

legend("topright", col = c("darkgray", "black"), lty=c(1,2), lwd=c(8,8),
legend = c("train", "test"), cex=1.75, bty="n", seg.len=4 )

}





}
  

