cem.example.sr <- function(n =1000, nstd=0.5, init=0, type=2){

library(cems)
library(rgl)
library(vegan)
  
#Create data 
data(swissroll)
d     <- swissroll(N=n, nstd = nstd)
dtest <- swissroll(N=n, nstd = nstd)

#init
if(init == 0){
  iso <- isomap(dist(d$Xn), k=15)
  z = iso$points[, 1:2] #d$X[, 2:3]
}else{
  z = d[, 3]
}


ps0 <- cem(y = d$Xn, z = z, knnX=300, knnY=50, iter=0, verbose=2, stepZ=1,
    sigmaY=0.3, sigmaX = 0.3, stepBW=0.1, type=0)


  ps = ps0
  ps$type = type

  if(ps$type==0){
    col = "gold2"  
  }else{
    col = "dodgerblue2"
  }

  px <-predict(ps);
  perr = Inf


  #plot(iso$points[, 1:2], pch=19, col="darkgray") 
  plot(px, pch=19, col="darkgray") 


  for(i in 1:200){
    pz = ps$z
    ps <- cem.optimize(ps, iter=1, verbose=2, stepZ=1, stepBW=0.1)
    x <- predict(ps)

    #segments(ps$z[,1], ps$z[, 2], pz[,1], pz[,2], lwd=2, col="#1C86EE55")
    segments(px[,1], px[, 2], x[,1], x[,2], lwd=2, col="#1C86EE55")

    tx <- predict(ps, dtest$Xn)
    ty <- predict(ps, tx)
    err <- mean( rowSums( (ty$y-dtest$Xn)^2 ) )
    print(err)
    if(ps$type==0){
      if(err >= perr) break 
      perr=err;
    }
    else{
      if(sum( (px - x) != 0 ) == 0 ) break;
    }
    px = x
  } 
   
  coords <- predict(ps, dtest$Xn)
  p <- predict(ps, coords)
     
  mean( rowSums((dtest$X - p$y)^2) )




  #build surface model
  range(x[,1])
  range(x[,2])
  r1 <- range(x[,1])
  r2 <- range(x[,2]) #*0.95
  s1 <- seq(r1[1], r1[2], length.out=20)
  s2 <- seq(r2[1], r2[2], length.out=20)
  #s2 <- s2[4:7]
  l1 <- length(s1)
  l2 <- length(s2)

  gx <- matrix(nrow=l1*l2, ncol=2)
  index = 1;
  for(x1 in s1){
    for(x2 in s2){
      gx[index, 1] = x1
      gx[index, 2] = x2
      index = index +1
    }
  }

  gy <- predict(ps, gx)

  indices = c()
  index = 1
  for(i in 1:(l1-1) ){
    for(j in 1:(l2-1) ){
      indices[index] =   (i-1)*l2+1   +j-1
      indices[index+1] = i*l2+1       +j-1
      indices[index+2] = i*l2+1       +j
      indices[index+3] = (i-1)*l2+1   +j
      index = index + 4
    }
  }

  qm <- qmesh3d(vertices=t(gy$y), indices=indices, homogeneous=F)

  #rgl.open()
  #rgl.bg(color="white")
  #plot3d(d$Xn, type="s", radius=0.35, box=F, alpha=0.25)
  #material3d(color=col, alpha=0.75, ambient=col, depth_mask=F)
  #shade3d(qm)
  #material3d(col="darkgray", ambient="darkgray", lit=F, alpha=0.5, depth_mask=T)
  #wire3d(qm)



  pgx <- predict(ps, dtest$gYn)
  pgy <- predict(ps, pgx)

  rgl.open()
  rgl.bg(color="white")
  plot3d(pgy$y, type="s", col = col,  radius=0.3, box=F, axes=F, , alpha=1, xlab="", ylab="", zlab="")

  material3d(lwd=2, alpha=0.75)
  wire3d(d$qm)
  #y <- predict(ps, x)
  #spheres3d(y$y, radius=0.3)



if(F){
plot3d(d$Xn, type="s", col = "gray", radius=0.3, box=F, axes=F, , alpha=0.5, xlab="", ylab="", zlab="")

material3d(lwd=2, alpha=0.75)
wire3d(d$qm)


col = "darkgray"
alpha=0.5

#qm = qm1
#col = "dodgerblue2"
#alpha=0.9

#qm = qm0
#col = "gold2"
#alpha = 0.3

material3d(color=col, alpha=alpha, ambient=col, lit=T)
shade3d(qm)
material3d(col=col, ambient="black", lit=F, alpha=0.5, lwd=3)
wire3d(qm)



}


}	
