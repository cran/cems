swissroll <- function(N=1000, phi = pi, height=4, nstd = 0.1){
h = height
tt = phi*(1+2*runif(N))  
height = height*runif(N)
X = rbind(tt * cos(tt), height, tt * sin(tt));

Xa = rbind(cos(tt)- tt * sin(tt), matrix(0, 1, N), sin(tt) + tt * cos(tt))
Xo = rbind(-Xa[3,], 0, Xa[1, ]);
Xo = Xo / matrix(sqrt(colSums(Xo^2, 1)), ncol=N, nrow=3);
Xn = X;
Xn = X + Xo* t(matrix(nstd*rnorm(N), N, 3));

rownames(X) <- c("x", "y", "z")
rownames(Xn) <- c("x", "y", "z")
obj <- structure(list( Xn = t(Xn), X = t(X) ) )

obj
}
