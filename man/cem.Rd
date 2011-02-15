\name{cem}
\alias{cem}
\alias{cem.optimize}
\alias{predict.cem}
\title{Conditional Expectation Manifolds}
\description{
This package computes principal surfaces based on the approach described in
Gerber et. al. 2009. Prinicpal surfaces are typically found by minimizing 
\eqn{ E[ || Y -g(\lambda(Y))|| ] }{E[ || Y -g(\lambda_g(Y))|| ] } over the
functions \eqn{g:R^m \mapsto R^n}{g:R^m -> R^n} with \eqn{m < n} {m < n}
and \eqn{ \lambda_g:R^n \mapsto R^m}{\lambda_g:R^n -> R^m} defined as an 
orthogonal projection onto \eqn{g}{g}. In Gerber et. al. 2009 the oppoiste 
approach is described; fixing 
\eqn{g_{\lambda}(x) = E[Y | \lambda(Y) = x] }{ g_\lambda(x) = E[Y | \lambda(Y) = x } 
and minimzing over \eqn{\lambda}{\lambda}, i.e. optimizing the conditonal
expectation manifold (CEM) given \eqn{\lambda}{\lambda}. Gerber et. al. 2009
called this approach kernel map manifolds (KMM) since both mappings where defined
by kernel regression. 
}


\usage{
cem(y, z, knn = 15, iter = 100, step = 0.1, fudge = 2.5, verbose=1)
cem.optimize(object, iter = 50, step=0.1, verbose=1)
\method{predict}{cem}(object, newdata, ...)
}

\arguments{
  \item{y}{\eqn{n}{n}-dimensional data to compute conditional expectation
manifold for.}
  \item{z}{Initalization for low dimensional mapping \eqn{\lambda}{\lambda}. For
example an isomap or lle or PCA embedding of \eqn{y}{y}.}
  \item{knn}{Number of nearest neighbors for graph computation.}    
  \item{iter}{Number of optimization iterations, i.e. number of gradient desecent with
line search setps.}
  \item{step}{Gradient descent step size}
  \item{fudge}{Fudge factor, amount of over-smoothing of inital CEM.}.
  \item{verbose}{Report gradient descent information. 1 reports iteration number
and mean squared projetcion distances. 2 has additonal information on step size
and line search.} 
  \item{object}{CEM object to do prediction for}
  \item{newdata}{Data to do prediction for. If \code{ncol(newdata) == m} for each
point x in newdata g(x) is computed. If \code{col{newdata} == n} for each point
y in newdata \eqn{\lambda(y)}{\lambda(y)} is computed.}
  \item{...}{Additional arguments have no effect.}
}

\value{
  An object of class \code{"cem"}.
}

\author{
  Samuel Gerber
}
\references{
Samuel Gerber, Tolga Tasdizen, Ross Whitaker,
Dimensionality Reduction and Principal Surfaces via Kernel Map Manifolds, 
In Proceedings of the 2009 International Conference on Computer Vison (ICCV 2009).
}
\examples{
library(vegan)
library(rgl)
library(cems)

###Swissroll data set example

#Create swissroll toy data set
data(swissroll)
d <- swissroll(500, nstd = 0.5)
plot3d(d$Xn)
#d$Xn contains noisy (orthogonal Normal distirbuted to the swissroll) of d$X

#Compute isomap
iso <- isomap(dist(d$Xn), k=15)

#Create and optimize CEM
ps <- cem(y = d$Xn, z = iso$points[, 1:2], knn=20, iter=5)

#Create test set
dtest <- swissroll(500, nstd = 0.25)

#parametrize manifold coordinates from test set
coords <- predict(ps, dtest$Xn)

#reconstruct points on manifold from coordinates
p <- predict(ps, coords)

#mean squared test error
mean( rowSums((dtest$X - p)^2) )

#optimize some more with samller step size
ps <- cem.optimize(ps, iter=5, step = 0.05)

#mean sqyuared error
coords <- predict(ps, dtest$Xn)
p <- predict(ps, coords)
mean( rowSums((dtest$X - p)^2) )



###Frey faces example
#Do not run
\dontrun{
data("frey_faces")

#create isomap
iso <- isomap(dist(faces), k=25)

#Create conditional expectation surface (no optimization here, 20 iteration take
#roughly 2h on a 2Ghz laptop)
ps <- cem(y=faces, z = iso$points[, 1:3], knn=45, iter=0)

#parametrize manifold coordinates
coords <- predict(ps)

#reconstruct points on manifold from coordinates
p <- predict(ps, coords)

#mean squared test error
mean( rowSums((faces - p)^2) )

#plot images
split.screen(c(2,3))


ind <- c(1, 120, 333)
for(i in 1:length(ind)){
 screen(i)
 im <- matrix(faces[ind[i], 560:1], 20, 28)
 image(1:nrow(im), 1:ncol(im), im, xlab="", ylab="")
 screen(i+3)
 im <- matrix(p[ind[i], 560:1], 20, 28)
 image(1:nrow(im), 1:ncol(im), im, xlab="", ylab="")
}
}

}
\keyword{nonparametric,models,nonlinear}
