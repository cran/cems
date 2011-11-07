\name{cem}
\alias{cem}
\alias{cem.single}
\alias{cems}
\alias{cems-package}
\alias{cem.optimize}
\alias{predict.cem}
\title{Conditional Expectation Manifolds}
\description{
This package computes principal surfaces based on the approach described in
Gerber et. al. 2009 and Gerber and Whitaker 2011. 

Prinicpal surfaces are typically found by minimizing \eqn{ E[ || Y
-g(\lambda(Y))||^2 ] } over the functions \eqn{g:R^m \mapsto R^n}{g:R^m -> R^n}
with \eqn{m < n} {m < n} and \eqn{ \lambda_g:R^n \mapsto R^m}{\lambda_g:R^n ->
R^m} defined as an orthogonal projection onto \eqn{g}{g}. 

In Gerber et. al. 2009 the oppoiste approach is described; fixing
\eqn{g_{\lambda}(x) = E[Y | \lambda(Y) = x] }{ g_\lambda(x) = E[Y | \lambda(Y) =
x } 
and minimzing over \eqn{\lambda}{\lambda}, i.e. optimizing the conditonal
expectation manifold (CEM) given \eqn{\lambda}{\lambda}. Gerber et. al. 2009
called this approach kernel map manifolds (KMM) since both mappings where defined
by kernel regression.

In Gerber and Whitaker 2011 the same formulation is exploited to provide an
approach that solves the model selection problem for principal curves.  The
orthogonal projection distance minimization
\eqn{E[ ||Y - g_\lambda(Y)||^2]}
yields principal curves that are saddle points and thus model selection (i.e.
bandwidth slection in this implementation) is not possible even with
cross-validation.  The approach in Gerber and Whitaker 2011 formulates an
alternate optimization problem minimzing orthogonality
\eqn{E[ < Y - g_\lambda(Y), \frac{d}{ds} g(s)|_{s=|lambda(Y)}>^2 ] }
which leads to principal curves at minima.

This package implements the approach in Gerber et. al. 2009 for both
formulation, i.e. minimzing  projection distance
\eqn{E[ ||Y - g_\lambda(Y)||^2]} or orthogonality
\eqn{E[ < Y - g_\lambda(Y), \frac{d}{ds} g(s)|_{s=|lambda(Y)}>^2 ] }.
. The implementation is based on a kernel regression for \eqn{\lambda} and
\eqn{g} and uses a numerical gradient descent for minimization. The gradient
descent includes an optimization of the bandwidth, i.e. model selection. For
minimzing the projection distance this dopes not lead to good results since
principal curves are saddle points. Thus \code{stepBW} should be set to 0 in
this case.


}


\usage{
cem( y, z,  knnY=0.5*nrow(y), knnX = 5*ncol(z), sigmaY= 1/3, sigmaX=1/3, 
     iter =100, stepZ = 1, stepBW = 0.1, verbose=1, scales=nrow(y), type=2,
     sigmaAsFactor = T, optimalSigmaX =  T )
cem.single( y, z,  knnY=0.5*nrow(y), knnX = 5*ncol(z), sigmaY= 1/3, sigmaX=1/3, 
     iter =100, stepZ = 1, stepBW = 0.1, verbose=1, type=2, sigmaAsFactor = T, 
     optimalSigmaX =  T )
cem.optimize(object, iter = 50, stepZ=1, stepBW=0.1, verbose=1, optimalSigmaX =  T )
\method{predict}{cem}(object, newdata, ...)
}

\arguments{
  \item{y}{\eqn{n}{n}-dimensional data to compute conditional expectation
manifold for.}
  \item{z}{Initalization for low dimensional mapping \eqn{\lambda}{\lambda}. For
example an isomap or lle or PCA embedding of \eqn{y}{y}.}
  \item{knnX}{Number of nearest neighbors for kernel regression of \eqn{g}, i.e.
the regression is trunctaed to only the knnX nearest neighbors}    
  \item{knnY}{Number of nearest neighbors for kernel regression of \eqn{\lambda}, i.e. the
regression is trunctaed to only the knnX nearest neighbors}  
 \item{sigmaX}{Initalize bandwidth of \eqn{g} to sigmaX. If \code{sigamAsFactor}
is set to true the bandiwdth is computed as sigmaX times average knnX nearest
neighbor distance.}  
 \item{sigmaY}{Initalize bandwidth of \eqn{\lambda} to sigmaY. If \code{sigamAsFactor}
is set to true the bandiwdth is computed as sigmaX times average knnY nearest
neighbor distance.}  
  \item{iter}{Number of optimization iterations, i.e. number of gradient desecent with
line search setps.}
  \item{stepZ}{Gradient descent step size for optimizing Z}
  \item{stepBW}{Gradient descent step size for optimizing bandwidths}
  \item{verbose}{Report gradient descent information. 1 reports iteration number
and mean squared projetcion distances. 2 has additonal information on step size
and line search.}
  \item{sigmaAsFactor}{Use sigmaX and sigmaY as multipliers of the average
nearest neighbor distances in \eqn{Y} and \eqn{\lambda(Y)}. respectively.}
  \item{optimalSigmaX}{If true optimizes sigmaX before every iteration - will
not work for MSE minimzation, i.e. sigmaX will go to 0- owrk well for orthogonal
projectiona nd speeds up computation isgnificantly}
  \item{scales}{A list of the number of points used for iteratively computing
the manifold, i.e. start with a small number of points to compute a quick rough
approximation and refine the estimation by adding more points. If the whole data
set should be used at once set scales to \code{nrow(y)}}  
  \item{type}{Which objective function should be minimized. 1 = \eqn{E[ ||Y -
g_\lambda(Y)||^2]}. 2 = \eqn{E[<g(f(y)) - y , g'(f(y))>^2]}. 3 = 2 but with
\eqn{g'(f(y))>} ortho normalized. 4=3 with \eqn{g(f(y)) - y } normalized.}
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
\seealso{
  \code{\link{cem.example.arc}}
  \code{\link{cem.example.sr}}
}
\references{
Samuel Gerber, Tolga Tasdizen, Ross Whitaker,
Dimensionality Reduction and Principal Surfaces via Kernel Map Manifolds, 
In Proceedings of the 2009 International Conference on Computer Vison (ICCV 2009).

Samuel Gerber and Ross Whitaker,
Conditional Expectation Curves, 
Submitted 2011.

}
\examples{
library(cems)


##Noisy half circle example
phi <- runif(1000)*pi
arc <- cbind(sin(phi), cos(phi)) * (1+rnorm(1000) * 0.1)
plot(arc)

pc <-  cem(y=arc, z=phi, knnX=15, knnY=15, iter=10)

x <- predict(pc)
y <- predict(pc, x);
plot(y$y)


}
\keyword{nonparametric,models,nonlinear}
