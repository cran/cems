\name{cem.example.arc}
\alias{cem.example.arc}
\title{Conditional Expectation Manifold Example on Arc}
\description{
This function runs the arc example in:

Samuel Gerber and Ross Whitaker,
Conditional Expectation Curves, 
Submitted 2011.


}


\usage{
cem.example.arc(n=150, noise=0.2, type=2, sigmaX= 0.1, sigmaY=0.1, init = 0,
                plotEach=10)
}

\arguments{
  \item{n}{ Sample size.}
  \item{noise}{ Amount of normal distributed noise orthogonal to the arc.}
  \item{type}{ Optimization objective}
  \item{sigmaX}{Intial bandwidth for the curve \eqn{g}}
  \item{sigmaY}{Intial bandwidth for the coordinate mapping \eqn{\lambda}}
  \item{init}{Type of initialization. 0 = ground truth, 1 = random, 2 = y-values
              of arc (i.e. close to principal component)}
  \item{plotEach}{Plot curve after \code{plotEach} iterations.}
  
}

\seealso{
  \code{\link{cem}}
}

\author{
  Samuel Gerber
}
\references{
Samuel Gerber, Tolga Tasdizen, Ross Whitaker,
Dimensionality Reduction and Principal Surfaces via Kernel Map Manifolds, 
In Proceedings of the 2009 International Conference on Computer Vison (ICCV 2009).

Samuel Gerber and Ross Whitaker,
Conditional Expectation Curves, 
Submitted 2011.

}

\keyword{nonparametric,models,nonlinear}
