\name{cem.example.sr}
\alias{cem.example.sr}
\title{Conditional Expectation Manifold Example on Swissroll}
\description{
This function runs the swissroll example in:

Samuel Gerber and Ross Whitaker,
Conditional Expectation Curves, 
Submitted 2011.


}


\usage{
cem.example.sr(n =1000, nstd=0.5, init=0, type=2)
}

\arguments{
  \item{n}{ Sample size.}
  \item{nstd}{ Amount of normal distributed noise orthogonal to the swissroll.}
  \item{type}{ Optimization objective}
  \item{init}{Type of initialization. 0 = ground truth, 1 = random, 2 = y-values
              of arc (i.e. close to principal component)}
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
