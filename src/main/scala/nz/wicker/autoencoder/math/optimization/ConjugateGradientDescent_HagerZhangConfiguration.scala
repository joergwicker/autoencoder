package nz.wicker.autoencoder.math.optimization

case class ConjugateGradientDescent_HagerZhangConfiguration(
  maxIters: Integer = 64,
  maxEvalsPerLineSearch: Integer = 16,
  delta: Double = 0.001,     // parameter in Wolfe conditions     
  sigma: Double = 0.1,     // parameter in Wolfe conditions
  eta: Double = 0.01,       // used to bound beta
  epsilon: Double = 1e-9,   // used for termination criterion in approx. Wolfe
  theta: Double = 0.5,      // for case of opposite slope condition violation
  gamma: Double = 0.7       // determines when bisection is performed    
)