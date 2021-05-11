package nz.wicker.autoencoder.math.optimization

/**
 * Version of nonlinear conjugate gradient descent based on the minimize.m
 * implementation of Dr. Carl Edward Rasmussen.
 */
class NonlinearConjugateGradientDescent_Rasmussen 
  extends CubicInterpolationLineSearch 
  with PolakRibiere 
  with SlopeRatioInitialStep