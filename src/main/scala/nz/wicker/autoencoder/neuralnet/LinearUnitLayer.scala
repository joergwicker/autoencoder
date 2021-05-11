package nz.wicker.autoencoder.neuralnet

import nz.wicker.autoencoder.math.matrix.Mat

/**
 * Layer that does nothing but adding biases to the input
 */

class LinearUnitLayer(biases: Mat) extends BiasedUnitLayer(biases) {
  
  override def build(newBiases: Mat) = 
    new LinearUnitLayer(newBiases)

  override def activation(x: Double) = x
  override def activation(xs: Mat) = xs

  override def derivative(x: Double) = 1
  override def derivative(xs: Mat) = Mat.fill(xs.height, xs.width, 0d, 1d)
  
  override def toString = 
    "LinearUnitLayer with biases:" + parameters
}
