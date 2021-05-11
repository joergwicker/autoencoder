package nz.wicker.autoencoder.neuralnet

import scala.math.{exp}

import nz.wicker.autoencoder.math.matrix.Mat

/**
 * Layer consisting of a single row of sigmoid units.
 */
class SigmoidUnitLayer(val scalingFactor: Double, biases: Mat)
  extends BiasedUnitLayer(biases) with Serializable {

  import SigmoidUnitLayer.{sigmoid, sigmoidDerivative}

  override def activation(x: Double) = sigmoid(scalingFactor * x)
  override def derivative(x: Double) = 
    scalingFactor * sigmoidDerivative(scalingFactor * x)

  override def build(newBiases: Mat): SigmoidUnitLayer = 
    new SigmoidUnitLayer(scalingFactor, newBiases)

  override def toString = 
    "SigmoidUnitLayer (scaling factor: " + scalingFactor + ")" +
	" with biases: " +
	parameters
}

object SigmoidUnitLayer {
  def sigmoid(x: Double) = 1d / (1 + exp(-x))
  def sigmoidDerivative(x: Double) = {
    val expMinusX = exp(-x)
	val sqrtDenom = 1 + expMinusX
	expMinusX / (sqrtDenom * sqrtDenom)
  }
}

/**
 * Little performance optimization: removing unnecessary multiplications with 1
 */
class UnscaledSigmoidUnitLayer(biases: Mat) 
  extends SigmoidUnitLayer(1, biases) with Serializable {
  
  def this() = this(Mat.empty(0,0,0))
  
  import SigmoidUnitLayer._
  override def activation(x: Double) = sigmoid(x)
  override def derivative(x: Double) = sigmoidDerivative(x)
  override def build(newBiases: Mat): SigmoidUnitLayer = 
    new UnscaledSigmoidUnitLayer(newBiases)
}