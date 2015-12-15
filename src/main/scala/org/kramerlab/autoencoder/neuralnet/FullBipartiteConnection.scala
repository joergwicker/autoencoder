package org.kramerlab.autoencoder.neuralnet

import org.kramerlab.autoencoder.math.matrix.Mat
import java.util.Random

/**
 * Full bipartite graph between two layers with edges labeled by matrix entries
 * of the `weight` matrix. The `weight` matrix has the `width` equal to the
 * dimension of the output, and 'height' equal to the dimension of the input.
 */
class FullBipartiteConnection(weights: Mat) 
  extends MatrixParameterizedLayer(weights) with Serializable {

  def this(inputDim: Int, outputDim: Int) = {
    this(
      Mat.fill(inputDim, outputDim) {
        case (x, y) => new Random().nextGaussian() * 0.1
      }
    )
  }
  
  override def build(newParams: Mat): FullBipartiteConnection = 
    new FullBipartiteConnection(newParams)

  override def propagate(input: Mat): Mat = {
    cachedInput = input
	input * parameters
  }
  override def reversePropagate(output: Mat) = output * parameters.transpose

  protected var cachedInput: Mat = null
  override def gradAndBackpropagationError(
    backpropagatedError: Mat
  ): (FullBipartiteConnection, Mat) = {
    (build(cachedInput.transpose * backpropagatedError),
	 reversePropagate(backpropagatedError))
  }

  override def toString = 
    "FullBipartiteConnection (input dimension: " + parameters.height +
	", output dimension: " + parameters.width + ")" + parameters
	
  def reverseLayer = new FullBipartiteConnection(parameters.transpose)
  
  def copy = new FullBipartiteConnection(weights.clone)
  
  def reinitialize(weightScale: Double): FullBipartiteConnection = {
    val rnd = new Random
    val newWeights = Mat.fill(weights.height, weights.width) { case (x, y) => 
      rnd.nextGaussian * weightScale
    }
    new FullBipartiteConnection(newWeights)
  }
  
  val inputDimension = weights.height
  val outputDimension = weights.width
}
