package org.kramerlab.autoencoder.neuralnet

import org.kramerlab.autoencoder.math.matrix.Mat

/**
 * Abstract layer representing a single row of units with a differentiable
 * activation function and some biases (one bias value for each unit).
 *
 * One only has to override `activation` and `derivative` methods, as well
 * as the abstract methods inherited from `Layer` (`reparameterized`),
 * everything else is already implemented.
 */
abstract class BiasedUnitLayer(biases: Mat) 
  extends MatrixParameterizedLayer(biases) with Serializable {

  /**
   * Activation function of the neurons
   */
  def activation(d: Double): Double
  def activation(ds: Mat): Mat = ds.map{x => activation(x)}

  /**
   * Calculates the derivative of the activation function
   */
  def derivative(d: Double): Double
  def derivative(ds: Mat): Mat = ds.map{x => derivative(x)}

  protected var cachedInputPlusBias: Mat = null
  
  /**
   * Adds biases to each row of the input and applies the 
   * activation function pointwise.
   *
   * Caches the input matrix with added biases in `cachedInputPlusBias`
   */
  override def propagate(input: Mat): Mat = {
    cachedInputPlusBias = input + Mat.ones(input.height, 1) * parameters
    activation(cachedInputPlusBias)
  }

  /**
   * Does exactly the same as the `propagate` method.
   */
  override def reversePropagate(output: Mat): Mat = propagate(output)

  /**
   * Calculates the gradient by pointwise multiplying the 
   * backpropagated error 
   * passed from above with the pointwise application of the `derivative`
   * function to the `cachedInputPlusBias`, and summing the rows.
   * The matrix obtained before summing the rows is the new backpropagated
   * error.
   */
  def gradAndBackpropagationError(backpropagatedError: Mat): 
    (MatrixParameterizedLayer, Mat) =  {

    val derivatives = derivative(cachedInputPlusBias)
	val nextBackpropagatedError = backpropagatedError :* derivatives
	val grad = nextBackpropagatedError.sumRows
	(build(grad), nextBackpropagatedError)
  }
  
  /**
   * The reversal is trivial, just 
   */
  def reverseLayer = build(parameters.clone)
  
  val inputDimension = biases.width
  val outputDimension = biases.width
}
