package org.kramerlab.autoencoder.neuralnet

import org.kramerlab.autoencoder.math.matrix.Mat

/**
 * A mock layer that does nothing but propagating the 
 * signals and errors unchanged.
 * 
 * It caches everything it propagates, so that the
 * signals can be inspected in the tests.
 */
case class InspectionLayer(
  var cachedSignal: Mat = null,
  var cachedBackpropagatedError: Mat = null,
  var cachedReverseSignal: Mat = null
) extends MatrixParameterizedLayer(Mat(0,0)()) {

  override def propagate(signal: Mat): Mat = {
    cachedSignal = signal
	signal
  }

  override def reversePropagate(signal: Mat): Mat = {
    cachedReverseSignal = signal
    signal
  }

  override def gradAndBackpropagationError(backpropagatedError: Mat) = {
    cachedBackpropagatedError = backpropagatedError
	(this, backpropagatedError)
  }

  override def build(params: Mat) = new InspectionLayer

  override def toString = 
    "InspectionLayer" + 
	"\ncachedSignal:" + String.valueOf(cachedSignal) +
	"\ncachedBackpropagatedError:" + String.valueOf(cachedBackpropagatedError) +
	"\ncachedReverseSignal:" + String.valueOf(cachedReverseSignal)
	
  override def reverseLayer = this
  
  def inputDimension = 0
  def outputDimension = 0
}
