package org.kramerlab.autoencoder.neuralnet

import org.scalatest.FlatSpec
import org.kramerlab.autoencoder.math.matrix.Mat

class FullBipartiteConnectionTest extends FlatSpec {
  
  val signal = Mat(2,3)(
    1d, 2d, 3d,
	5d, 8d, 9d
  )
  val error = Mat(2,2)(
    1d, 2d,
	9d, 6d
  )

  val weights = Mat(3,2)(
    1d, 1d,
    1d, 1d,
    0d, 2d
  )

  def connection = new FullBipartiteConnection(weights)

  "A FullBipartiteConnection" must 
  "propagate the signal by multiplying the signal row vector with " +
  "the weight matrix from the left" in {
	 val c = connection
	 val propagated = c.propagate(signal)
	 assert(propagated.sameEntries(signal * weights))
  }

  it must "propagate the signal in reverse direction by multiplying it " +
  "from the left with the adjoint of the weight matrix" in {
	val reversePropagated = connection.reversePropagate(error)
	assert(reversePropagated.sameEntries(error * weights.transpose))
  }

  it must "backpropagate the error in the same way as usual signal, " + 
  "the gradient with respect to weights shall be calculated as transposed input times the error" in {
	val c = connection
    val propagated = c.propagate(signal)
	val (grad, backprop) = c.gradAndBackpropagationError(error)
	assert(grad.parameters.size === c.parameters.size)
	assert(backprop.sameEntries(connection.reversePropagate(error)))
	assert(grad.parameters.sameEntries(signal.transpose * error), 
	  grad.parameters + " should be the same as " + signal.transpose * error)
  } 
}
