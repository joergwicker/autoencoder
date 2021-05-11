package nz.wicker.autoencoder.neuralnet

import org.scalatest.FlatSpec
import org.scalatest.matchers.MustMatchers
import nz.wicker.autoencoder.math.matrix.Mat
import scala.math.{E, exp, pow}
import nz.wicker.autoencoder.CustomMatchers

class SigmoidUnitLayerTest 
  extends FlatSpec 
  with MustMatchers 
  with CustomMatchers {
  
  val signal = Mat(2,3)(
    1d, 2d, 3d,
	5d, 8d, 9d
  )
  val error = Mat(2,3)(
    1d, 2d, 7d,
	9d, 6d, 3d
  )

  val biases = Mat(1,3)(
    1d, 1d, -1d
  )

  def sigmoidLayer = new SigmoidUnitLayer(1d, biases)

  "A SigmoidUnitLayer" must 
  "by default work with the unscaled, centered sigmoid function " +
  "that takes values between 0 and 1" in {
     import SigmoidUnitLayer._
	 sigmoid(0) must be (0.5)
	 sigmoid(Double.NegativeInfinity) must be (0d)
	 sigmoid(Double.PositiveInfinity) must be (1d)
	 sigmoid(1) must be ((1d/(1d + 1d/E)) plusOrMinus 1e-8)
	 sigmoidDerivative(0) must be ((0.25) plusOrMinus 1e-8)
  }

  it must "handle the scaling factor correctly" in {
    val s = new SigmoidUnitLayer(2d, biases)
	s.activation(0) must be ((0.5) plusOrMinus 1e-8)
	s.derivative(0) must be ((0.5) plusOrMinus 1e-8)
	s.derivative(1) must be ((2 * exp(-2) / pow(1 + exp(-2), 2)) plusOrMinus 1e-8)
	for (x <- -10 to 10) {
	  s.activation(x) must be ((sigmoidLayer.activation(2 * x)) plusOrMinus 1e-8)
	  s.derivative(x) must be ((2 * sigmoidLayer.derivative(2 * x)) plusOrMinus 1e-8)
	}
  }

  it must "handle the biases correctly" in {
    val s = sigmoidLayer
	s.propagate(-biases).asInstanceOf[Mat] must be (closeTo(Mat.fill(1,3,0d,0.5).asInstanceOf[Mat]))
  }
  
  it must "calculate the reverse propagation in same way as forward propagation" in {
    val s = sigmoidLayer
    s.reversePropagate(signal) must be (closeTo(s.propagate(signal)))
  }

  it must "calculate the gradient and backpropagated error correctly" in {
    import SigmoidUnitLayer._
    val s = sigmoidLayer
	s.propagate(signal)
	val (grad, bpErr) = s.gradAndBackpropagationError(error)

	val expectedBpErr: Mat = 
	  error :* ((Mat.ones(2,1) * biases + signal).map{x => 
	  sigmoid(x) * (1 - sigmoid(x))
	}) // see blue book page 271

	bpErr must be (closeTo(expectedBpErr))
	grad.parameters.asInstanceOf[Mat] must be (closeTo((Mat.ones(1,2) * expectedBpErr).asInstanceOf[Mat]))
  }
}
