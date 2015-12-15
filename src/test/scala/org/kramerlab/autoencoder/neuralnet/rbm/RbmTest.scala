package org.kramerlab.autoencoder.neuralnet.rbm

import org.scalatest.FlatSpec
import org.scalatest.matchers.MustMatchers
import org.kramerlab.autoencoder.CustomMatchers

import org.kramerlab.autoencoder.math.matrix.Mat
import org.kramerlab.autoencoder.neuralnet._

class RbmTest 
  extends FlatSpec
  with MustMatchers
  with CustomMatchers {

  "Rbm" must 
  "calculate the activations of the hidden units exactly" +
  " given the visible units in each step of Gibbs sampling" in {
    val input = Mat(1,5)(1,2,3,4,5)
    val visibleBiases = Mat(1,5)(1,2,1,2,1)
	val hiddenBiases = Mat(1,3)(-3, -2, 1)
	val visibleLayer = new BernoulliUnitLayer(visibleBiases)
	val hiddenLayer = new BernoulliUnitLayer(hiddenBiases)
	val connection = new FullBipartiteConnection(
	  Mat(5,3)(
	    0.1, 0.2, 0.1,
		0.2, 0.1, 0.2,
		-0.1, 0.2, -0.1,
		0.1, -0.2, 0.1,
		-0.2, 0.1, -0.2
	  )
	)

	val rbm = new Rbm(
	  visibleLayer,
	  connection,
	  hiddenLayer
	)
    
	val (confabulation, hiddenActivation) = rbm.gibbsSampling(input, 0)

	confabulation must have (sameEntries(input))
	hiddenActivation must be (closeTo(
	  hiddenLayer.propagate(connection.propagate(input))
	))

  }
}
