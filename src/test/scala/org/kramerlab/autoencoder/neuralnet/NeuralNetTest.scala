package org.kramerlab.autoencoder.neuralnet

import org.scalatest.FlatSpec
import org.scalatest.matchers.MustMatchers
import org.kramerlab.autoencoder.CustomMatchers
import org.kramerlab.autoencoder.math.optimization.SquareErrorFunctionFactory
import scala.util.Random
import org.kramerlab.autoencoder.math.matrix.Mat

class NeuralNetTest 
  extends FlatSpec 
  with MustMatchers 
  with CustomMatchers {

  "NeuralNetLike nets" must 
  "calculate feed-forward and backpropagation correctly" in {
   
    val rnd = new Random(0)
    def runif(left: Double, right: Double) = {
	  rnd.nextDouble * (right - left) + left
	}
	
	import SigmoidUnitLayer.{sigmoid => sigma}

    // builds a small but nontrivial neural net with 4 layers 
	// from a set of randomly chosen variables
    // and compares the gradient calculated by the backpropagation
	// with a symbolically evaluated gradient
 
    for (i <- 1 to 10) {
	  
	  // initialize parameters
	  val b1 = runif(-5,5)         // weights of the topmost linear unit layer
	  val b2 = runif(-3,3)         
	  val b3 = runif(-3,2)         // weights of the second sigmoid unit layer
	  val b4 = runif(-4,8)
	  val w1 = runif(-1,1)         // weights between sigmoid and linear layer
	  val w2 = runif(-0.2, 0.3)
	  val w3 = runif(-0.1, 0.1)
	  val w4 = runif(-0.7, 0.7)
	  val x1 = runif(-2, 2)        // input variables
	  val x2 = runif(-1, 1)
	  val t1 = runif(0, 1)         // target values
	  val t0 = runif(0, 1)
      
	  // build the network
      val layer0 = new LinearUnitLayer(Mat(1, 2)(0d, 0d))
	  val insp01 = new InspectionLayer
	  val layer1 = new SigmoidUnitLayer(1d, Mat(1, 2)(b3, b4))
	  val insp12 = new InspectionLayer
	  val layer2 = new FullBipartiteConnection(Mat(2, 2)(w1, w2, w3, w4))
	  val insp23 = new InspectionLayer
	  val layer3 = new LinearUnitLayer(Mat(1, 2)(b1, b2))
	  val insp3 = new InspectionLayer
	  val net = new NeuralNet(
	    List(
		  layer0,
		  insp01,
		  layer1,
		  insp12,
		  layer2,
		  insp23,
		  layer3,
		  insp3
		)
	  )
      
	  // define the error function and the input
	  val errFct = SquareErrorFunctionFactory(Mat(1, 2)(t0, t1))
      val input = Mat(1, 2)(x1, x2)
	  val target = Mat(1, 2)(t0, t1)

	  // calculate the value and the gradients symbolically
	  val o10 = sigma(x1 + b3)   // output of the second layer (sigmoid)
	  val o11 = sigma(x2 + b4)
	  val o20 = w1 * o10 + w3 * o11    // output after the third layer
	  val o21 = w2 * o10 + w4 * o11
	  val o30 = b1 + o20               // output of the fourth layer
	  val o31 = b2 + o21
	  val expectedOutput: Mat = Mat(1, 2)(o30, o31)

	  val d0 = o30 - t0                // differences of outputs and targets
	  val d1 = o31 - t1
	  val err = (d0 * d0 + d1 * d1) / 2    // expected error
      
	  val g10 = d0 * w1 + d1 * w2
	  val g11 = d0 * w3 + d1 * w4

	  val visibleGrad = new LinearUnitLayer(Mat(1, 2)(0d, 0d))
	  val sigmoidGrad = new SigmoidUnitLayer(1d, 
	    Mat(1, 2)(g10 * o10 * (1 - o10), g11 * o11 * (1 - o11))
      )
	  val connectionGrad = new FullBipartiteConnection(
	    Mat(2, 2)(d0 * o10, d1 * o10, d0 * o11, d1 * o11)
	  )
	  val linearGrad = new LinearUnitLayer(Mat(1, 2)(d0, d1))
	  val netGrad = new NeuralNet(
	    List(
		  visibleGrad,
		  new InspectionLayer,
		  sigmoidGrad, 
		  new InspectionLayer,
		  connectionGrad,
		  new InspectionLayer,
		  linearGrad,
		  new InspectionLayer
		)
	  )

	  // calculate the output values
	  val actualOutput = net(input)

	  // create the composition of fixed input and clamped error 
	  // function, calculate it's neural-net-valued gradient
	  import NeuralNetLike.{differentiableComposition, ParameterVector}

	  val f = differentiableComposition[NeuralNet](
	    input, SquareErrorFunctionFactory(target)
	  )
      
	  // wrap the net into an ad-hoc vector structure
	  val x = ParameterVector(net)

	  // calculate error and gradients with the backprop algorithm
	  val (actualError, actualGrad) = f.valueAndGrad(x)
      
	  // compare the results
	  insp01.cachedSignal must have (sameEntries(input))

      actualOutput must be (closeTo(expectedOutput))
	  actualError must be ((err) plusOrMinus 1e-9)
	  actualGrad must be (closeTo(ParameterVector(netGrad)))
	}
  }

  "NeuralNets" must "correctly map their input to the output" in {
    val layer0 = new SigmoidUnitLayer(1d, Mat(1, 5)(1, -1, 2, -2, 5))
	val layer1 = new FullBipartiteConnection(Mat.fill(5, 4){(i,j) => i - j})
	val layer2 = new LinearUnitLayer(Mat(1, 4)(-3, 2, 4, 7))

	val net = new NeuralNet(
	  List(
	    layer0,
		layer1,
		layer2
      )
	)

	val input = Mat(1, 5)(5, 6, 7, -19, 78)
	net(input) must have (sameEntries(layer2.propagate(layer1.propagate(input))))
  }
}




























