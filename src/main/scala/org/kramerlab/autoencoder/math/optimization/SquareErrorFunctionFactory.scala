package org.kramerlab.autoencoder.math.optimization

import org.kramerlab.autoencoder.math.structure.VectorSpace
import org.kramerlab.autoencoder.math.matrix.Mat

/**
 * Returns half of the square of the norm 
 * induced by the scalar products structure on
 * Real^n. The normalization factor 1/2 is there so that the 
 * residual corresponds to the gradient.
 */
object SquareErrorFunctionFactory 
  extends DifferentiableErrorFunctionFactory[Mat] {
  
  override def apply(
    target: Mat
  ): DifferentiableFunction[Mat] = {
    new DifferentiableFunction[Mat] {
	  override def apply(x: Mat): Double = {
	    val diff = x - target
		diff.normSq / 2
	  }
	  override def grad(x: Mat): Mat = {
	    val diff = x - target
		diff
      }
	  override def valueAndGrad(x: Mat): (Double, Mat) = {
	    val diff = x - target
		(diff.normSq / 2, diff)
	  }
	}
  }
}
