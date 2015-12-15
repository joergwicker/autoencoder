package org.kramerlab.autoencoder.math.optimization

import org.scalatest.FlatSpec
import org.kramerlab.autoencoder.math.matrix.Mat

class CGTest extends FlatSpec {
  
  "A Conjugate Gradient method" must 
    "be able to find minimum on simple paraboloids" in {
    
    val f = new DifferentiableFunction[Mat]() {
      private val a = Mat(2, 2)(2, 0, 1, 3)
      private val g = a.transpose + a
      
      override def apply(x: Mat) = (x.transpose * a * x)(0, 0)
      override def grad(x: Mat) = g * x 
    }
    
    val cg = new ConjugateGradientDescent_HagerZhang(
      new ConjugateGradientDescent_HagerZhangConfiguration(
        maxIters = 100, 
        maxEvalsPerLineSearch = 32)
    )
    
    val min = cg.minimize(f, Mat(2, 1)(1, 1))
    
    assert(min.l2Norm < 1E-10)
  }
}