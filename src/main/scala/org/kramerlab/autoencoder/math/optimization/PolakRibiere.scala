package org.kramerlab.autoencoder.math.optimization

import org.kramerlab.autoencoder.math.structure.VectorSpace

trait PolakRibiere extends NonlinearConjugateGradientDescent {
  override def searchDirectionBeta[V <: VectorSpace[V]](
    previousSearchDirection: V,
    previousGradient: V,
    currentGradient: V
  ): Double = {
    currentGradient.dot(currentGradient - previousGradient) / 
    previousGradient.normSq
  }
}