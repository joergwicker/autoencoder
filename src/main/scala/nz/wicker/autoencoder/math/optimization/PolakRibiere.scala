package nz.wicker.autoencoder.math.optimization

import nz.wicker.autoencoder.math.structure.VectorSpace

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