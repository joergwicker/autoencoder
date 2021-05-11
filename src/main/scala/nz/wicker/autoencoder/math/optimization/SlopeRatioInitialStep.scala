package nz.wicker.autoencoder.math.optimization

import scala.math.{min, max, abs}

trait SlopeRatioInitialStep extends NonlinearConjugateGradientDescent {
  override def initialStep(slope: Double): Double = 1 / (1 + abs(slope))
  override def initialStep( 
    previousSlope: Double, 
    currentSlope: Double,
    previousStep: Double
  ): Double = {
    val res = min(abs(previousSlope / currentSlope), 100)  * previousStep
    println("Proposing initial step: " + res)
    res
  }
}