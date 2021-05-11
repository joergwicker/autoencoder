package nz.wicker.autoencoder.neuralnet.rbm

import nz.wicker.autoencoder.math.matrix.Mat
import nz.wicker.autoencoder.neuralnet.SigmoidUnitLayer.{sigmoid}

/**
 * Implements a reasonable training strategy for `Rbm`s.
 *
 * Momentum and number of Gibbs-sampling steps are interpolated from their 
 * initial values to their final values by a sigmoid curve which is centered 
 * at the specified center and scaled with the specified factor (larger factor
 * means that the change occurs more rapidly, smaller factor means that the
 * change occurs more gradually)
 */
class DefaultRbmTrainingConfiguration(
  epochs: Int = 128,
  minibatchSize: Int = 8,
  learningRate: Double = 0.01,
  initialBiasScaling: Double = 0.01,
  initialWeightScaling: Double  = 0.001,
  initialMomentum: Double = 0.5,
  finalMomentum: Double = 0.875,
  initialGibbsSamplingSteps: Int = 1,
  finalGibbsSamplingSteps: Int = 2,
  sampleVisibleUnitsDeterministically: Boolean = false,
  weightPenaltyFactor: Double = 0.00001, // to 0.00001, Hinton told.
  val gibbsSamplingStepsSteepness: Double = 1,
  val sigmoidScalingFactor: Double = 1,
  val interpolationCenter: Double = 0.5
) extends RbmTrainingConfiguration(
  epochs,
  minibatchSize,
  learningRate,
  initialBiasScaling,
  initialWeightScaling,
  initialMomentum,
  finalMomentum,
  initialGibbsSamplingSteps,
  finalGibbsSamplingSteps,
  weightPenaltyFactor,
  sampleVisibleUnitsDeterministically
) {
  
  def momentum(epoch: Int): Double = 
    interpolate(epoch, initialMomentum, finalMomentum)
  
  def gibbsSamplingSteps(epoch: Int): Int = {
    interpolate(
      epoch,
      initialGibbsSamplingSteps,
      finalGibbsSamplingSteps
    ).round.toInt
  }
  
  private def interpolate(epoch: Int, x: Double, y: Double): Double = {
    val t = sigmoid(
      (epoch - interpolationCenter).toDouble / epochs * sigmoidScalingFactor
    )
    x * t + y * (1 - t)
  }
  
  override def toString = {
    "DefaultRbmConfig(\n" +
    "  rate:    " + learningRate + "\n" +  
    "  penalty: " + weightPenaltyFactor + "\n" + 
    "  epochs:  " + epochs + "\n" +
    "  initialMomentum: " + initialMomentum + "\n" + 
    "  finalMomentum:   " + finalMomentum + "\n" +
    "  minibatch size: " + minibatchSize + "\n" +
    ")" 
  }
}
