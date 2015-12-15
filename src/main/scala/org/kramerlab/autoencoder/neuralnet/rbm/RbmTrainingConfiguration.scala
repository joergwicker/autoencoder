package org.kramerlab.autoencoder.neuralnet.rbm

import org.kramerlab.autoencoder.math.matrix.Mat

/**
 * This class provides settings necessary for rbm training.
 * It stores information about number of epochs, dependence of
 * the momentum and number of steps of the gibbs sampling on the
 * current epoch, as well as functions that determine how big 
 * weights will be penalized.
 */
abstract class RbmTrainingConfiguration(
  val epochs: Int,
  val minibatchSize: Int,
  val learningRate: Double,
  val initialBiasScaling: Double,
  val initialWeightScaling: Double,
  val initialMomentum: Double,
  val finalMomentum: Double,
  val initialGibbsSamplingSteps: Int,
  val finalGibbsSamplingSteps: Int,
  val weightPenaltyFactor: Double,
  val sampleVisibleUnitsDeterministically: Boolean
) {
  def momentum(epoch: Int): Double 
  def gibbsSamplingSteps(epoch: Int): Int
  def weightPenalty(weights: Mat): Mat = weights * weightPenaltyFactor
}
 
