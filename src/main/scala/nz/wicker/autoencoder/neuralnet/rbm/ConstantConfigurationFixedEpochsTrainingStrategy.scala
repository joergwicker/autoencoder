package nz.wicker.autoencoder.neuralnet.rbm

import nz.wicker.autoencoder.visualization.TrainingObserver
import nz.wicker.autoencoder.math.matrix.Mat
import nz.wicker.autoencoder.math.matrix._
import nz.wicker.autoencoder.math.optimization.ResultSelector
import nz.wicker.autoencoder.math.optimization.LimitNumberOfSteps

/**
 * Dumbest imaginable training strategy: it just takes an RBM and an
 * RBM training configuration, and does exactly what's told in the 
 * configuration, without monitoring the progress in any way.
 */
case class ConstantConfigurationFixedEpochsTrainingStrategy(
  config: RbmTrainingConfiguration
) extends RbmTrainingStrategy {
  def train(rbm: Rbm, data: Mat, trainingObservers: List[TrainingObserver]) = {
    
    val clonedData = data.clone
    clonedData.shuffleRows()
    
    // fitness is very simple: the latter, the better
    val timeAsFitness = { rbm: Rbm => System.currentTimeMillis }
    
    val resultSelector = new ResultSelector[Rbm, Long] (timeAsFitness, 1)
    
    rbm.train(
      data, 
      config, 
      trainingObservers, 
      new LimitNumberOfSteps(config.epochs), 
      resultSelector
    )
  }
}