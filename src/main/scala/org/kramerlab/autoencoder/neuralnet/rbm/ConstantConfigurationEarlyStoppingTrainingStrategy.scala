package org.kramerlab.autoencoder.neuralnet.rbm

import org.kramerlab.autoencoder.visualization.TrainingObserver
import org.kramerlab.autoencoder.math.matrix.Mat
import org.kramerlab.autoencoder.math.matrix._
import org.kramerlab.autoencoder.math.optimization.EarlyStopping

/**
 * Simple training strategy that just takes a fixed training configuration,
 * and trains the Rbm with it, until no improvement can be observed.
 */
case class ConstantConfigurationEarlyStoppingTrainingStrategy(
  config: RbmTrainingConfiguration,
  relativeValidationSetSize: Double
) extends RbmTrainingStrategy {
  def train(rbm: Rbm, data: Mat, trainingObservers: List[TrainingObserver]) = {
    
    val validationSetSize = (data.height * relativeValidationSetSize).toInt
    val trainingSetSize = data.height - validationSetSize
    val clonedData = data.clone
    clonedData.shuffleRows()
    val validationSet = clonedData(0 ::: validationSetSize, 0 ::: end)
    val trainingSet = clonedData(validationSetSize ::: end, 0 ::: end)
    
    // reconstruction after one step is compared to the input
    val negErrorOnValidationSet = { rbm: Rbm => 
      -(rbm.gibbsSampling(validationSet, 1, true)._1 - validationSet).l2NormSq 
    }
    
    val earlyStopping = new EarlyStopping(
      negErrorOnValidationSet,
      numberOfInitialSteps = 16,
      fitnessReevaluationInterval = 8,
      maxStepsWithoutNewRecord = 128,
      maxStepsWithoutImprovement = 64
    )
    
    rbm.train(
      data, 
      config, 
      trainingObservers, 
      earlyStopping, 
      earlyStopping
    )
  }
}