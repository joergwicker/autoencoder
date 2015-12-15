package org.kramerlab.autoencoder.neuralnet.rbm

import org.kramerlab.autoencoder.visualization.TrainingObserver
import org.kramerlab.autoencoder.math.optimization.ResultSelector
import org.kramerlab.autoencoder.math.optimization.EarlyStopping
import org.kramerlab.autoencoder.math.matrix.Mat
import org.kramerlab.autoencoder.math.matrix._
import org.kramerlab.autoencoder.experiments.Metaparameters

/**
 * This training strategy generates multiple Rbm training configurations at
 * random, and selects the Rbm which has best reconstruction error on a held 
 * out validation set. This strategy differs from the 
 * RandomRetryTraininStrategy in that the training of a new random RBM can
 * be interrupted much earlier, if it the progress of the currently trained
 * Rbm does not look very promising compared to the best Rbm seen so far.
 */
class CompetitiveRetryTrainingStrategy(
  numRetries: Int,
  relativeValidationSetSize: Double,
  acceptableErrorMultiplier: Double
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
    
    val rbmSelector = new ResultSelector[(Rbm, Map[Int, Double]), Double](
      {case (rbm, history) => -(history.values.min)}, 
      1
    )
    
    for (t <- 0 until numRetries) {
      val earlyStopping = new EarlyStopping(
        negErrorOnValidationSet,
        numberOfInitialSteps = 16,
        fitnessReevaluationInterval = 4,
        maxStepsWithoutImprovement = 16,
        maxStepsWithoutNewRecord = 32,
        timeDependentLowerBound = rbmSelector.getResult.map { 
          case (rbm, history) => 
          { (step: Int) => 
            history.get(step).map{
              _ * acceptableErrorMultiplier
            }
          }
        }
      )
      
      val randomConfig = 
        Metaparameters.createRandomRbmConfiguration(trainingSet.height)
      
      val newInitialRbm = rbm.reinitialize(randomConfig)
      
      val trainedRbm = newInitialRbm.train(
        trainingSet, 
        randomConfig, 
        trainingObservers, 
        earlyStopping,
        earlyStopping
      )
      
      rbmSelector.consider{(trainedRbm, earlyStopping.trainingHistory)}
    }
    
    rbmSelector.result._1
  }
}