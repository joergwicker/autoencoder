package nz.wicker.autoencoder.neuralnet.rbm

import nz.wicker.autoencoder.visualization.TrainingObserver
import nz.wicker.autoencoder.math.optimization.ResultSelector
import nz.wicker.autoencoder.math.optimization.EarlyStopping
import nz.wicker.autoencoder.math.matrix.Mat
import nz.wicker.autoencoder.math.matrix._
import nz.wicker.autoencoder.experiments.Metaparameters

/**
 * This training strategy generates multiple Rbm training configurations at
 * random, and selects the Rbm which has best reconstruction error on a held 
 * out validation set.
 */
class RandomRetryTrainingStrategy(
  numRetries: Int,
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
    
    val rbmSelector = new ResultSelector(negErrorOnValidationSet, 1)
    
    for (t <- 0 until numRetries) {
      val earlyStopping = new EarlyStopping(
        negErrorOnValidationSet,
        numberOfInitialSteps = 16,
        fitnessReevaluationInterval = 8,
        maxStepsWithoutImprovement = 50,
        maxStepsWithoutNewRecord = 100
      )
      
      val randomConfig = 
        Metaparameters.createRandomRbmConfiguration(trainingSet.height)
      
      val newInitialRbm = rbm.reinitialize(randomConfig)
      
      val trainedRbm = newInitialRbm.train(
        data, 
        randomConfig, 
        trainingObservers, 
        earlyStopping, 
        earlyStopping
      )
      
      rbmSelector.consider{trainedRbm}
    }
    
    rbmSelector.result
  }
}