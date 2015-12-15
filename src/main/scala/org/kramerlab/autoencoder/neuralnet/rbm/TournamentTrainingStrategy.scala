package org.kramerlab.autoencoder.neuralnet.rbm

import org.kramerlab.autoencoder.visualization.TrainingObserver
import org.kramerlab.autoencoder.math.optimization.ResultSelector
import org.kramerlab.autoencoder.math.optimization.EarlyStopping
import org.kramerlab.autoencoder.math.matrix.Mat
import org.kramerlab.autoencoder.math.matrix._
import org.kramerlab.autoencoder.experiments.Metaparameters
import org.kramerlab.autoencoder.math.optimization.ResultSelector
import scala.math.{min, max}

/**
 * This training strategy generates multiple Rbm training configurations at
 * random, and then trains multiple Rbm's with these configurations in rounds 
 * with specified number of epochs. After each round, a fraction of best Rbm's
 * is selected, and the training continues with the next round, until there is
 * only one Rbm left. This rbm is trained until the error on the validation set
 * does not decrease any more. 
 */
class TournamentTrainingStrategy(
  initialNumberOfCandidates: Int,
  epochsPerRound: Int,
  fraction: Double,
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
    
    // create random configurations
    val startConfigs = List.fill(initialNumberOfCandidates){
      Metaparameters.createRandomRbmConfiguration(trainingSet.height)
    } 
    
    // create Rbm's with these configurations
    val candidates = startConfigs.map{ config => 
      (config, rbm.reinitialize(config))
    }
    
    // define the selective process
    def select(configsAndRbms: List[(RbmTrainingConfiguration, Rbm)]): 
      (RbmTrainingConfiguration, Rbm) = {
      
      if (configsAndRbms.size == 1) {
        configsAndRbms.head
      } else {
        // train all candidate rbm's with their configurations 
        // for specified number of epochs
        val nextGenerationCandidates = 
          for ((config, rbm) <- configsAndRbms) yield {
            val earlyStopping = new EarlyStopping(
              negErrorOnValidationSet,
              numberOfInitialSteps = 0,
              fitnessReevaluationInterval = 2,
              maxStepsWithoutImprovement = 8,
              maxStepsWithoutNewRecord = 16,
              maxSteps = epochsPerRound
            )  
            val nextRbm = rbm.train(
              trainingSet, 
              config, 
              trainingObservers, 
              earlyStopping,
              earlyStopping
            )
            (config, nextRbm)
          }
        
        // calculate the fitness of each new candidate
        val nextGenWithFitness: List[(RbmTrainingConfiguration, Rbm, Double)] = 
          nextGenerationCandidates.map{ c =>
            (c._1, c._2, negErrorOnValidationSet(c._2))
          }
        
        // take the specified fraction of candidates into next round
        val n = nextGenWithFitness.size 
        val newNumberOfCandidates = max(1, min((fraction * n).toInt, n - 1))
        
        val nextCandidatesWithFitness = 
          nextGenWithFitness.sortBy{_._3}.reverse.take(newNumberOfCandidates)
      
        val nextCandidates = nextCandidatesWithFitness.map{x => (x._1, x._2)}
        select(nextCandidates)
      }
    }
    
    // start the selection process
    val bestConfigAndRbm = select(candidates)
    
    // now train the best rbm to the bitter end
    val earlyStopping = new EarlyStopping(
      negErrorOnValidationSet,
      numberOfInitialSteps = 0,
      fitnessReevaluationInterval = 2,
      maxStepsWithoutImprovement = 8,
      maxStepsWithoutNewRecord = 16,
      improvementPenalty = {x: Double => x - 0.01 * math.abs(x)}
    )  
    
    bestConfigAndRbm._2.train(
      trainingSet, 
      bestConfigAndRbm._1, 
      trainingObservers, 
      earlyStopping,
      earlyStopping
    )
  }
}
