package org.kramerlab.autoencoder.math.optimization

import scala.Ordering.Implicits._

/**
 * This termination criterion first just waits until `numberOfInitialSteps`
 * have passed. After this initial period, it evaluates the fitness of the
 * current intermediate result every `fitnessReevaluationInterval` steps.
 * If at some point there was no improvement for at least 
 * `maxStepsWithoutImprovement`, the optimization is terminated, and the
 * fittest intermediate result found so far is returned.
 */
class EarlyStopping[X, V: Ordering](
  fitness: X => V,
  numberOfInitialSteps: Int,
  fitnessReevaluationInterval: Int,
  maxStepsWithoutNewRecord: Int,
  maxStepsWithoutImprovement: Int = Int.MaxValue,
  timeDependentLowerBound: Option[Int => Option[V]] = None,
  maxSteps: Int = Int.MaxValue, // ok, this class completely degenerated, it's frankensteined from ALL terimnation criteria we had previously...
  improvementPenalty: (V => V) = {x: V => x}
) extends ResultSelector[X, V](fitness, fitnessReevaluationInterval, improvementPenalty) 
  with TerminationCriterion[X, Any] {

  require(maxStepsWithoutNewRecord >= fitnessReevaluationInterval)
  
  def apply(currentResult: X, usedResources: Any): Boolean = {
    val initialIntervalExpired = currentStep > numberOfInitialSteps 
    val tooLongMonotonousDecrease = 
      currentStep - lastImprovementStep > maxStepsWithoutImprovement
    val tooLongNoNewRecords = 
      currentStep - lastRecordStep > maxStepsWithoutNewRecord
    val fallenBelowLowerBound = {
      if (history.isEmpty) {
        false
      } else {
        val (lastStep, lastFitness) = 
        (history.toList.sortBy{case (step, fitness) => step}).last
        (for {
          boundFunction <- timeDependentLowerBound
          currentBound <- boundFunction(lastStep)
        } yield {
          lastFitness < currentBound
        }).getOrElse(false)
      }
    } 
    val tooManySteps = currentStep > maxSteps
      
    val reasonMessage = if (tooLongMonotonousDecrease) {
      "Monotonous fitness decrease for over " + 
      maxStepsWithoutImprovement + 
      "steps"
    } else if(tooLongNoNewRecords) {
      "No new records for over " + maxStepsWithoutNewRecord + "steps"
    } else if(fallenBelowLowerBound) {
      "The fitness has fallen below lower bound"
    } else if(tooManySteps) {
      "Maximum number of steps used"
    }
      
    initialIntervalExpired && 
    (
      tooLongMonotonousDecrease ||
      tooLongNoNewRecords ||
      fallenBelowLowerBound ||
      tooManySteps
    ) && {
      // printf("Early stopping: terminating (%s)\n", reasonMessage); 
      true
    }
  }
}