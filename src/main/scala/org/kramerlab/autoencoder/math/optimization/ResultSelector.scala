package org.kramerlab.autoencoder.math.optimization

import scala.Ordering.Implicits._
import scala.collection._

/**
 * Result selector takes a look at a sequence of intermediate results of some
 * optimization or search, and caches the best result it has seen so far.
 * The fitness of the intermediate results is measured by instances of type `F`.
 * Elements of `F` have to be ordered (usually, integers or double should be 
 * sufficient).
 */

class ResultSelector[X, F: Ordering](
  fitness: X => F,
  reevaluationInterval: Int,
  improvementPenalty: (F => F) = {x: F => x}
) {
  
  private var bestIntermediateResult: Option[X] = None
  private var bestFitness: Option[F] = None
  private var lastFitness: Option[F] = None
  private var startTime: Option[Long] = None
  protected var history: mutable.Map[Int, F] = new mutable.HashMap[Int, F]
  protected var currentStep: Int = 0
  protected var lastRecordStep: Int = -1 
  protected var lastImprovementStep: Int = -1 
  
  def consider(intermediateResult: => X): Unit = {
    if (startTime.isEmpty) {
      startTime = Some(System.currentTimeMillis)
    }
    
    if (currentStep % reevaluationInterval == 0) {
      val currentFitness = fitness(intermediateResult)
      
      // printf(
      //   "Fitness at step %d (after %d minutes): %f   last improvement: %d\n", 
      //   currentStep, 
      //   ((System.currentTimeMillis - startTime.get) / 60000).toInt,
      //   currentFitness,
      //   lastRecordStep
      // )
      
      history(currentStep) = currentFitness
      
      if (
        lastFitness.isEmpty || 
        lastFitness.get <= improvementPenalty(currentFitness)
      ) {
        lastImprovementStep = currentStep
      }
      lastFitness = Some(currentFitness)
      
      if (
        bestFitness.isEmpty || 
        bestFitness.get <= improvementPenalty(currentFitness)
      ) {
        bestFitness = Some(currentFitness)
        bestIntermediateResult = Some(intermediateResult)
        lastRecordStep = currentStep
      }
      
    }
    currentStep += 1
  }
  
  def result: X = {
    val winner = bestIntermediateResult.get
    // println("Returning the solution with fitness " + fitness(winner))
    winner
  }
  
  def getResult: Option[X] = bestIntermediateResult
  
  def trainingHistory = immutable.Map[Int, F]() ++ history
}