package nz.wicker.autoencoder.math.optimization

/**
 * Termination criterion that limits the number of line searches for CG-type
 * optimization algorithms. 
 * It assumes that the pair of integers is interpreted as 
 * `(numberOfLineSearches, numberOfFunctionEvaluations)`. 
 */
class LimitNumberOfEvaluations(maxEvaluations: Int) 
  extends TerminationCriterion[Any, (Int, Int)] {
  
  def apply(x: Any, lineSearchesEvals: (Int, Int)): Boolean = {
    lineSearchesEvals._2 > maxEvaluations
  }
}