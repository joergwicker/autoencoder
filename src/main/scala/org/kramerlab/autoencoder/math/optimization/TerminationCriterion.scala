package org.kramerlab.autoencoder.math.optimization

/**
 * A fairly generic termination criterion that takes a look at the
 * intermediate result achieved and resources used so far, and decides if an
 * optimization procedure or some kind of search should be terminated.
 */
trait TerminationCriterion[-IntermediateResult, -UsedResources] 
  extends ((IntermediateResult, UsedResources) => Boolean) { 
  
  self =>
  
  /**
   * Terminates when this or the other termination criterion is fulfilled
   */
  def | [X <: IntermediateResult, R <: UsedResources](
    other: TerminationCriterion[X, R]
  ): TerminationCriterion[X, R] = {
    new TerminationCriterion[X, R] {
      override def apply(x: X, r: R) = { 
        self(x, r) | other(x, r)
      }
    }
  }
  
  /**
   * Terminates when this and the other termination criteria are fulfilled
   */
  def & [X <: IntermediateResult, R <: UsedResources](
    other: TerminationCriterion[X, R]
  ): TerminationCriterion[X, R] = {
    new TerminationCriterion[X, R] {
      override def apply(x: X, r: R) = { 
        self(x, r) & other(x, r)
      }
    }
  }
}