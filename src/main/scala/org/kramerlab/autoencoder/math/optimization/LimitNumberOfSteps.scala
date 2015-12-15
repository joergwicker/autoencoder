package org.kramerlab.autoencoder.math.optimization

class LimitNumberOfSteps(maxSteps: Int) extends TerminationCriterion[Any, Int] {
  def apply(x: Any, step: Int) = step >= maxSteps
}