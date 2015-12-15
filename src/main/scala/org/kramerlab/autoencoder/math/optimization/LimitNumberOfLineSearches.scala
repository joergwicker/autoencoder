package org.kramerlab.autoencoder.math.optimization

case class LimitNumberOfLineSearches(maxLineSearches: Int) 
  extends TerminationCriterion[Any, (Int, Int)] {
  
  def apply(x: Any, lineSearchesEvals: (Int, Int)): Boolean = {
    lineSearchesEvals._1 > maxLineSearches
  }
}