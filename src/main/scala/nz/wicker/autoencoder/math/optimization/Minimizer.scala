package nz.wicker.autoencoder.math.optimization

import nz.wicker.autoencoder.math.structure.VectorSpace
import nz.wicker.autoencoder.visualization.Observer

trait Minimizer {
  
  /**
   * Finds a local minimum for a differentiable function `f` using it's 
   * values and gradient.
   *
   */
  def minimize[V <: VectorSpace[V]](
    f: DifferentiableFunction[V], 
	start: V,
	progressObservers: List[Observer[V]]
  ): V
  
  def manimize[V <: VectorSpace[V]](
    f: DifferentiableFunction[V],
    start: V
  ): V = {
    minimize(f, start, Nil)
  }
  
  /**
   * If implemented, this minimization method can be used in order to
   * use the minimization process as a source of possible candidate solutions,
   * where the actual solution is not the one that minimizes `f`, but some 
   * other point, that maximizes some other fitness function. This can be very
   * useful for learning algorithms where the candidate solutions are tested
   * on a separate validation set in order to avoid overfitting. In general,
   * this method just throws UnsupportedOperationException. 
   */
  def minimize[V <: VectorSpace[V], Fitness: Ordering](
    f: DifferentiableFunction[V],
    start: V,
    terminationCriterion: TerminationCriterion[V, (Int, Int)],
    resultSelector: ResultSelector[V, Fitness],
    progressObservers: List[Observer[V]]
  ): V = throw new UnsupportedOperationException(
    "Not implemented. If you do not know exactly which parts of this library" +
    " you should use, better don't use it: it's just extremely crufty academic" +
    " code."
  )
}
