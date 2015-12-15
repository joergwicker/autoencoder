package org.kramerlab.autoencoder.math.optimization

import scala.collection._
import scala.math._
import org.kramerlab.autoencoder.math.structure.VectorSpace
import org.kramerlab.autoencoder.visualization.Observer

/**
 * Generic nonlinear conjugate gradient method for optimization of 
 * nonlinear twice differentiable functions which are bounded from below.
 *  
 * The choice of search direction, the concrete implementation of line search
 * and the termination criterion are left abstract.   
 */
abstract class NonlinearConjugateGradientDescent(
  maxLineSearches: Int = 256,
  maxFunctionEvaluations: Int = 256,
  maxEvaluationsPerLineSearch: Int = 16
) extends Minimizer {

  /**
   * We always want to keep a point together with the corresponding function
   * value and gradient. This is because re-evaluation of function `f` can
   * be prohibitively expensive, and a point by itself is useless: we always 
   * need to know something about `f` at this point in order to do something
   * useful with it. These three things should be like quarks: never in 
   * isolation. The entry start point is the only exception. 
   */
  protected type PointValueGrad[V] = (V, Double, V)
  
  /**
   * Convenience method for evaluation of `f`. As long as we use this method
   * to evaluate `f`, we can be sure that we keep "pvg-triples" together, and
   * aren't loosing any valuable information.
   */
  def pointValueGrad[V](f: DifferentiableFunction[V], point: V) = {
    val (value, grad) = f.valueAndGrad(point)
    (point, value, grad)
  } 
    
  /**
   * Attempts to minimize the differentiable function `f` starting at `point` in
   * the given `direction`. The function `f` should not be evaluated more than
   * `maxEvals` times. This method returns the approximate minimum lying on
   * the specified ray, and the actual number of evaluations needed.
   *
   * This method provides a default implementation, which makes use of the 
   * simplified line search. This implementation is, however, pretty wasteful,
   * since it calculates function values and gradients multiple times. For 
   * better performance, it is advisable to set simplifiedLineSearch to `???`
   * and override this method directly. It would allow to control which 
   * values are cached, and which values are discarded.  
   * 
   * Guideline: use simplifiedLineSearch to quickly prototype and to compare
   * different line search methods. Once it's decided which method is the best,
   * implement it properly, taking care to calculate everything just once.
   * 
   * @param f twice differentiable function bounded from below
   * @param pvg point, function value, and function gradient
   * @param direction search direction
   * @param maxEvals maximum number of evaluations of function `f`
   * @return 
   *   next solution approximation, value and gradient at this point,
   *   alpha used for the line search, 
   *   actual number of function evaluations
   */ 
  protected def lineSearch[V <: VectorSpace[V]](
    f: DifferentiableFunction[V],
    currentPvg: PointValueGrad[V],
    direction: V,
    initialAlpha: Double,
    maxEvals: Int
  ): (PointValueGrad[V], Double, Int) = {
    
    // if someone should ever read this: this default implementation is not
    // terribly important for the understanding of the algorithm. It's just 
    // a wasteful convenience method, that will be overridden in subclasses.
    
    // This is the restriction to line that we want to 
    // minimize with line search in each step:
    // phi(t) := f(x + t * d) 
    // 
    // where x is a point and d is the search direction.
    //
    // We use memoization strategy and cache values and derivatives, 
    // so that we don't have to care about it while performing line search.
    // The memory overhead for memoization is negligible, since both values
    // and gradients are just doubles.
    def phi = { 
      new DifferentiableFunction[Double] {
        private val cachedValues = new mutable.HashMap[Double, (Double, Double)]
        override def valueAndGrad(t: Double): (Double, Double) = {
          cachedValues.getOrElseUpdate(t, {
            val (value, fGrad) = f.valueAndGrad(currentPvg._1 + direction * t)
            (value, fGrad.dot(direction))
          })
        }
        override def apply(t: Double) = valueAndGrad(t)._1
        override def grad(t: Double) = valueAndGrad(t)._2
        
        // beware: that's actually a dirty hack, this value indeed depends on
        // the history of the object, it changes over time. But this allows us
        // to avoid returning the actual number of evaluations from the 
        // simplified line search: we traded interface clumsiness for some
        // controlled local mutablity
        def numberOfFunctionEvaluations = cachedValues.size
      }
    }
    
    val alpha = simplifiedLineSearch(phi, maxEvals, initialAlpha)
    (
      pointValueGrad(f, currentPvg._1 + direction * alpha),
      alpha,
      phi.numberOfFunctionEvaluations
    )
  }
  
  protected def simplifiedLineSearch(
    phi: DifferentiableFunction[Double], 
    maxEvals: Int,
    initialAlpha: Double
  ): Double
  
  def searchDirectionBeta[V <: VectorSpace[V]](
    previousSearchDirection: V, 
    previousGrad: V,
    currentGrad: V
  ): Double
  
  def terminationCriterion(currentValue: Double): Boolean = {
    abs(currentValue) < 1E-20
  }
  
  def initialStep(currentSlope: Double): Double
  
  def initialStep(
    previousSlope: Double, 
    currentSlope: Double,
    previousStep: Double
  ): Double
  
  override def minimize[V <: VectorSpace[V]](
    f: DifferentiableFunction[V], 
    startPoint: V,
    progressObservers: List[Observer[V]] = Nil
  ): V = {

    // the main recursion that runs until the termination
    // criterion is satisfied. Because we want to avoid unnecessary
    // re-evaluation of the function `f`, we always pass all
    // available information (point, value, gradient) from one recursion 
    // step to the next. Furthermore, the previous search direction is
    // required by all conjugate gradient algorithms (otherwise we get just
    // a gradient descent). Finally, we keep track of the number of line
    // searches and function evaluations: the counters are updated 
    // accordingly from step to step.
    // Please notice that this recursion never evaluates `f` itself: 
    // everything is done by the line search. This is critical in order to
    // avoid multiple evaluation.
    def searchMinimum(
      previousPvg: PointValueGrad[V],
      currentPvg: PointValueGrad[V], // this is to avoid re-evaluation
      previousSearchDirection: V,
      previousSlope: Double,
      previousAlpha: Double,
      remainingLineSearches: Int,
      remainingFunctionEvaluations: Int
    ): V = {
      println("search minimum...")
      val (currentPoint, currentValue, currentGrad) = currentPvg
      if (
        remainingLineSearches == 0 || 
        remainingFunctionEvaluations == 0 || 
        terminationCriterion(currentValue)
      ) {
        currentPoint
      } else {
        
        val beta = searchDirectionBeta(
          previousSearchDirection, 
          previousPvg._3,
          currentPvg._3
        )
        
        val currentSearchDirection = 
          -currentGrad + previousSearchDirection * beta
        val maxEvals = 
          min(remainingFunctionEvaluations, maxEvaluationsPerLineSearch)
        
        val currentSlope = currentSearchDirection dot currentGrad
        val initialAlpha = initialStep(
          previousSlope, 
          currentSlope,
          previousAlpha
        )
        
        val (nextPvg, currentAlpha, evals) = lineSearch(
          f,          
          currentPvg,
          currentSearchDirection,
          initialAlpha,
          maxEvals
        )
        
        searchMinimum(
          currentPvg,
          nextPvg, 
          currentSearchDirection,
          currentSlope,
          currentAlpha,
          remainingLineSearches - 1,
          remainingFunctionEvaluations - evals
        )
      }
    }
    
    // now all parts are in place, start the machinery
    val startPvg = pointValueGrad(f, startPoint)
    val startSearchDirection = -startPvg._3
    val startSlope = -startSearchDirection.normSq
    val initialAlpha = initialStep(startSlope)
    val maxEvals = min(maxFunctionEvaluations, maxEvaluationsPerLineSearch)
    val (firstPvg, startAlpha, evals) = lineSearch(
      f, 
      startPvg, 
      startSearchDirection, 
      initialAlpha,
      maxEvals
    )
    
    searchMinimum(
      startPvg,
      firstPvg, 
      startSearchDirection,
      startSlope,
      startAlpha,
      maxLineSearches, 
      maxFunctionEvaluations - evals
    )
  }
}