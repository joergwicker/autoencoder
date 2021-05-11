package nz.wicker.autoencoder.math.optimization

import scala.math.abs
import scala.math.max
import scala.math.min
import scala.math.sqrt

import nz.wicker.autoencoder.math.structure.VectorSpace
import nz.wicker.autoencoder.visualization.Observer

/**
 * Implementation of the conjugate gradient descent as described in the
 * article "A new conjugate gradient method wyth guaranteed descent and
 * an efficient line search" by William W. Hager. and Hongchao Zhang.
 * 
 * It uses Polak-Ribiere-Polyak-like update method for calculation of the 
 * next direction and inexact line search with approximate Wolfe conditions.
 * 
 * This implementation seems broke. It does not outperform
 * naive gradient descent on fairly simple 2D-functions.
 */
class ConjugateGradientDescent_HagerZhang(
  configuration: ConjugateGradientDescent_HagerZhangConfiguration
) extends Minimizer {
  
  import configuration._
  
  def minimize[V <: VectorSpace[V]](
    f: DifferentiableFunction[V], 
    start: V,
    progressObservers: List[Observer[V]] = Nil
  ): V = {
        
    // we formulate the algorithm tail-recursively
    def rec(
      currentPosition: V, 
      currentGradient: V, 
      currentDirection: V,
      remainingIterations: Int,
      history: History
    ): V = {
      if (remainingIterations <= 0) {
        // force the algorithm to terminate
        println("OPTI TERMINATED. Full history: " + history)
        currentPosition
      } else {
        // TODO: add some "good-enough"-conditions
        val (alpha, nextHistory) = lineSearch(
          f, 
          currentPosition, 
          currentDirection, 
          history
        )
        val nextPosition = currentPosition + currentDirection * alpha
        val nextGradient = f.grad(nextPosition)
        
        // helper variables that are used to calculate the next direction
        val gradientDifference = nextGradient - currentGradient
        val denominator = currentDirection dot gradientDifference
        val beta = 
          (gradientDifference - 
            currentDirection * 2 * gradientDifference.normSq / denominator
          ) dot nextGradient / denominator
        val bound = -1 / sqrt(
          currentDirection.normSq * min(eta * eta, currentGradient.normSq)
        )
        val boundedBeta = max(beta, bound)
        val nextDirection = - nextGradient + currentDirection * boundedBeta
        rec(
          nextPosition, 
          nextGradient, 
          nextDirection, 
          remainingIterations - 1, 
          nextHistory
        )
      }
    } 
    
    // call the recursive helper method with the start values
    val startGradient = f.grad(start);
    rec(start, startGradient, -startGradient, maxIters, new History())
  }
  
  /**
   * Sometimes it's possible to learn something about the problem while 
   * executing line searches, and to adjust some parameters. Such adjusted
   * parameters are returned by line search, and passed to the next line 
   * search.
   */
  protected case class History(
    bisectionSteps: List[Double] = Nil
  ) {
    /**
     * adds the most recent bisection step to the log, returns the
     * step together with new History
     */
    def withBisectionStep(step: Double): (Double, History) = {
      (step, History(step :: this.bisectionSteps))
    }     
    
    def proposeInitialBisectionStep: Double = {
      if (bisectionSteps.isEmpty) 1 else bisectionSteps.head 
    }
  }
  
  /**
   * Hybrid line search algorithm, which starts with the dumb & brutal
   * backtracking algorithm, and switches to a faster strategy if it
   * finds appropriate conditions.
   */
  protected def lineSearch[V <: VectorSpace[V]](
    f: DifferentiableFunction[V], 
    position: V, 
    direction: V,
    history: History
  ): (Double, History) = {
    
    // define the differentiable function phi(alpha) = f(pos + alpha * dir)
    val phi = new DifferentiableFunction[Double]() {
      
      // dirty work-around to avoid multiple evaluation on interval 
      // boundaries. One should rewrite the method signatures instead...
      private var cachedValues = List[(Double, Double, Double)]()
      
      override def valueAndGrad(alpha: Double): (Double, Double) = {
        for ((t, value, gradient) <- cachedValues) {
          if (t == alpha) {
            return (value, gradient)
          }
        }
        
        val timeStart = System.currentTimeMillis()
        val (value, fGrad) = f.valueAndGrad(position + direction * alpha)
        val grad = fGrad dot direction 
        val duration = System.currentTimeMillis() - timeStart
//        println("OPTI: " +
//          "eval phi(" + alpha + ") = " + value + " phi' = " +  grad + 
//          " [" + duration + "ms]")
        cachedValues ::= (alpha, value, grad)
        (value, grad)
      }
      override def apply(alpha: Double) = valueAndGrad(alpha)._1
      override def grad(alpha: Double) = valueAndGrad(alpha)._2
    }
    
    // evaluate the function at 0, these values are used often 
    val (phiValueAtZero, phiGradAtZero) = phi.valueAndGrad(0)
  
    // interval update
    // The notation in the paper is just gut-wrenching... 
    // weird non-obviously complementary if-conditions and goto's... :(((
    def intervalUpdate(a: Double, b: Double, c: Double): (Double, Double) = {
      val scaledEpsilon = epsilon * abs(phiValueAtZero) // \epsilon_k in paper
      val (phiValueAtC, phiGradAtC) = phi.valueAndGrad(c)
      if (c <= a || c >= b) {
        (a, b)
      } else {
        if (phiGradAtC >= 0) {
          (a, c)
        } else {
          if (phiValueAtC <= phiValueAtZero + scaledEpsilon) {
            (c, b)
          } else {
            def rec(a: Double, b: Double): (Double, Double) = {
              val d = (1 - theta) * a + theta * b
              val (phiValueAtD, phiGradAtD) = phi.valueAndGrad(d)
              if (phiGradAtD >= 0) {
                (a, d)
              } else {
                if (phiValueAtD <= phiValueAtZero + scaledEpsilon) 
                  rec(d, b)
                else
                  rec(a, d)
              }
            }
            rec(a, c)
          }
        }
      }
    }
    
    // secant function
    def secant(a: Double, phiGradAtA: Double, b: Double, phiGradAtB: Double) = {
      (a * phiGradAtB - b * phiGradAtA) / (phiGradAtB - phiGradAtA)
    }
    
    // double secant step
    def doubleSecant(a: Double, b: Double) = {
      val (phiValueAtA, phiGradAtA) = phi.valueAndGrad(a)
      val (phiValueAtB, phiGradAtB) = phi.valueAndGrad(b)
      val c = secant(a, phiValueAtA, b, phiValueAtB)
      val (nextA, nextB) = intervalUpdate(a, b, c)
      if (c != nextA && c != nextB) {
        (nextA, nextB)
      } else {
        val cBar = if (c == nextA) {
          val (phiValueAtNextA, phiGradAtNextA) = phi.valueAndGrad(nextA)
          secant(a, phiValueAtA, nextA, phiValueAtNextA)
        } else {
          val (phiValueAtNextB, phiGradAtNextB) = phi.valueAndGrad(nextA)
          secant(b, phiValueAtB, nextA, phiValueAtNextB)
        }
        intervalUpdate(nextA, nextB, cBar)
      }
    }
    
    // find an initial interval satisfying (4.4)
    // (could NOT find any algorithm for that anywhere
    //  I can't even understand why this interval is supposed to 
    //  exist [good luck searching this interval for exp(-x) starting at 0???])
    // val (a0, b0) = initialInterval(phi)
    
    // perform nesting interval updates until either the original or the
    // approximate Wolfe conditions are satisfied
    def fastLineSearch(
      start: Double,
      end: Double,
      remainingEvaluations: Int,
      history: History
    ): (Double, History) = {
      if (remainingEvaluations <= 0) {
        history.withBisectionStep((start + end) / 2)
      } else {
        val (phiValueAtStart, phiGradAtStart) = phi.valueAndGrad(start)
        val (phiValueAtEnd, phiGradAtEnd) = phi.valueAndGrad(end)
        if (wolfeConditions(start, phiValueAtStart, phiGradAtStart) || 
            approximateWolfeConditions(start, phiValueAtStart, phiGradAtStart)
        ) {
          history.withBisectionStep(start)
        } else if(wolfeConditions(end, phiValueAtEnd, phiGradAtEnd) || 
            approximateWolfeConditions(end, phiValueAtEnd, phiGradAtEnd)) {
          history.withBisectionStep(end)
        } else {
          val (nextStart, nextEnd) = {
            val (startCandidate, endCandidate) = doubleSecant(start, end)
            val candidateLength = startCandidate - endCandidate
            val length = start - end
            if (candidateLength > gamma * length) {
              intervalUpdate(
                startCandidate,
                endCandidate,
                (startCandidate + endCandidate) / 2
              )
            } else {
              (startCandidate, endCandidate)
            }
          }
          fastLineSearch(
            nextStart, 
            nextEnd, 
            remainingEvaluations - 1,
            history
          )
        }
      }
    }
  
    def wolfeConditions(
      alpha: Double,
      valueAtAlpha: Double,
      derivativeAtAlpha: Double
    ): Boolean = {
      (valueAtAlpha - phiValueAtZero <= phiGradAtZero * delta * alpha) &&
      (derivativeAtAlpha >= phiGradAtZero * sigma)
    }
    
    def approximateWolfeConditions(
      alpha: Double,
      valueAtAlpha: Double,
      derivativeAtAlpha: Double
    ): Boolean = {
      (valueAtAlpha <= phiValueAtZero + epsilon * abs(valueAtAlpha)) &&
      (phiGradAtZero * (2 * delta - 1) >= derivativeAtAlpha) &&
      (derivativeAtAlpha >= sigma * phiGradAtZero)
    }
    
    // start with a simpler algorithm, as soon as the prerequisites for
    // the more sophisticated algorithm are fulfilled, start the more
    // sophisticated algorithm.
    def brutalWolfeBisection(
      start: Double, 
      t: Double, 
      end: Double,
      remainingEvaluations: Int
    ): (Double, History) = {
      if (remainingEvaluations == 0) {
        // return the current value and terminate
        history.withBisectionStep(t)
      } else {
        // keep iterating
        val (phiValue, phiGrad) = phi.valueAndGrad(t)
        if (phiValue < phiValueAtZero + epsilon * abs(phiValueAtZero) && 
            phiGrad >= 0) {
            // launch the faster algorithm of Hager & Zhang
          fastLineSearch(start, t, remainingEvaluations, history)
        } else {
          // continue with the brutal bisection algorithm
          if (phiValue > phiValueAtZero + delta * t * phiGradAtZero) {
            brutalWolfeBisection(
              start, 
              (start + t) / 2, 
              t, 
              remainingEvaluations - 1
            )
          } else if (phiGrad < sigma * phiGradAtZero) {
            brutalWolfeBisection(
              t, 
              if (end.isInfinite()) 2 * t else (t + end) / 2,
              end,
              remainingEvaluations - 1
            )
          } else {
            history.withBisectionStep(t)
          }
        }
      }
    }
    
    brutalWolfeBisection(
      0, 
      history.proposeInitialBisectionStep, 
      Double.PositiveInfinity, 
      maxEvalsPerLineSearch
    )
  }
  
  /**
   * Finds an initial interval that satisfies condition (4.4)
   * 
   * 1) Why should there be something like that
   * 2) How am I supposed to find it?... 
   */
  private def initialInterval(
    f: DifferentiableFunction[Double]
  ): (Double, Double) = {
    (0, 1d/0d)
  }
  
}