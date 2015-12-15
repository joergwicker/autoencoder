package org.kramerlab.autoencoder.math.optimization
import scala.math._
import org.kramerlab.autoencoder.math.matrix.Mat
import org.kramerlab.autoencoder.math.polynomial._

/**
 * Line search algorithm as described in Carl Edward Rasmussen's unpublished (?)
 * document "Function minimization using conjugate gradients: conj" 
 * (May 15 1996)
 */
abstract class CubicInterpolationLineSearch(
  val rho: Double = 0.25,
  val sigma: Double = 0.5
) extends NonlinearConjugateGradientDescent {
  
  override def simplifiedLineSearch(
    phi: DifferentiableFunction[Double], 
    maxEvals: Int,
    initialGuess: Double
  ): Double = {
    
    println("simplified line search...")
    val (startValue, startGrad) = phi.valueAndGrad(0)
    val absStartGrad = abs(startGrad)
    
    // this helper function takes the current bracketing interval (with all
    // the values and gradients available at the boundary points), 
    // as well as the upper bound on the left boundary of the "blacklisted" 
    // region, and moves the right point to the left
    // until neither the "slope-too-high" (a2) Wolfe condition, nor the
    // "decrease too small" (b) Wolfe conditions are violated.
    //
    // It uses the cubic and quadratic interpolation dependent on the
    // function values (if function values are too extreme, simpler and 
    // coarser quadratic interpolation is used, otherwise full information 
    // at 0 and current alpha is used by the cubic interpolation.
    // 
    // @returns 
    //   new right point (with value and gradient),
    //   upper bound for extrapolation,
    //   number of used evaluations
    def backtrack(
      leftPvg: PointValueGrad[Double],
      rightPvg: PointValueGrad[Double],
      extrapolationMaximum: Double, 
      usedEvals: Int,
      maxEvals: Int
    ): (PointValueGrad[Double], Double, Int) = {
      println(
        "backtrack(\n left = " + leftPvg + 
        "\n right = \n" + rightPvg + 
        "\n max = " + extrapolationMaximum
      )
      
      if (usedEvals == maxEvals) {
        val res = (rightPvg, extrapolationMaximum, maxEvals)
        println("No evals left, returning: " + res)
        res
      } else {
        
        val (leftAlpha, leftValue, leftGrad) = leftPvg
        val (rightAlpha, rightValue, rightGrad) = rightPvg
        println("Keep backtracking in the interval \n a =  \n" + leftPvg)
        println("\n b =  \n" + rightPvg)
        
        val intervalLength = rightAlpha - leftAlpha
        
        // check whether a2 or b are violated
        if (rightGrad > sigma * absStartGrad || 
            rightValue > leftValue + rightAlpha * rho * startGrad 
        ) {
          // move right interval boundary to the left 
          val nextRightAlpha = interpolate(leftPvg, rightPvg, startValue)
          val (nextRightValue, nextRightGrad) = phi.valueAndGrad(nextRightAlpha)
          // the right interval boundary was too far away: blacklist all
          // points beyond it, keep backtracking if necessary
          backtrack(
            leftPvg, 
            (nextRightAlpha, nextRightValue, nextRightGrad), 
            rightAlpha, // never look beyond the current right alpha
            usedEvals + 1,
            maxEvals
          )
        } else {
          // both conditions hold, return the right interval boundary (with all
          // the associated stuff), and the start of the "blacklisted" interval
          val res = (rightPvg, extrapolationMaximum, usedEvals)
          println("Powell Wolfe satisfied, returning: \n" + res)
          res
        }
      }
    }
    
    // This helper method takes a bracketing interval (with all the values
    // and gradients), as well as an extrapolation maximum,
    // and tries to shift the bracketing interval to the right as 
    // far as possible. It assumes that the right points satisfies (a2) and (b)
    // 
    // If the interval is shifted far enough, the right point is returned.
    // 
    // @return extrapolated right point, used number of evaluations
    def explore(
      leftPvg: PointValueGrad[Double],
      rightPvg: PointValueGrad[Double],
      extrapolationMaximum: Double, 
      usedEvals: Int,
      maxEvals: Int
    ): (Double, Int) = {
      println(
        "explore \n left = " + leftPvg + "\n right = " + rightPvg + " max = " + 
        extrapolationMaximum
      )
      
      val (rightAlpha, rightValue, rightGrad) = rightPvg
      
      // check if the right point satisfies the 
      // "slope-not-too-small" (a1) Wolfe condition
      if (usedEvals == maxEvals || rightGrad > -absStartGrad) {
        (rightPvg._1, usedEvals)
      } else {
        // push the right boundary further to the right
        val extrRightAlpha = extrapolate(
          leftPvg, 
          rightPvg, 
          extrapolationMaximum
        )
        
        val (extrRightValue, extrRightGrad) = phi.valueAndGrad(extrRightAlpha)
        val extrRightPvg = (extrRightAlpha, extrRightValue, extrRightGrad)
        
        // make sure we do not go too far
        val (nextRightPvg, newMax, newUsedEvals) = backtrack(
          rightPvg, 
          extrRightPvg, 
          extrapolationMaximum,
          usedEvals + 1,
          maxEvals
        )
        
        // keep exploring further
        explore(
          rightPvg, 
          nextRightPvg, 
          extrapolationMaximum, 
          newUsedEvals, 
          maxEvals
        )
      }
    }

    // make one step using the initial guess, backtrack, and launch the
    // exploration
    val startPvg = (0d, startValue, startGrad)
    val (initialGuessValue, initialGuessGrad) = phi.valueAndGrad(initialGuess)
    val initialPvg = (initialGuess, initialGuessValue, initialGuessGrad) 
    val (
      firstBacktrackedRight, 
      firstExtrapolationMax, 
      firstBacktrackEvals
    ) = backtrack(
      startPvg,
      initialPvg,
      Double.PositiveInfinity,
      1,
      maxEvals
    )
    
    explore(
      startPvg, 
      firstBacktrackedRight, 
      firstExtrapolationMax, 
      firstBacktrackEvals,
      maxEvals
    )._1
  }
  
  private def extrapolate(
    leftPvg: PointValueGrad[Double], 
    rightPvg: PointValueGrad[Double], 
    extrapolationMaximum: Double
  ): Double = {
    val (a, fa, ga) = leftPvg
    val (b, fb, gb) = rightPvg
    val length = b - a
    
    // extrapolate with cubic polynomial
    val aSq = a * a
    val bSq = b * b
    val vandermonde = Mat(4, 4)(
      1, a, aSq, aSq * a,
      1, b, bSq, bSq * b,
      0, 1, 2 * a, 3 * aSq,
      0, 1, 2 * b, 3 * bSq
    )
    val rightSide = Mat(4, 1)(fa, fb, ga, gb)
    val coeffs = vandermonde \ rightSide
    
    // try to find minimum
    val minimum = cubicMinimum(coeffs(3), coeffs(2), coeffs(1), coeffs(0))
    
    // TODO: compare it to the original again, he formulated it somewhat weird.
    // 3x interval length from what?...
    val extrapolationPoint = minimum match {
      case Some(x) => {
        max(
          min(min(extrapolationMaximum, a + length * 3), x),
          a + length / 10
        )
      }
      case None => min(a + 3 * length, extrapolationMaximum)
    }
    
    extrapolationPoint
  }
     
  def interpolate(
    leftPvg: PointValueGrad[Double],
    rightPvg: PointValueGrad[Double],
    valueAtZero: Double
  ): Double = {
    val (a, fa, ga) = leftPvg
    val (b, fb, gb) = rightPvg
    val minimum = if (fb > valueAtZero) {
      
      // the value seems weird, use quadratic interpolation
      val vandermonde = Mat(3, 3)(
        1, a, a * a,
        1, b, b * b,
        0, 1, 2 * a
      )
      val rightSide = Mat(3, 1)(fa, fb, ga)
      val coeffs = vandermonde \ rightSide
      
      // try to find minimum of the parabola
      quadraticMinimum(coeffs(2), coeffs(1), coeffs(0))
      
    } else {
      
      // use all the available information, fit a cubic
      val aSq = a * a
      val bSq = b * b
      val vandermonde = Mat(4, 4)(
        1, a, aSq, aSq * a,
        1, b, bSq, bSq * b,
        0, 1, 2 * a, 3 * aSq,
        0, 1, 2 * b, 3 * bSq
      )
      val rightSide = Mat(4, 1)(fa, fb, ga, gb)
      val coeffs = vandermonde \ rightSide
      
      // attempt to find minimum of the cubic
      cubicMinimum(coeffs(3), coeffs(2), coeffs(1), coeffs(0))
    }
    
    // if no minimum found, use bisection
    val innerPoint = minimum match {
      case Some(x) => x
      case None => (a + b) / 2
    }
    
    // make sure the point is not too close to the boundary
    val offset = (b - a) / 10
    
    // return
    min(max(innerPoint, a + offset), b - offset)
  }
}