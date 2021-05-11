package nz.wicker.autoencoder.math

import math._

package object polynomial {
  
  /**
   * Solves ax2 + bx + c = 0
   */
  def quadraticRoots(a: Double, b: Double, c: Double): RootSet = {
    if (a == 0) {
      if (b == 0) {
        if (c == 0) {
          RealLine
        } else {
          DiscreteRoots(Nil)
        }
      } else {
        DiscreteRoots(List(-c / b))
      }
    } else {
      val pHalf = b / (2 * a)
      val q = c / a
      val discr = pHalf * pHalf - q
      if (discr < 0) {
        DiscreteRoots(Nil) // no real solutions
      } else if (discr == 0) {
        DiscreteRoots(List(-pHalf))
      } else {
        val pm = sqrt(discr)
        DiscreteRoots(List(-pHalf + pm, -pHalf - pm))
      }
    }
  }
  
  /**
   * Attempts to find local minimum of a quadratic polynomial
   * p(x) = ax2 + bx + c
   */
  def quadraticMinimum(a: Double, b: Double, c: Double): Option[Double] = {
    if (a <= 0) {
      None
    } else {
      Some(-b / (2 * a))
    }
  }
  
  /**
   * Finds a minimum of a polynomial of third degree, if possible.
   * There is at most one, so we use an Option as return type.
   * 
   * The coefficients are specified as follows:
   * p(x) = ax3 + bx2 + cx + d
   * 
   * See second red numerics notebook p. 48 for all the funny 
   * signum-manipulations.
   */
  def cubicMinimum(
    a: Double, b: Double, c: Double, d: Double
  ): Option[Double] = {
    if (a == 0) {
      quadraticMinimum(b, c, d)
    } else {
      val discr = b * b - 3 * a * c
      if (discr <= 0) {
        None
      } else {
        Some((-b + signum(a) * sqrt(discr)) / (3 * a)) 
      }
    }
  }
}