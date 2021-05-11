package nz.wicker.autoencoder.math

import scala.math.{exp => m_exp, _}

/**
 * Contains few methods for generation of random numbers with distributions 
 * that are not directly available in the `Random` class. 
 */
package object random {
  private val rnd = new scala.util.Random
  def bernoulli(p: Double) = rnd.nextDouble < p
  def unif(a: Double, b: Double) = rnd.nextDouble * (b - a) + a
  def unif(a: Int, b: Int) = rnd.nextInt(b - a + 1) + a
  def geom(a: Int, b: Int) = round(exp(a, b)).toInt
  def exp(a: Double, b: Double) = m_exp(unif(log(a), log(b)))
  def normal(m: Double, sigma: Double) = rnd.nextGaussian * sigma + m
  def permutation(size: Int) = {
    val result = (0 until size).toArray
    var r = 0
    val rnd = new java.util.Random
    while (r < size) {
      val otherIndex = r + rnd.nextInt(size - r)
      val tmp = result(r)
      result(r) = result(otherIndex)
      result(otherIndex) = tmp
      r += 1
    }
    result
  }
}