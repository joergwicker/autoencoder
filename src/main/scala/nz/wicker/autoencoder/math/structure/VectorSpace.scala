package nz.wicker.autoencoder.math.structure

import math._

/**
 * Vector space structure with `Double` as field,
 * for everything that looks like Real^n for some
 * natural n. 
 *
 * We are aware of the fact that Real numbers is
 * not the only field, and we are also aware that
 * scalar product structure does not necessary 
 * exist for every vector space. Real^n is really 
 * all we need at the moment.
 */

trait VectorSpace[V <: VectorSpace[V]] {
  self: V =>
  def +(v: V): V
  def unary_- : V
  def zero: V
  def -(v: V): V = this + (-v)
  def *(d: Double): V
  def /(d: Double): V = this * (1/d)
  def dot(v: V): Double
  def normSq: Double = this.dot(this)
  def norm: Double = sqrt(normSq)
  def normalized = this / norm
  
  def isNaN: Boolean
  def isInfinite: Boolean
  def isInvalid = isNaN || isInfinite
}
