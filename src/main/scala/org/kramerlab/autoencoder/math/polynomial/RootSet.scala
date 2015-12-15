package org.kramerlab.autoencoder.math.polynomial

/**
 * This trait represents sets of real solutions of equations of type
 * p(x) = 0
 * where p is a polynomial with real coefficients 
 */
sealed trait RootSet

case class DiscreteRoots(roots: List[Double]) extends RootSet
object RealLine extends RootSet
