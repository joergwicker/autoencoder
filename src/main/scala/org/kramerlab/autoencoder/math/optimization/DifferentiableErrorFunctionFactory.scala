package org.kramerlab.autoencoder.math.optimization

import org.kramerlab.autoencoder.math.structure.VectorSpace

/**
 * Curried error function, which gets a target value
 * and returns a differentiable function that represents 
 * some sort of error between the input and the target value.
 * 
 * To be able to speak about differentiability, we need some
 * underlying vector space V. To be able to speak about 
 * errors, we need some metric there. In this version, 
 * we simply require that all subclasses of the error 
 * function work with everything that looks like Real^n.
 */
trait DifferentiableErrorFunctionFactory[X <: VectorSpace[X]] {
  /**
   * Takes the target value as input and returns a 
   * differentiable function which represents the error
   * betweenthe input and the target value.
   */
  def apply(targetValue: X): DifferentiableFunction[X]
}
