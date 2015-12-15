package org.kramerlab.autoencoder.math.optimization

/**
 * Differentiable `Double`-valued function suitable for line-search 
 * algorithms. The domain of the function is some vectorspace `V`, 
 * and the corresponding gradient is also from `V`.
 */
trait DifferentiableFunction[V] extends (V => Double) {
  /**
   * Returns the function value `f(x)` together with the gradient
   * `(grad f)(x)`
   */
  def apply(x: V): Double
  def grad(x: V): V
  def valueAndGrad(x: V): (Double, V) = (apply(x), grad(x))
}
