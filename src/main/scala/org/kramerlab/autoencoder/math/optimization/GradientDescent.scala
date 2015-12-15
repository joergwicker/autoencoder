package org.kramerlab.autoencoder.math.optimization

import org.kramerlab.autoencoder.math.structure.VectorSpace
import org.kramerlab.autoencoder.visualization.Observer

case class GradientDescent(val maxIters: Int) extends Minimizer {
  override def minimize[V <: VectorSpace[V]](
    f: DifferentiableFunction[V], 
	start: V,
    progressObservers: List[Observer[V]]
  ) = {
	/*TODO*/
    // question: what for?...
	start
  }
}
