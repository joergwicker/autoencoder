package nz.wicker.autoencoder.math.optimization

import nz.wicker.autoencoder.math.structure.VectorSpace
import nz.wicker.autoencoder.visualization.Observer

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
