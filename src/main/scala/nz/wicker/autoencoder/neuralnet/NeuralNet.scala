package nz.wicker.autoencoder.neuralnet

import nz.wicker.autoencoder.math.matrix.Mat
import nz.wicker.autoencoder.math.optimization._
import nz.wicker.autoencoder.math.structure.VectorSpace

/**
 * Base trait for all neural networks. 
 * 
 */
class NeuralNet(override val layers: List[Layer]) 
  extends NeuralNetLike[NeuralNet] with Serializable {

  override def build(layers: List[Layer]): NeuralNet = new NeuralNet(layers)

  protected def layersToString(layers: List[Layer]) = {
    layers.reverse.map{_.toString}.mkString("\n")
  }

  override def toString = 
    "NeuralNet (" + layers.size + " layers)" + layersToString(layers)
    
}
