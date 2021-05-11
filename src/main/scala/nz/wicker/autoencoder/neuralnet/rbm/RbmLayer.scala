package nz.wicker.autoencoder.neuralnet.rbm

import nz.wicker.autoencoder.math.matrix.Mat
import nz.wicker.autoencoder.visualization._
import nz.wicker.autoencoder.neuralnet.{
  Layer, 
  LinearUnitLayer, 
  SigmoidUnitLayer
}
import scala.util.Random
import nz.wicker.autoencoder.neuralnet.UnscaledSigmoidUnitLayer

sealed trait RbmLayer extends Layer with Cloneable with Serializable {
  
  def parameters: Mat
  
  override def toImage = draw(parameters)
  
  def sample(activation: Double): Double
  def sample(activation: Mat): Mat = activation.map{x => sample(x)}
  def copy: RbmLayer
  def reinitialize(biasScaling: Double): RbmLayer
}

object RbmLayer {
  protected[rbm] val rnd = new Random
}

class GaussianUnitLayer(biases: Mat) extends LinearUnitLayer(biases) 
  with RbmLayer with Serializable {

  import RbmLayer._
  
  def this(dimension: Int) = this(
    Mat.fill(1, dimension){
      case (x: Int, y: Int) => new Random().nextGaussian * 0.001
    })
  
  override def sample(activation: Double) = rnd.nextGaussian + activation

  override def build(biases: Mat) = new GaussianUnitLayer(biases)
  override def copy = new GaussianUnitLayer(biases.clone)
  def reinitialize(biasScaling: Double): RbmLayer = {
    val rnd = new Random
    val newBiases = Mat.fill(1, biases.width) { case (x, y) => 
      rnd.nextGaussian * biasScaling
    }
    new GaussianUnitLayer(newBiases)
  }
}

class BernoulliUnitLayer(biases: Mat) 
  extends UnscaledSigmoidUnitLayer(biases) with RbmLayer with Serializable {

  import RbmLayer._
  
  def this(dimension: Int) = this(
    Mat.fill(1, dimension){
      case (x: Int, y: Int) => new Random().nextGaussian * 0.01
    }
  )
  
  def this() = this(0)
  
  override def sample(activation: Double) = {
    if (rnd.nextDouble < activation) 1d else 0d
  }

  override def build(biases: Mat) = new BernoulliUnitLayer(biases)
  override def copy = new BernoulliUnitLayer(biases.clone)
  
  def reinitialize(biasScaling: Double): RbmLayer = {
    val rnd = new Random
    val newBiases = Mat.fill(1, biases.width) { case (x, y) => 
      rnd.nextGaussian * biasScaling
    }
    new BernoulliUnitLayer(newBiases)
  }
}