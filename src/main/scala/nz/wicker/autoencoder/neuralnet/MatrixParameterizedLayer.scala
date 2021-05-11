package nz.wicker.autoencoder.neuralnet

import nz.wicker.autoencoder.math.matrix.Mat
import nz.wicker.autoencoder.visualization._

/**
 * This is a base class for all layers that 
 * are parameterized by a rectangular array of
 * double values. It implements the vector-space
 * structure for layers, simply ignoring the precise
 * type of the `other` argument in most cases: for 
 * all binary operations, it simply checks that the 
 * `other` argument is also parameterized by a matrix,
 * combines the other matrix with this matrix, and 
 * augments it with structure of this Layer, simply
 * discarding the structure of the other Layer, which
 * is acceptable, if one keeps in mind that the 
 * `other` Layer should have been produced as output 
 * by this Layer in the backpropagation step.
 * 
 * All one has to do in the subclasses is to override
 * `build` method, which takes the `Mat`-valued 
 * parameters, and creates `Layer` of same type as 
 * `this`.
 */

abstract class MatrixParameterizedLayer(val parameters: Mat) 
  extends Layer with Serializable {
  def build(newParameters: Mat): MatrixParameterizedLayer

  private def throwIncompatibleLayerTypesError(other: Layer): Nothing = {
    throw new Error("Incompatible layer types: attempt to combine " + 
	  "Layers of type " + MatrixParameterizedLayer.this.getClass + 
	  " and " + other.getClass + "; " +
	  "This should never occur in backpropagation algorithm, the neural " +
	  "net implementation must have wrecked the structure somewhere.")
  }

  private def cast(other: Layer): MatrixParameterizedLayer = other match {
    case o: MatrixParameterizedLayer => o
	case _ => throwIncompatibleLayerTypesError(other)
  }

  override def +(other: Layer) = build(cast(other).parameters + parameters)
  override def -(other: Layer) = build(parameters - cast(other).parameters)
  override def *(d: Double) = build(parameters * d)
  override def /(d: Double) = build(parameters / d)
  override def unary_- = build(-parameters)
  override def zero = build(parameters.zero)
  override def dot(other: Layer) = cast(other).parameters.dot(parameters)
  override def isNaN = parameters.isNaN
  override def isInfinite = parameters.isInfinite
  override def isInvalid = parameters.isInvalid
  override def toImage = draw(parameters)
}
