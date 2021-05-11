package nz.wicker.autoencoder.math.optimization

import nz.wicker.autoencoder.math.structure.VectorSpace
import nz.wicker.autoencoder.math.matrix.Mat

/**
 * Factory for error functions that can be applied to data with values in
 * [0, 1] and reconstructions with values in (0, 1). This error function 
 * has very strong reaction on bit-flips.
 */
object CrossEntropyErrorFunctionFactory 
  extends DifferentiableErrorFunctionFactory[Mat] {
  
  override def apply(
    target: Mat
  ): DifferentiableFunction[Mat] = {
    val targetComplement = target.map{x => 1 - x}
    new DifferentiableFunction[Mat] {
      override def apply(x: Mat): Double = {
        val xCompl = x.map{p => 1 - p}
        val xLog = x map math.log
        val xComplLog = xCompl map math.log
        -((target dot xLog) + (targetComplement dot xComplLog))
      }
      override def grad(x: Mat): Mat = {
        val xCompl = x.map{p => 1 - p}
        -((target :/ x) - (targetComplement :/ xCompl))
      }
      override def valueAndGrad(x: Mat): (Double, Mat) = {
        val xCompl = x.map{p => 1 - p}
        val xLog = x map math.log
        val xComplLog = xCompl map math.log
        (
          -((target dot xLog) + (targetComplement dot xComplLog)),
          -((target :/ x) - (targetComplement :/ xCompl))
        )
      }
    }
  }
}