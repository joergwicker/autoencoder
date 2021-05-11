package nz.wicker.autoencoder.math.optimization

import scala.Ordering.Implicits._

/**
 * Trivial termination criterion that is never fulfilled.
 * This means, that the optimization runs forever, unless the optimization 
 * algorithm itself stops for some reason.
 */
object TerminateNever extends TerminationCriterion[Any, Any] {
  def apply(x: Any, r: Any) = false
}