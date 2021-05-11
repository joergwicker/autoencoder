package nz.wicker.autoencoder

import org.scalatest.matchers.{
  MatchResult, 
  BeMatcher, 
  HavePropertyMatcher, 
  HavePropertyMatchResult
}
import nz.wicker.autoencoder.math.structure.VectorSpace
import nz.wicker.autoencoder.math.matrix.Mat

trait CustomMatchers {

  case class NormMatcher[V <: VectorSpace[V]](expected: V, tolerance: Double) 
    extends BeMatcher[V] {

	override def apply(is: V) = {
	  val dist = (is - expected).norm
	  MatchResult(
	    dist < tolerance,
	    is + " was too far away from " + expected +
	    "(distance = " + 
	    dist + " > " + tolerance + " = tolerance)",
	    is + " was close enough to " + expected +
	    "(distance = " + dist + " < " + 
	    tolerance + " = tolerance)"
	  )
    }
  }
 
  def closeTo[V<:VectorSpace[V]](expected: V, tolerance: Double = 1e-9) = 
    new NormMatcher[V](expected, tolerance)
  
  case class EntryWiseMatcher(expected: Mat) 
    extends HavePropertyMatcher[Mat, Mat] {
    
	override def apply(actual: Mat) = {
	  HavePropertyMatchResult[Mat](
	    actual.sameEntries(expected), 
		"entries", 
		expected, 
		actual
	  )
	}
  }

  def sameEntries(m: Mat) = EntryWiseMatcher(m)
}
