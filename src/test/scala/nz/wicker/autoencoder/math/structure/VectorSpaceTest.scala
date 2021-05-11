package nz.wicker.autoencoder.math.structure

import org.scalatest.FlatSpec
import math._

class VectorSpaceTest extends FlatSpec {

  
  case class Point(x: Double, y: Double) extends VectorSpace[Point] {
    override def zero = Point(0d, 0d)
    override def +(other: Point) = Point(x + other.x, y + other.y)
	override def unary_- = Point(-x, -y)
	override def *(d: Double) = Point(x * d, y * d)
	override def dot(other: Point) = x * other.x + y * other.y
	
	override def isNaN = false
	override def isInfinite = false
	override def isInvalid = false
  }

  val x = Point(2,3)
  val y = Point(3,2)

  "A VectorSpace" must "be sufficiently easy to implement for `Point`" in {
  	assert(x + y === Point(5,5))
	assert(x - y === Point(-1,1))
	assert((x dot y) === 12)
	assert(x.normSq === 13)
	assert(x.norm === sqrt(13))
	assert(x * 2 === Point(4, 6))
	assert(x / 4 === Point(2d/4d, 3d/4d))
	assert(x.zero === x * 0)
  }

  it must "work well in user code, example: linear interpolation" in {
    def interpolate[V <: VectorSpace[V]](x: V, y: V, t: Double): V = {
	  x * (1 - t) + y * t
	}

	assert(interpolate(x,y,0) === x)
	assert(interpolate(x,y,1) === y)
	assert(interpolate(x,y,0.5) === (x + y) / 2d)
  }
}
