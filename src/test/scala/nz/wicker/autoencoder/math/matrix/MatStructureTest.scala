package nz.wicker.autoencoder.math.matrix

import org.scalatest.FlatSpec

/**
 * Test suite for the minimalistic `Mat` matrix class.
 * The `Mat` class itself is supposed to be used for
 * tests of more sophisticated matrix frameworks, the
 * purpose of the tests here is to ensure that the
 * fundament for the matrix testing itself is solid.
 * 
 * This suite covers only the functionality of `Mat` as
 * data structure, a mere rectangle with entries.
 * 
 * At no point does one actually require that the
 * entries are double, because no double-specific
 * operations are used.
 */

class MatStructureTest extends FlatSpec {

  "A Mat" must "be instantiable" in {
	val mat3 = Mat.empty(5,5)
	val mat4 = Mat.empty(5,5,1d)
	val mat5 = Mat(2,2)(1,2,3,4)
	val mat6 = Mat(2,2,1d)(1,2,3,4)
  }
  
  it must "have the right dimensions after instantiation" in {
    val mat = Mat.empty(42, 39)
	assert(mat.height === 42)
	assert(mat.width === 39)
	assert(mat.size === (42, 39))
  }

  it must "provide a usable foreachWithIndex method" in {
    val mat = Mat.empty(2,2)
	val arr = Array.ofDim[Double](2,2)
	mat foreachWithIndex { (r,c,d) =>
	  arr(r)(c) = r + 3 * c
	}
	assert(arr(0)(0) === 0)
	assert(arr(0)(1) === 3)
	assert(arr(1)(0) === 1)
	assert(arr(1)(1) === 4)
  }

  it must "provide a usable foreach method" in {
    val mat = Mat.empty(42, 39)
	var counter = 0
	for (x <- mat) counter += 1
	assert(counter === 42 * 39)
  }
  
  it must "provide a usable forall method" in {
    val mat = Mat.empty(42, 39)
	mat(23, 32) = 12
	mat(12, 19) = 15
	assert(mat.forall {x => (x >= 0 && x < 20)})
	assert(!mat.forall {x => (x < 14) })
  }

  it must "provide an exists method" in {
    val mat = Mat.empty(42, 39)
	mat(23,32) = 10
	mat(7, 9) = -11
	assert(mat.exists{_ < 0})
	assert(!mat.exists{_ > 100})
  }

  it must "store the assigned entries" in {
    val mat = Mat.empty(10, 10)
	for (r <- 0 until mat.height; c <- 0 until mat.width) {
	  mat(r, c) = r * c
	}
    for (r <- 0 until mat.height; c <- 0 until mat.width) {
	  assert(mat(r, c) === r * c)
	}
  }

  it must "be empty if instantiated with only dimension specified" in {
    for (h <- 0 to 3; w <- 0 to 3) {
      val mat = Mat.empty(h, w)

	  // check that it's empty
	  for (r <- 0 until h; c <- 0 until w) {
	    assert(mat(r, c) === 0)
      }
	}
  }

  it must "have be filled with defaultValues if specified" in  {
    val mat = Mat.empty(10, 10, 42d)
	mat foreach { x => assert(x === 42d) }
  }

  it should "have a readable toString" in {
    val normalSquareMat = Mat(2,2)(1,2,3,4)
	var expectedString =
      "\n _       _ \n| 1.0 2.0 |\n| 3.0 4.0 |\n -       - \n"

    assert(normalSquareMat.toString === expectedString)

	val degenerateZeroColumnMat = Mat(2,0)()
	expectedString = "\n __ \n|  |\n|  |\n -- \n"

    assert(degenerateZeroColumnMat.toString === expectedString)

	val degenerateZeroLineMat = Mat(0,3)()
	expectedString ="\n _     _ \n -     - \n"

    assert(degenerateZeroLineMat.toString === expectedString)

	val reallyTotallyEmptyMat = Mat(0,0)()
	expectedString = "\n __ \n -- \n"

    assert(reallyTotallyEmptyMat.toString === expectedString)
  }

  it must "have a sameEntries method" in {
    val a = Mat(2,3)(1,2,3,4,5,6)
    val b = Mat(2,3)(1,2,3,4,5,6)
	val c = Mat(2,3)(0,8,7,8,6,5)
	val g = Mat(1,1)(1)
	assert(a.sameEntries(b))
	assert(a.sameEntries(a))
	assert(!a.sameEntries(c))
	assert(!a.sameEntries(g))
	assert(!g.sameEntries(a))
  }

  it must "be reshapable" in {
    val mat = Mat(2,3)(1,2,3,4,5,6)
	val reshaped = mat.reshape(3,2)
	val expected = Mat(3,2)(1,2,3,4,5,6)
	assert(reshaped.sameEntries(expected))
  }

  it must "be transposable" in {
    val mat = Mat(2,3)(1,2,3,4,5,6)
	val transposed = mat.transpose
	val expected = Mat(3,2)(1,4,2,5,3,6)
	assert(transposed.sameEntries(expected))
  }

  it must "have a map method that respects the default" in {
    val a = Mat(2,2,7)(1,2,3,4)
	val b = a.map{ x => x * x}
	val expected = Mat(2,2)(1,4,9,16)
	assert(b.sameEntries(expected))
	assert(b.defaultValue === 49)
  }

  it must "have a filter method" in {
    val a = Mat(3,2,0)(1,-2,3,-4,5,-6)
	val nonNeg = a.filter {_ >= 0d}
	val neg = a.filter { _ < 0d}
	val expectedNonNeg = Mat(3,2,0d)(1,0,3,0,5,0)
	val expectedNeg = Mat(3,2,0d)(0,-2,0,-4,0,-6)
	assert(nonNeg.sameEntries(expectedNonNeg))
	assert(neg.sameEntries(expectedNeg))
	val anotherNeg = a.transpose.filter{_ < 0d}.transpose
	assert(neg.sameEntries(anotherNeg))
  }
}
