package org.kramerlab.autoencoder.math.matrix

import org.scalatest.FlatSpec

/**
 * Test suite for algebraic properties of `Mat`.
 * This test suite heavily relies on the fact that
 * at least the corresponding structural test suite
 * works. Here we attempt to ensure the correctness
 * of all operations somehow related to the field
 * structure of `Double`, such as addition and 
 * subtraction of matrices, multiplication with 
 * scalars, and multiplication of matrices with
 * each other.
 */

class MatAlgebraicTest extends FlatSpec {
  
  def telephone = Mat(3,3)(1 to 9)
  def countVec = Mat(10,1)(1 to 10)
  def emptyVec = Mat.empty(5,0)
  def skew = Mat(2,3)(1 to 6)

  "A Mat" must "support addition with other matrices" in {
    val telesum = telephone + telephone.transpose
	val expected = Mat(3,3)(2, 6, 10, 6, 10, 14, 10, 14, 18)
	assert(expected.sameEntries(telesum),
	  expected + " wasn't equal " + telesum)

	val skewSum = skew + skew - skew + skew + skew
	assert(skew.map{_ * 3d}.sameEntries(skewSum))

	assert(emptyVec.sameEntries(emptyVec + emptyVec))
  }

  it must "support multiplication by scalars" in {
	assert((telephone * 42).sameEntries(telephone.map{_ * 42}))
	assert(((telephone * 42) / 42).sameEntries(telephone))

	assert((skew * 0).forall{_ == 0d})
	assert((skew.transpose / 1d).sameEntries(skew.transpose))
  }

  it must "be updatable with +=, -=, *= etc." in {
    val t = telephone
	t += t * 2
	assert(t.sameEntries(telephone * 3), 
	  t + " wasn't equal " + (telephone * 3))

	val s = skew
	s += s
	assert(s.sameEntries(skew.map{_ * 2}))

	val c = countVec
	c *= 0
	assert(c.sameEntries(Mat.empty(c.height, c.width, 0d)))
  }

  it must "multiply correctly" in {
    val s = skew
	val expected_it = Mat(2,2)(14,32,32,77)
	val expected_ti = Mat(3,3)(17, 22, 27, 22, 29, 36, 27, 36, 45)
	assert((s * s.transpose).sameEntries(expected_it))
	assert((s.transpose * s).sameEntries(expected_ti))
  }

  it must "multiply large random matrices correctly" in {
    val rnd = new util.Random(0)
    val dim = 100
    val h = rnd.nextInt(dim) + dim
    val k = rnd.nextInt(dim) + dim
    val w = rnd.nextInt(dim) + dim
    
    for (i <- 1 to 8) {
      val a = Mat.fill(h, k){ case _ => rnd.nextGaussian }
      val b = Mat.fill(k, w){ case _ => rnd.nextGaussian }
    
      val naiveResult = a naive_* b
      val result = a * b
      assert((naiveResult - result).l2Norm < 0.0000001)
    }
  }
  
  it must "multiply correctly even the weird empty matrices" in {
    val z = emptyVec
	val bigZ = z * z.transpose
	assert(bigZ.sameEntries(Mat.empty(z.height, z.height,0d)))

	val singularity = z.transpose * z
	assert(singularity.size === (0,0))
  }

  it must "behave as expected with geometric series" in {
    val nilpotent = Mat.fill(10,10){ (r, c) => 
	  if (r < c) (r + c).toDouble else 0d
	}

    val identity = Mat.fill(10,10){ (r, c) => 
	  if (r == c) 1d else 0d 
	}

	val inverse = Mat.empty(10,10)
	var power = identity
	for (i <- 0 to 11) {
	  inverse += power
	  power = nilpotent * power
	}

	val a = identity - nilpotent
	val shouldBeId = a * inverse
	assert(shouldBeId.sameEntries(identity), 
	  "Tried to invert " + a + " but got " + inverse +
	  " so that the product was " + shouldBeId)
  }
}
