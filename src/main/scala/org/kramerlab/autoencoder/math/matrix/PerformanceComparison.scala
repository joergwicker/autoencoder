package org.kramerlab.autoencoder.math.matrix

import org.ejml.simple.SimpleMatrix

/**
 * Compares various matrix-matrix multiplication methods
 */
object PerformanceComparison {

  def measuringTime[X](f: => X): (X, Long) = {
    val start = System.currentTimeMillis
    val result = f
    val end = System.currentTimeMillis()
    (result, end - start)
  }
  
  def main(args: Array[String]) {
    val deterministic = true
    val dim = 2000
    val runs = 20
    val rnd = new util.Random(0)
    val jRnd = new java.util.Random(0)
    
    var totalNaiveTime = 0L
    var totalTiledTime = 0L
    var totalParTime = 0L
    var totalEjmlSimpleTime = 0L
    
    for (i <- 1 to runs) {
      val (h, k, w) = if (deterministic) {
        (dim, dim, dim)
      } else {
        (rnd.nextInt(dim) + dim, 
         rnd.nextInt(dim) + dim, 
         rnd.nextInt(dim) + dim)
      }
                
      val a = Mat.fill(h, k){ case _ => rnd.nextGaussian }
      val b = Mat.fill(k, w){ case _ => rnd.nextGaussian }
    
      val ejmlA = SimpleMatrix.random(h, k, -100, 100, jRnd)
      val ejmlB = SimpleMatrix.random(k, w, -100, 100, jRnd)
      
      val (naiveResult, naiveDt) = measuringTime{ /* a naive_* b */} // 12s for dim=1000
      val (tiledResult, tiledDt) = measuringTime{ /* a tiled_* b */} // 2.2s for dim=1000
      val (parResult, parDt) = measuringTime{a tiled_parallel_* b}
      val (ejmlSimpleResult, ejmlSimpleDt) = measuringTime{ejmlA.mult(ejmlB)}
      
//      assert((naiveResult - tiledResult).l2Norm < 0.00001)
//      if ((naiveResult - parResult).l2Norm > 0.00001) {
//         println("Unequal results obtained")
//         println(naiveResult)
//         println(parResult)
//         return ();
//      }
      
      totalNaiveTime += naiveDt
      totalTiledTime += tiledDt
      totalParTime += parDt
      totalEjmlSimpleTime += ejmlSimpleDt
    }
    
    println("Naive: " + (totalNaiveTime / runs.toDouble))
    println("Tiled: " + (totalTiledTime / runs.toDouble))
    println("Par  : " + (totalParTime / runs.toDouble))
    println("EJML simple: " + (totalEjmlSimpleTime / runs.toDouble))
  }
  
  // dim = 2000x2000
  // Par  : 9472.4
  // EJML simple: 13941.35
  // octave: 1415.5
  
}