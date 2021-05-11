//package nz.wicker.autoencoder.demo
//
//import nz.wicker.autoencoder.math.matrix.Mat
//
//object PolynomialExample {
//  
//  def randomPoly(degree: Int) = Array.fill(degree){math.random}
//  
//  def evalPoly(coeffs: Array[Double], x: Double) = {
//    def rec(k: Int): Double = {
//      if (k == coeffs.size) 0 else coeffs(k) + x * rec(k + 1)
//    }
//    rec(0)
//  }
//  
//  def evalAtInterval(coeffs: Array[Double], dataDim: Int) = {
//    for (i <- 0 until dataDim; x = i.toDouble / dataDim) yield { 
//      evalPoly(coeffs, x)
//    }
//  }
// 
//  def arrToRow(arr: Array[Double]) = {
//    val res = new Mat(1, arr.size, 0)
//    for (i <- 0 until arr.size) res(0, i) = arr(i)
//    res
//  }
//  
//  def main(args: Array[String]) {
//    
//    // generate data
//    val realDim = 4
//    val dataDim = 16
//    val numberOfExamples = 10000
//    val data = new Mat(numberOfExamples, dataDim, 0)
//    for (r <- 0 until numberOfExamples) {
//      val rowData = evalAtInterval(randomPoly(realDim), dataDim)
//      for (c <- 0 until dataDim) {
//        data(r, c) = rowData(c)
//      }
//    }
//    
//    // train autoencoder
//    import nz.wicker.autoencoder._
//    val autoencoder = trainAutoencoder(
//      Array(Linear, Sigmoid, Linear),
//      Array(dataDim, 10, 6),
//      data
//    )
//    
//    // take a look at the results
//    val x = arrToRow(evalAtInterval(randomPoly(realDim), dataDim).toArray)
//    println("compressing")
//    println("compressor is: " + autoencoder.compressor)
//    val y = autoencoder.compress(x)
//    println("decompressing")
//    println("decompressor is: " + autoencoder.decompressor)
//    val z = autoencoder.decompress(y)
//    
//    println("voila: ")
//    println(x)
//    println(z)
//    println("dist: " + (x - z).l2Norm)
//  }
//}