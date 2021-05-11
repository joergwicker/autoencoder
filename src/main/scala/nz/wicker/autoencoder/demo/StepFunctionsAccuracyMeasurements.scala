//package nz.wicker.autoencoder.demo
//
//import nz.wicker.autoencoder.neuralnet.rbm.Rbm
//import nz.wicker.autoencoder.math.matrix.Mat
//import nz.wicker.autoencoder.neuralnet.FullBipartiteConnection
//import nz.wicker.autoencoder.neuralnet.rbm.RbmStack
//import nz.wicker.autoencoder.neuralnet.rbm.BernoulliUnitLayer
//import nz.wicker.autoencoder.neuralnet.rbm.DefaultRbmTrainingConfiguration
//import nz.wicker.autoencoder.{trainAutoencoder, Linear, Sigmoid}
//import nz.wicker.autoencoder.math.matrix.Matrix
//import nz.wicker.autoencoder.math.optimization.ConjugateGradientDescent_HagerZhangConfiguration
//import nz.wicker.autoencoder.demo.Indexing._
//
//object StepFunctionsAccuracyMeasurements {
//    
//  def randomStepFunction(steps: Int): (Double => Double) = {
//    val onOff = (0 to steps).map{ x => if (math.random > 0.5) 1d else 0d }
//    (x => onOff((x * steps).floor.toInt))
//  }
//  
//  def interval(dataDim: Int) = (for (i <- 0 until dataDim) yield {
//    i.toDouble / dataDim
//  }).toArray
//  
//  def arrToRow(arr: Array[Double]) = {
//    val res = new Mat(1, arr.size, 0)
//    for (i <- 0 until arr.size) res(0, i) = arr(i)
//    res
//  }
//  
//  def rowToArr(row: Matrix) = {
//    val res = new Array[Double](row.width)
//    for (i <- 0 until row.width) {
//      res(i) = row(0, i)
//    }
//    res
//  }
//  
//  
//  
//  def main(args: Array[String]) {
//    
//    // generate data
//    val steps = 10
//    val visDim = 1000
//    val hidDim = 50
//    val numberOfExamples = 5000
//    val data = new Mat(numberOfExamples, visDim, 0)
//    for (r <- 0 until numberOfExamples) {
//      val function = randomStepFunction(steps)
//      val rowData = interval(visDim) map function
//      for (c <- 0 until visDim) {
//        data(r, c) = rowData(c)
//      }
//    }
//    
//    def testIt(maxEpochs: Int, maxCgIters: Int): Unit = {
//      println("===============================================================")
//        val startTime = System.currentTimeMillis()
//        println("numExamples: " + numberOfExamples)
//        println("visDim: " + visDim)
//        println("epochs: " + maxEpochs + " cgIters: " + maxCgIters)
//        
//        // generate a whole autoencoder with, say, 3 levels
//        val autoencoder = trainAutoencoder(
//          Array(Sigmoid, Sigmoid, Sigmoid),
//          Array(visDim, 128, 16),
//          data,
//          new DefaultRbmTrainingConfiguration(epochs = maxEpochs),
//          new ConjugateGradientDescent_HagerZhangConfiguration(maxIters = maxCgIters)
//        )
//        
//        // create random examples and see the reconstructions produced after
//        // compression and decompression
//        
//        var errSum = 0d
//        for (i <- 0 until 100) {
//          val f = randomStepFunction(steps)
//          val input = arrToRow(interval(visDim) map f)
//          val compressed = autoencoder.compress(input)
//          val output = autoencoder.decompress(compressed)
//          val err = (input - output).l2NormSq
//          errSum += err
//        }
//        println("total error: " + errSum)
//        val totalTime = System.currentTimeMillis() - startTime
//        println("time: " + (totalTime / 60000d) + " minutes")
//    }
//    
//    // baseline: basically no training
//    testIt(1, 1)
//    testIt(5, 10)
//    testIt(20, 10)
//    testIt(20, 40)
//    
//    val seq = List(50, 200, 500).toStream
//    
//    for ((epochs, maxCgIters) <- seq x seq) {
//      testIt(epochs, maxCgIters)
//    }
//  }
//}