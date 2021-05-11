package nz.wicker.autoencoder.experiments

import nz.wicker.autoencoder.math.matrix.Mat

case class ClassificationResult(tp: Int, tn: Int, fp: Int, fn: Int) {
  def total = tp + tn + fp + fn
  def actualPositive = tp + fn
  def actualNegative = tn + fp
  def predictedPositive = tp + fp
  def predictedNegative = tn + fn
  def predictedTrue = tp + tn
  def predictedFalse = fp + fn
  def accuracy = predictedTrue.toDouble / total
  def precision = tp.toDouble / predictedPositive
  def sensitivity = tp.toDouble / actualPositive
  def specificity = tn.toDouble / actualNegative
  def balancedAccuracy = (sensitivity + specificity) / 2
}

object ErrorMeasures {
  
  def reconstructionError(binaryData: Mat, binaryReconstruction: Mat): Int = {
    val binaryDifference = binaryReconstruction - binaryData
    // val oneToZeroErrors = binaryDifference.filter(_ < 0).normSq
    // val zeroToOneErrors = binaryDifference.filter(_ > 0).normSq
    binaryDifference.normSq.toInt
  }
  
  def averageBalancedAccuracy(
    binaryData: Mat, 
    binaryReconstruction: Mat
  ): Double = {
    val w = binaryData.width
    val h = binaryData.height
    val row = Mat.ones(1, h)
    
    val falsePositives = 
      (binaryReconstruction - binaryData).map{x => math.max(x, 0d)}
    val falseNegatives = 
      (binaryData - binaryReconstruction).map{x => math.max(x, 0d)}
    val falsePositivesForColumns = row * falsePositives
    val falseNegativesForColumns = row * falseNegatives
    val positivesForColumns = row * binaryReconstruction
    val negativesForColumns = Mat.ones(1, w) * h - positivesForColumns
    val truePositivesForColumns = positivesForColumns - falsePositivesForColumns
    val trueNegativesForColumns = negativesForColumns - falseNegativesForColumns
    
    val classificationResults = Array.tabulate(w){ i =>
      ClassificationResult(
        truePositivesForColumns(0, i).toInt,
        trueNegativesForColumns(0, i).toInt,
        falsePositivesForColumns(0, i).toInt,
        falseNegativesForColumns(0, i).toInt
      )
    }
    
    classificationResults.map{_.balancedAccuracy}.sum / w
  }
  
  def main(args: Array[String]): Unit = {
    val coin = ClassificationResult(45, 5, 45, 5)
    val const = ClassificationResult(90, 0, 10, 0)
    val good = ClassificationResult(85, 9, 5, 1)
    val veryGood = ClassificationResult(89, 9, 1, 1)
    val perfect = ClassificationResult(90, 10, 0, 0)
    println(coin.balancedAccuracy)    // 0.5
    println(const.balancedAccuracy)   // 0.5
    println(good.balancedAccuracy)    // 0.8156146179401993
    println(veryGood.balancedAccuracy)// 0.9444444444444444
    println(perfect.balancedAccuracy) // 1.0
    
    val reality = Mat(6, 3)(
      1, 0, 1,
      0, 1, 1,
      0, 1, 1,
      1, 0, 0,
      1, 0, 1,
      0, 0, 1
    )
    
    val prediction = Mat(6, 3)(
      0, 0, 1,
      1, 1, 1,
      1, 0, 1,
      1, 0, 0,
      1, 0, 0,
      0, 0, 1
    )
    
    val aba = averageBalancedAccuracy(reality, prediction)
    
    // TODO: move this to unit tests. Or rescue the whole file content to
    // somewhere else, all this confusion-matrix stuff is really annoying 
    // to implement...
    println("manually calculated: " + 43d / 60)
    println("result: " + aba)
  }
}