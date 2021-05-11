package nz.wicker.autoencoder.experiments

import nz.wicker.autoencoder.math.matrix.Mat
import nz.wicker.autoencoder.math.matrix._

class CrossValidation[PerformanceMeasure](
  dataset: Mat, 
  folds: Int,
  train: Mat => (Mat => PerformanceMeasure)
){
  def apply(): Seq[PerformanceMeasure] = {
    val backup = dataset.clone
    backup.shuffleRows()
    
    val h = dataset.height
    for (fold <- 0 until folds) yield {
      val a = (fold * h) / folds
      val b = ((fold + 1) * h) / folds
      val trainingSetSize = b - a
      val trainingSet = Mat.empty(h - trainingSetSize, backup.width)
      trainingSet(0 ::: a, :::) = backup(0 ::: a, :::)
      trainingSet(a ::: end, :::) = backup(b ::: end, :::) 
      val algorithmEvaluation = train(trainingSet)
      val testSet = backup(a ::: b, :::)
      algorithmEvaluation(testSet)
    }
  }
}

object CrossValidation {
  def main(args: Array[String]): Unit = {
    val data = Mat(4, 2)(0,1,2,3,4,5,6,7)
    val cv = new CrossValidation[Unit](data, 3, {
      trainingSet => {testSet => 
        println(
          "=====\ntraining on : " + trainingSet + 
          " \n testing on: " + testSet
        )
      }
    })
    cv()
  }
}