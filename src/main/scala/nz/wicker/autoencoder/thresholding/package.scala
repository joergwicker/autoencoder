package nz.wicker.autoencoder

import scala.math._
import nz.wicker.autoencoder.math.matrix.Mat
import nz.wicker.autoencoder.math.matrix._

package object thresholding {

  /**
   * Simple 1D-grid search that determines the best threshold for transforming
   * sigmoid activities into binary activities
   */
  def findOptimalThreshold(continuous: Mat, binary: Mat): Double = {
    def squash(t: Double) = t // acos((t + 1) / 2) / Pi
    val n = 1000
    val thresholdsErrors = for (k <- (0 until n)) yield {
      val t = squash(k / n.doubleValue)
      val thresholded = binarize(continuous, t)
      val error = (thresholded - binary).normSq
      (t, error)
    } 
    thresholdsErrors.minBy{_._2}._1
  }
  
  def binarize(mat: Mat, threshold: Double): Mat = {
    mat.map(x => if (x > threshold) 1 else 0)
  }
  
  def findOptimalColumnThresholds(continuous: Mat, binary: Mat): Mat = {
    val dim = continuous.width
    val n = 1000
    
    Mat.fill(1, dim) { (r, c) =>
      val continuousColumn = continuous(:::, c)
      val binaryColumn = binary(:::, c)
      findOptimalThreshold(continuousColumn, binaryColumn)
    }
  }
  
  def binarize(mat: Mat, columnThresholds: Mat): Mat = {
    mat.mapWithIndex{(r, c, x) => if (x > columnThresholds(c)) 1 else 0}
  }
  
}