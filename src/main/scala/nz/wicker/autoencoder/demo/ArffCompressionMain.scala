package nz.wicker.autoencoder.demo

import nz.wicker.autoencoder.math.matrix._
import scala.Array.canBuildFrom
import scala.collection.immutable.Stream.consWrapper
import scala.io.Source
import scala.math.pow
import nz.wicker.autoencoder.Sigmoid
import nz.wicker.autoencoder.math.optimization.ConjugateGradientDescent_HagerZhangConfiguration
import nz.wicker.autoencoder.neuralnet.rbm.DefaultRbmTrainingConfiguration
import nz.wicker.autoencoder.trainAutoencoder
import nz.wicker.autoencoder.math.combinatorics.Indexing._
import nz.wicker.autoencoder.{layerDims}
import nz.wicker.autoencoder.math.optimization.CG_Rasmussen2
import nz.wicker.autoencoder.math.matrix._
import nz.wicker.autoencoder.visualization._
import scala.swing.MainFrame
import java.awt.Dimension
import nz.wicker.autoencoder.neuralnet.rbm.RbmTrainingConfiguration
import nz.wicker.autoencoder.neuralnet.rbm.DefaultRbmTrainingConfiguration
import nz.wicker.autoencoder.visualization.VisualizationComponent
import nz.wicker.autoencoder.math.optimization.Minimizer
import nz.wicker.autoencoder.math.optimization.DifferentiableErrorFunctionFactory
import nz.wicker.autoencoder.thresholding._
import scala.Array.canBuildFrom
import scala.io.Source
import nz.wicker.autoencoder.math.matrix.Mat
import nz.wicker.autoencoder.visualization.VisualizationComponent
import scala.swing.MainFrame
import nz.wicker.autoencoder.visualization.TrainingObserver
import nz.wicker.autoencoder.neuralnet.autoencoder.Autoencoder
import nz.wicker.autoencoder.wekacompatibility._

trait ArffCompressionMain {

  def fileName: String
  def isSparse: Boolean 
  def compressionDimension: Int
  def numberOfHiddenLayers: Int
  
  val ExpectedConfigFilePath = "arffCompression.conf"
  
  def main(args: Array[String]): Unit = {
    
    // parse config file, find path to the datasets
    var directoryPath: Option[String] = None
    for (line <- Source.fromFile(ExpectedConfigFilePath).getLines) {
      val trimmed = line.trim
      if (trimmed.startsWith("DATASETS_PATH")) {
        directoryPath = Some(trimmed.split("=")(1).trim)
      }
    }
    
    if (args.size > 0) {
      directoryPath = Some(args(0))
    }
    
    val guiMode = !(args.mkString(" ").contains("GUI_OFF"))
    
    directoryPath match {
      case Some(p) => {
        run(directoryPath.get, guiMode)
      }
      case None => {
        println(
          "Expected to find a configuration file with " +
          "variable DATASETS_PATH in " + ExpectedConfigFilePath + " or " +
          "path to the directory with datasets as first argument " +
          "(ending with '/')"
        )
      }
    }
  }
  
  def createAutoencoder(
    mat: Mat,
    numHidLayers: Int, 
    observers: List[TrainingObserver]
  ): Autoencoder
  
  def run(
    inputDirectoryPath: String, 
    guiMode: Boolean): Unit = {
    // read input args
    val inputFilePath = inputDirectoryPath + fileName
     
    // load input file
    val mat = 
      if (isSparse) {
        readSparseBooleanArff(inputFilePath)
      } else { 
        readDenseBooleanArff(inputFilePath)
      }
    
    mat.shuffleRows()
    
    println("Training data: " + fileName)
    println("dimensions: " + mat.height + " x " + mat.width)
    println("number of ones: " + mat.normSq)
    
    // train the autoencoder
    val startTime = System.currentTimeMillis
    
    val observers = createObservers(guiMode)    
    val autoencoder = createAutoencoder(
      mat,
      numberOfHiddenLayers,
      observers
    )
    
    // calculate the errors
    val endTime = System.currentTimeMillis
    val totalTime = endTime - startTime
    println("TIME FOR TRAINING: " + (totalTime / 60000) + " min")
    
    val reconstruction = autoencoder(mat)
    val error = (reconstruction - mat).l2Norm
    println("L2 Error: " + error)
    
    val threshold = findOptimalThreshold(reconstruction, mat)
    val binaryReconstruction = binarize(reconstruction, threshold)
    
    val binaryDifference = binaryReconstruction - mat
    val oneToZeroErrors = binaryDifference.filter(_ < 0).normSq
    val zeroToOneErrors = binaryDifference.filter(_ > 0).normSq
    
    val binaryError = (binaryReconstruction - mat).normSq
    println("Total number of errors: " + binaryError)
    println("0 -> 1 errors: " + zeroToOneErrors)
    println("1 -> 0 errors: " + oneToZeroErrors)
    
    // display the input, compression, reconstruction
    val padding = 20
    val finalResultMatrix = 
      Mat.empty(
        mat.height, 
        mat.width * 2 + compressionDimension + 2 * padding, 0
      )
    finalResultMatrix(0 ::: end, 0 ::: mat.width) = mat
    finalResultMatrix(0 ::: end,
      (mat.width + padding) ::: (mat.width + compressionDimension + padding)
    ) = autoencoder.compress(mat).asInstanceOf[Mat] 
    finalResultMatrix(
      0 ::: end, (mat.width + compressionDimension + 2 * padding) ::: end) = 
      binaryReconstruction.asInstanceOf[Mat]
    
    for (obs <- observers){
      obs.notify(
        nz.wicker.autoencoder.visualization.VisualizableIntermediateResult(
          finalResultMatrix
        ),
        true
      )
    }
  }
 
  def createObservers(guiMode: Boolean): List[TrainingObserver] = {
    if (guiMode) {
      val vis = new VisualizationComponent
      val mainFrame = new MainFrame {
        title = "Autoencoder training progress"
        contents = vis
      }
      
      mainFrame.size = new Dimension(400, 800)
      mainFrame.visible = true
      
      List(vis)
    } else {
      Nil
    }
  }
}