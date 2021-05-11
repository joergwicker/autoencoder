package nz.wicker.autoencoder.demo

import nz.wicker.autoencoder._
import nz.wicker.autoencoder.math.matrix._
import scala.math.pow
import nz.wicker.autoencoder.math.matrix.Mat
import nz.wicker.autoencoder.math.optimization.ConjugateGradientDescent_HagerZhangConfiguration
import nz.wicker.autoencoder.neuralnet.rbm.DefaultRbmTrainingConfiguration
import nz.wicker.autoencoder.trainAutoencoder
import nz.wicker.autoencoder.math.optimization.CG_Rasmussen2
import nz.wicker.autoencoder.visualization.VisualizationComponent
import nz.wicker.autoencoder.visualization.TrainingObserver
import scala.swing.MainFrame
import java.awt.Dimension
import java.util.GregorianCalendar
import nz.wicker.autoencoder.thresholding._
import nz.wicker.autoencoder.wekacompatibility._
import nz.wicker.autoencoder.neuralnet.rbm.RbmTrainingStrategy
import nz.wicker.autoencoder.experiments.ErrorMeasures

object ConfigurableCompressionMain_Stream {
  
  def main(args: Array[String]): Unit = {
    try {
      var fileName: String = "NOT_SPECIFIED"
      var compressionFactor: Double = 0.66
      var guiMode: Boolean = false
      var maxDepth: Int = 2
      var reps: Int = 1
      var l2Error: Boolean = false
      var strategy: () => RbmTrainingStrategy = TournamentStrategy
      var strategyName = "TournamentStrategy (default, not overridden)"
      var fineTuneInner = false
      
      for (arg <- args) {
        val keyValue = arg.split("=")
        val key = keyValue(0)
        val value = keyValue(1)
        key match {
          case "FILE" => fileName = value
          case "COMPRESSION_FACTOR" => 
            compressionFactor = value.toDouble
          case "GUI_MODE" => 
            guiMode = value.toBoolean
          case "MAX_DEPTH" => 
            maxDepth = value.toInt
          case "REPS" =>
            reps = value.toInt
          case "L2ERROR" => 
            l2Error = value.toBoolean
          case "STRATEGY" => {
            var unknownStrategy = false
            value match {
              case "Tournament" => strategy = TournamentStrategy
              case "Hinton" => strategy = HintonsMiraculousStrategy
              case "Random" => strategy = RandomRetryStrategy
              case "None" => strategy = NoPretraining
              case _ => unknownStrategy = true
            }
            if (!unknownStrategy) {
              strategyName = value
            }
          }
          case "FINE_TUNE_INNER" => {
            fineTuneInner = value.toBoolean
          }  
          case x: String =>
            println("Warning: unrecognized argument " + x)
            
        }
      }
      
      println("After the interpretation of arguments: ")
      println("FILE: " + fileName)
      println("COMPRESSION_FACTOR: " + compressionFactor)
      println("GUI_MODE: " + guiMode)
      println("MAX_DEPTH: " + maxDepth)
      println("REPS: " + reps)
      println("L2ERROR: " + l2Error)
      println("STRATEGY: " + strategyName)
      println("FINE_TUNE_INNER: " + fineTuneInner)
      
      var rep = 0
      var results: List[Int] = Nil
      while (rep < reps) {
        run(
          fileName, 
          guiMode, 
          compressionFactor,
          maxDepth,
          l2Error,
          strategy,
          fineTuneInner
        )
        rep += 1
      }
      
    } catch {
      case e: Exception => {
        println("Exception occurred: ") 
        e.printStackTrace()
      }
    }
  }
  
  //def createAutoencoder(
  //  data: Mat, 
  //  compressionDimension: Int,
  //  numberOfHiddenLayers: Int,
  //  l2Error: Boolean,
  //  obs: List[TrainingObserver]
  //) = {
  //  autoencoder.trainAutoencoder(
  //    data, 
  //    compressionDimension, 
  //    numberOfHiddenLayers,
  //    l2Error,
  //    obs
  //  )
  //}
  
  def run(
    fileName: String, 
    guiMode: Boolean,
    compressionFactor: Double,
    maxDepth: Int,
    useL2Error: Boolean,
    pretrainingStrategyFactory: () => RbmTrainingStrategy,
    fineTuneInner: Boolean
  ): Unit = {
    
    // load input file
    val mat = readBooleanArff(fileName)
    
    mat.shuffleRows()
    
    println("Training data: " + fileName)
    println("dimensions: " + mat.height + " x " + mat.width)
    val numberOfEntries = mat.height * mat.width
    val numberOfOnes = mat.normSq
    println("number of ones: " + mat.normSq)
    println("maxDepth: " + maxDepth)
    
    // train the autoencoder
    val startTime = System.currentTimeMillis
    
    val trainingObservers = createObservers(guiMode)
    
    for (autoencoder <- deepAutoencoderStream(
      Sigmoid,
      maxDepth: Int,
      ???, // TODOnew Metaparameters(mat.width, compressionDim, ),
      mat: Mat,
      useL2Error: Boolean,
      pretrainingStrategyFactory,
      fineTuneInner,
      trainingObservers: List[TrainingObserver]
    )) {
      // calculate the errors
      val endTime = System.currentTimeMillis
      val totalTime = endTime - startTime
      println("TIME FOR TRAINING: " + (totalTime / 60000) + " min")
      println("Number of unit layers: " + (autoencoder.layers.size + 1) / 2)
      println("Compression dimension: " + autoencoder.compressionDimension)
      
      val compression = autoencoder.compress(mat)
      // println("Example compressions: " + compression(0 ::: 50, 0 ::: end))
      val reconstruction = autoencoder.decompress(compression)
      val error = (reconstruction - mat).l2Norm
      println("L2 Error: " + error)
      
      val thresholds = findOptimalColumnThresholds(reconstruction, mat)
      val binaryReconstruction = binarize(reconstruction, thresholds)
      
      val binaryDifference = binaryReconstruction - mat
      val oneToZeroErrors = binaryDifference.filter(_ < 0).normSq
      val zeroToOneErrors = binaryDifference.filter(_ > 0).normSq
      
      val binaryError = (binaryReconstruction - mat).normSq
      println("Total number of errors: " + binaryError)
      println("0 -> 1 errors: " + zeroToOneErrors)
      println("1 -> 0 errors: " + oneToZeroErrors)
      println(
        (binaryError / numberOfEntries.toDouble * 100) + "% of all entries")
      println((binaryError / numberOfOnes.toDouble * 100) +"% of ones")
      
      println("Average balanced accuracy: " + 
        ErrorMeasures.averageBalancedAccuracy(mat, binaryReconstruction)
      )
      
      // display the input, compression, reconstruction
      val padding = 20
      val compressionDimension = autoencoder.compressionDimension
      val finalResultMatrix = 
        Mat.empty(
          mat.height, 
          mat.width * 2 +  + 2 * padding, 0
        )
      finalResultMatrix(0 ::: end, 0 ::: mat.width) = mat
      finalResultMatrix(0 ::: end,
        (mat.width + padding) ::: (mat.width + compressionDimension + padding)
      ) = autoencoder.compress(mat).asInstanceOf[Mat] 
      finalResultMatrix(
        0 ::: end, (mat.width + compressionDimension + 2 * padding) ::: end) = 
        binaryReconstruction.asInstanceOf[Mat]
      
      for (obs <- trainingObservers){
        obs.notify(
          nz.wicker.autoencoder.visualization.VisualizableIntermediateResult(
            finalResultMatrix
          ),
          true
        )
      }
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
