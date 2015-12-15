package org.kramerlab.autoencoder.demo

import org.kramerlab.autoencoder
import org.kramerlab.autoencoder.math.matrix._
import scala.math.pow
import org.kramerlab.autoencoder.math.matrix.Mat
import org.kramerlab.autoencoder.math.optimization.ConjugateGradientDescent_HagerZhangConfiguration
import org.kramerlab.autoencoder.neuralnet.rbm.DefaultRbmTrainingConfiguration
import org.kramerlab.autoencoder.neuralnet.rbm.RbmTrainingStrategy
import org.kramerlab.autoencoder.trainAutoencoder
import org.kramerlab.autoencoder.math.optimization.CG_Rasmussen2
import org.kramerlab.autoencoder.visualization.VisualizationComponent
import org.kramerlab.autoencoder.visualization.TrainingObserver
import scala.swing.MainFrame
import java.awt.Dimension
import java.util.GregorianCalendar
import org.kramerlab.autoencoder.thresholding._
import org.kramerlab.autoencoder.wekacompatibility._
import org.kramerlab.autoencoder.experiments.ErrorMeasures._
import org.kramerlab.autoencoder.experiments.CrossValidation

object ConfigurableCompressionMain {
  
  def main(args: Array[String]): Unit = {
    
    // parse the arguments
    var fileName: String = "NOT_SPECIFIED"
    var compressionDimension: Int = 5
    var guiMode: Boolean = false
    var hiddenLayers: Int = 2
    var folds: Int = 1
    var l2Error: Boolean = false
    var strategy: () => RbmTrainingStrategy = autoencoder.TournamentStrategy
    var strategyName = "TournamentStrategy (default, not overridden)"
    
    for (arg <- args) {
      val keyValue = arg.split("=")
      val key = keyValue(0)
      val value = keyValue(1)
      key match {
        case "FILE" => fileName = value
        case "COMPRESSION_DIMENSION" => 
          compressionDimension = value.toInt
        case "GUI_MODE" => 
          guiMode = value.toBoolean
        case "HIDDEN_LAYERS" => 
          hiddenLayers = value.toInt
        case "FOLDS" =>
          folds = value.toInt
        case "L2ERROR" => 
          l2Error = value.toBoolean
        case "STRATEGY" => {
          var unknownStrategy = false
          value match {
            case "Tournament" => strategy = autoencoder.TournamentStrategy
            case "Hinton" => strategy = autoencoder.HintonsMiraculousStrategy
            case "Random" => strategy = autoencoder.RandomRetryStrategy
            case "None" => strategy = autoencoder.NoPretraining
            case _ => unknownStrategy = true
          }
          if (!unknownStrategy) {
            strategyName = value
          }
        }
      }
    }
    
    println("FILE: " + fileName)
    println("COMPRESSION_DIMENSION: " + compressionDimension)
    println("GUI_MODE: " + guiMode)
    println("HIDDEN_LAYERS: " + hiddenLayers)
    println("FOLDS: " + folds)
    println("L2ERROR: " + l2Error)
    println("STRATEGY: " + strategyName)

    // load input file, print few values
    val data = readBooleanArff(fileName)
    
    println("dimensions: " + data.height + " x " + data.width)
    val numberOfEntries = data.height * data.width
    println("number of entries: " + numberOfEntries)
    val numberOfOnes = data.normSq
    println("number of ones: " + numberOfOnes)
    
    // define cross validation, calculate errors and balanced accuracies 
    val cv = new CrossValidation[(Int, Double)](data, folds, { trainingSet =>
      
      // given a training set, create a single autoencoder,
      // and calculate the optimal column thresholds
      val autoencoder = createAutoencoder(
        trainingSet,
        compressionDimension,
        hiddenLayers,
        l2Error,
        strategy,
        createObservers(guiMode)  
      )
      
      val compression = autoencoder.compress(trainingSet)         
      val reconstruction = autoencoder.decompress(compression)
      val thresholds = findOptimalColumnThresholds(reconstruction, trainingSet)
      
      { testSet =>
        // compress, decompress, binarize, count errors
        val testCompression = autoencoder.compress(testSet)
        val testDecompression = autoencoder.decompress(testCompression)
        val testReconstruction = binarize(testDecompression, thresholds)
        
        val err = reconstructionError(testSet, testReconstruction)
        val aba = averageBalancedAccuracy(testSet, testReconstruction)
        (err, aba)
      }
    })
    
    // run the cross validation
    val cvResults = cv()
    
    val (errors, accuracies) = cvResults.unzip
    println("Errors: " + errors)
    println("Accuracies: " + accuracies)
    val finalError = errors.sum / errors.size.toDouble
    val finalAccuracy = accuracies.sum / accuracies.size.toDouble
    
    // print the relevant result
    println("Total error: " + finalError)
    println("Relative error: " + finalError / numberOfEntries)
    println("Hamming loss: " + finalError / numberOfOnes)
    println("Average balanced accuracy: " + finalAccuracy)
  }
  
  def createAutoencoder(
    data: Mat, 
    compressionDimension: Int,
    numberOfHiddenLayers: Int,
    l2Error: Boolean,
    pretrainingStrategyFactory: () => RbmTrainingStrategy,
    obs: List[TrainingObserver]
  ) = {
    autoencoder.trainAutoencoder(
      data, 
      compressionDimension, 
      numberOfHiddenLayers,
      l2Error,
      pretrainingStrategyFactory,
      obs
    )
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