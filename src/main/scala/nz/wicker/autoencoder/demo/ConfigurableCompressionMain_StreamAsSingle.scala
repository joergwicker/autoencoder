package nz.wicker.autoencoder.demo

import nz.wicker.autoencoder._
import nz.wicker.autoencoder.math.matrix._
import scala.math.pow
import nz.wicker.autoencoder.math.matrix.Mat
import nz.wicker.autoencoder.neuralnet.rbm.DefaultRbmTrainingConfiguration
import nz.wicker.autoencoder.neuralnet.rbm.RbmTrainingStrategy
import nz.wicker.autoencoder.trainAutoencoder
import nz.wicker.autoencoder.math.optimization.CG_Rasmussen2
import nz.wicker.autoencoder.visualization.VisualizationComponent
import nz.wicker.autoencoder.visualization.TrainingObserver
import scala.swing.MainFrame
import java.awt.Dimension
import java.util.GregorianCalendar
import nz.wicker.autoencoder.thresholding._
import nz.wicker.autoencoder.wekacompatibility._
import nz.wicker.autoencoder.experiments.ErrorMeasures._
import nz.wicker.autoencoder.experiments.CrossValidation
import nz.wicker.autoencoder.experiments.Metaparameters

object ConfigurableCompressionMain_StreamAsSingle {
  
  def main(args: Array[String]): Unit = {
    
    // parse the arguments
    var fileName: String = "NOT_SPECIFIED"
    var compressionDimension: Int = 5
    var finetuneInnerLayers: Boolean = false
    var guiMode: Boolean = false
    var hiddenLayers: Int = 2
    var folds: Int = 1
    var l2Error: Boolean = false
    var strategy: () => RbmTrainingStrategy = TournamentStrategy
    var strategyName = "TournamentStrategy (default, not overridden)"
    var cores = 8
    
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
        case "FINE_TUNE_INNER" => finetuneInnerLayers = value.toBoolean
        case "CORES"=> cores = value.toInt
      }
    }
    
    println("FILE: " + fileName)
    println("COMPRESSION_DIMENSION: " + compressionDimension)
    println("GUI_MODE: " + guiMode)
    println("HIDDEN_LAYERS: " + hiddenLayers)
    println("FOLDS: " + folds)
    println("L2ERROR: " + l2Error)
    println("STRATEGY: " + strategyName)
    println("FINE_TUNE_INNER: " + finetuneInnerLayers)
    println("CORES: " + cores)

    setParallelismGlobally(cores)
    
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
        finetuneInnerLayers,
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
    finetuneInnerLayers: Boolean,
    l2Error: Boolean,
    pretrainingStrategyFactory: () => RbmTrainingStrategy,
    obs: List[TrainingObserver]
  ) = {
    val result = deepAutoencoderStream(
      Sigmoid,
      numberOfHiddenLayers + 1,
      new Metaparameters(data.width, compressionDimension, numberOfHiddenLayers, false).layerDims.tail, 
      data,
      l2Error,
      pretrainingStrategyFactory,
      finetuneInnerLayers,
      obs
    )(numberOfHiddenLayers - 1)
    assert(result.layers.size == numberOfHiddenLayers * 4 + 1)
    assert(result.compressionDimension == compressionDimension)
    result
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