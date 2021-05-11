//package nz.wicker.autoencoder.demo
//
//import nz.wicker.autoencoder.math.matrix._
//import scala.Array.canBuildFrom
//import scala.collection.immutable.Stream.consWrapper
//import scala.io.Source
//import scala.math.pow
//import nz.wicker.autoencoder.Sigmoid
//import nz.wicker.autoencoder.math.matrix.Mat
//import nz.wicker.autoencoder.math.optimization.ConjugateGradientDescent_HagerZhangConfiguration
//import nz.wicker.autoencoder.neuralnet.rbm.DefaultRbmTrainingConfiguration
//import nz.wicker.autoencoder.trainAutoencoder
//import nz.wicker.autoencoder.math.combinatorics.Indexing._
//import nz.wicker.autoencoder.{layerDims}
//import nz.wicker.autoencoder.math.matrix.Matrix
//import nz.wicker.autoencoder.math.optimization.CG_Rasmussen2
//import nz.wicker.autoencoder.math.matrix._
//import nz.wicker.autoencoder.visualization.VisualizationComponent
//import scala.swing.MainFrame
//import java.awt.Dimension
//import nz.wicker.autoencoder.neuralnet.rbm.RbmTrainingConfiguration
//import nz.wicker.autoencoder.neuralnet.rbm.DefaultRbmTrainingConfiguration
//import nz.wicker.autoencoder.visualization.VisualizationComponent
//import nz.wicker.autoencoder.math.optimization.Minimizer
//import nz.wicker.autoencoder.math.matrix.Mat
//import nz.wicker.autoencoder.math.optimization.DifferentiableErrorFunctionFactory
//import java.io.File
//import java.io.FileWriter
//
//trait SpikePlotMain extends ArffCompressionMain {
//
//  def fileName: String
//  def isSparse: Boolean
//  
//  case class Params (
//    val compressionDimension: Int,
//    val numLayers: Int,
//    val layerAlpha: Double,
//    val rbmConfiguration: RbmTrainingConfiguration,
//    val errorFunctionFactory: DifferentiableErrorFunctionFactory,
//    val minimizer: Minimizer
//  ) {
//    override def toString = {
//      "compression dim = " + compressionDimension + "\n" +
//      "numLayers = " + numLayers + "\n" +
//      "alpha = " + layerAlpha + "\n" +
//      "rbmConfiguration = " + rbmConfiguration + "\n" +
//      "minimizer =  " + minimizer
//    }
//  }
//  
//  val ExpectedConfigFilePath = "arffCompression.conf"
//  
//  def params: Seq[Params]
//  
//  def main(args: Array[String]): Unit = {
//    
//    // parse config file, find path to the datasets
//    var directoryPath: Option[String] = None
//    var outputFile: Option[String] = None
//    for (line <- Source.fromFile(ExpectedConfigFilePath).getLines) {
//      val trimmed = line.trim
//      if (trimmed.startsWith("DATASETS_PATH")) {
//        directoryPath = Some(trimmed.split("=")(1).trim)
//      }
//      if (trimmed.startsWith("OUTPUT_FILE")) {
//        outputFile = Some(trimmed.split("=")(1).trim)
//      }
//    }
//    
//    if (!outputFile.isEmpty)
//    directoryPath match {
//      case Some(p) => {
//        run(directoryPath.get, outputFile.get)
//      }
//      case None => {
//        println(
//          "Expected to find a configuration file with " +
//          "variable DATASETS_PATH in " + ExpectedConfigFilePath + " or " +
//          "path to the directory with datasets as first argument " +
//          "(ending with '/')"
//        )
//      }
//    }
//  }
//  
//  def run(
//    inputDirectoryPath: String,
//    outputFilePath: String
//  ): Unit = {
//    
//    val outputFile = new File(outputFilePath)
//    if (!outputFile.exists) {
//      outputFile.createNewFile()
//    }
//    val writer = new FileWriter(outputFile)
//    val inputFilePath = inputDirectoryPath + fileName
//    
//    // load input file
//    val mat = 
//      if (isSparse) {
//        readSparseBooleanArff(inputFilePath)
//      } else { 
//        readDenseBooleanArff(inputFilePath)
//      }
//    
//    mat.shuffleRows()
//    
//    println("Training data: " + fileName)
//    println("dimensions: " + mat.height + " x " + mat.width)
//    println("number of ones: " + mat.normSq)
//    
//    for (p <- params) {
//      
//      val dimVis = mat.width
//      val dims = layerDims(
//        dimVis, 
//        p.compressionDimension, 
//        p.numLayers, 
//        p.layerAlpha
//      ) 
//      
//      // train the autoencoder
//      val startTime = System.currentTimeMillis
//      
//      val layerTypes = Array.fill[Int](dims.size){Sigmoid}
//      
//      val rbmConfig = new DefaultRbmTrainingConfiguration(
//        epochs = 200,
//        learningRate = 0.1,
//        weightPenaltyFactor = 0.0001
//      )
//      
//      val autoencoder = trainAutoencoder(
//        layerTypes,
//        dims.toArray,
//        mat,
//        p.rbmConfiguration,
//        p.minimizer,
//        p.errorFunctionFactory,
//        trainingObservers = Nil
//      )
//      
//      // calculate the errors
//      val endTime = System.currentTimeMillis
//      val totalTime = endTime - startTime
//      println("TIME FOR TRAINING: " + (totalTime / 60000) + " min")
//      
//      val reconstruction = autoencoder(mat)
//      val error = (reconstruction - mat).l2Norm
//      println("L2 Error: " + error)
//      
//      val threshold = 0.75
//      val binaryReconstruction = reconstruction.map{x => 
//        if (x > threshold) 1d else 0d  
//      }
//      
//      val binaryDifference = binaryReconstruction - mat
//      val zeroToOneErrors = binaryDifference.filter(_ < 0).normSq
//      val oneToZeroErrors = binaryDifference.filter(_ > 0).normSq
//      
//      val binaryError = (binaryReconstruction - mat).normSq
//      println(p)
//      println("Total number of errors: " + binaryError)
//      println("0 -> 1 errors: " + zeroToOneErrors)
//      println("1 -> 0 errors: " + oneToZeroErrors)
//      
//      writer.write("=" * 60)
//      writer.write(p.toString)
//      writer.write("# errors = " + binaryError)
//      writer.write("# 0 -> 1 errors = " + zeroToOneErrors)
//      writer.write("# 1 -> 0 errors = " + oneToZeroErrors)
//    }
//  }
//}