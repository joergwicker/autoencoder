//package nz.wicker.autoencoder.demo
//
//import nz.wicker.autoencoder.math.matrix._
//import scala.Array.canBuildFrom
//import scala.collection.immutable.Stream.consWrapper
//import scala.io.Source
//import scala.math.pow
//import nz.wicker.autoencoder.Sigmoid
//import nz.wicker.autoencoder.math.optimization.ConjugateGradientDescent_HagerZhangConfiguration
//import nz.wicker.autoencoder.neuralnet.rbm.DefaultRbmTrainingConfiguration
//import nz.wicker.autoencoder.trainAutoencoder
//import nz.wicker.autoencoder.math.combinatorics.Indexing._
//import nz.wicker.autoencoder.{layerDims}
//import nz.wicker.autoencoder.math.optimization.CG_Rasmussen2
//import nz.wicker.autoencoder.math.matrix._
//import nz.wicker.autoencoder.visualization._
//import scala.swing.MainFrame
//import java.awt.Dimension
//import nz.wicker.autoencoder.neuralnet.rbm.RbmTrainingConfiguration
//import nz.wicker.autoencoder.neuralnet.rbm.DefaultRbmTrainingConfiguration
//import nz.wicker.autoencoder.visualization.VisualizationComponent
//import nz.wicker.autoencoder.math.optimization.Minimizer
//import nz.wicker.autoencoder.math.optimization.DifferentiableErrorFunctionFactory
//import nz.wicker.autoencoder.thresholding._
//import nz.wicker.autoencoder
//
//trait RandomExperimentMain extends ArffCompressionMain {
//  
//  def fileName: String
//  def isSparse: Boolean
//  def compressionDimension: Int
//  
//  override def createAutoencoder(
//    data: Mat,
//    numberOfHiddenLayers: Int,
//    obs: List[TrainingObserver]) = {
//    autoencoder.trainAutoencoder(
//      data, 
//      compressionDimension, 
//      numberOfHiddenLayers, 
//      false,
//      obs
//    )
//  }
//  
//}