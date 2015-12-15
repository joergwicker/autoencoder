//package org.kramerlab.autoencoder.demo
//
//import org.kramerlab.autoencoder.math.matrix._
//import scala.Array.canBuildFrom
//import scala.collection.immutable.Stream.consWrapper
//import scala.io.Source
//import scala.math.pow
//import org.kramerlab.autoencoder.Sigmoid
//import org.kramerlab.autoencoder.math.optimization.ConjugateGradientDescent_HagerZhangConfiguration
//import org.kramerlab.autoencoder.neuralnet.rbm.DefaultRbmTrainingConfiguration
//import org.kramerlab.autoencoder.trainAutoencoder
//import org.kramerlab.autoencoder.math.combinatorics.Indexing._
//import org.kramerlab.autoencoder.{layerDims}
//import org.kramerlab.autoencoder.math.optimization.CG_Rasmussen2
//import org.kramerlab.autoencoder.math.matrix._
//import org.kramerlab.autoencoder.visualization._
//import scala.swing.MainFrame
//import java.awt.Dimension
//import org.kramerlab.autoencoder.neuralnet.rbm.RbmTrainingConfiguration
//import org.kramerlab.autoencoder.neuralnet.rbm.DefaultRbmTrainingConfiguration
//import org.kramerlab.autoencoder.visualization.VisualizationComponent
//import org.kramerlab.autoencoder.math.optimization.Minimizer
//import org.kramerlab.autoencoder.math.optimization.DifferentiableErrorFunctionFactory
//import org.kramerlab.autoencoder.thresholding._
//import org.kramerlab.autoencoder
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