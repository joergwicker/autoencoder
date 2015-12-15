package org.kramerlab.autoencoder.experiments

import scala.math._
import org.kramerlab.autoencoder
import org.kramerlab.autoencoder.neuralnet.rbm.RbmTrainingConfiguration
import org.kramerlab.autoencoder.math.optimization.Minimizer
import org.kramerlab.autoencoder.math.optimization.DifferentiableErrorFunctionFactory
import org.kramerlab.autoencoder.math.optimization.SquareErrorFunctionFactory
import org.kramerlab.autoencoder.math.optimization.CG_Rasmussen2
import org.kramerlab.autoencoder.math.optimization.SquareErrorFunctionFactory
import org.kramerlab.autoencoder.math.optimization.CrossEntropyErrorFunctionFactory
import org.kramerlab.autoencoder.neuralnet.rbm.DefaultRbmTrainingConfiguration
import org.kramerlab.autoencoder.math.random

/**
 * Instances of this class contain a full description of all parameters
 * that are required to train an autoencoder.
 */
case class Metaparameters(
  inputDim: Int,
  compressionDim: Int,
  numHidLayers: Int,
  linearCentralLayer: Boolean
) {
  
//  /**
//   * This method returns dimensions of layers for specified input dimension,
//   * central layer dimension, number of layers, and a parameter alpha, which
//   * determines, the "convexity" of the [layer-index -> layer-size] function
//   * (alpha = 1 corresponds to linear
//   * interpolation between number of visible and number of hidden units,
//   * alpha < 1 corresponds to "slim" networks, alpha > 1 corresponds to "fat"
//   * networks)
//   */
//  def layerDims: Array[Int] = {
//    (for (k <- 0 to numLayers) yield {
//      (compressionDim + (inputDim - compressionDim) * pow(
//         1 - pow(k.toDouble / numLayers, layerDimensionsAlpha), 
//         1d / layerDimensionsAlpha
//       )
//      ).round.toInt
//    }).toArray
//  }
  
  def layerDims: Array[Int] = {
    val exponent = math.pow(
      compressionDim.toDouble / inputDim, 
      1d / numHidLayers
    )
    val inner = (1 until numHidLayers).map{
      k => (inputDim * math.pow(exponent, k)).toInt
    }.toList
    (inputDim :: (inner ++ List(compressionDim))).toArray
  }
  
  def layerTypes = if (linearCentralLayer) {
    (List.fill(numHidLayers){autoencoder.Sigmoid} :+ autoencoder.Linear).toArray
  } else {
    Array.fill(numHidLayers + 1){autoencoder.Sigmoid}
  }
  
}

object Metaparameters {
  
  /**
   * Creates a random configuration for a single RBM.
   */
  def createRandomRbmConfiguration(
    datasetSize: Int
  ): RbmTrainingConfiguration = {
    
    val learningRate = random.unif(0.005, 0.25)
    val weightPenaltyFactor = learningRate * random.exp(0.00001, 0.002)
    val initialMomentum = random.exp(0.5, 0.75)
    val finalMomentum =  random.unif(initialMomentum, 0.95)
    val initialWeightScaling = random.exp(0.01, 0.1) 
    val initialBiasScaling = random.unif(0.01, 1) * initialWeightScaling
    
    val result = new DefaultRbmTrainingConfiguration(
      epochs = 20, // influences the "center" of other varying params
      minibatchSize = 10,
      learningRate = learningRate,
      initialBiasScaling = initialBiasScaling,
      initialWeightScaling = initialWeightScaling,
      initialMomentum = initialMomentum,
      finalMomentum = finalMomentum,
      initialGibbsSamplingSteps = 1,
      finalGibbsSamplingSteps = random.geom(2, 5),
      sampleVisibleUnitsDeterministically = false,
      weightPenaltyFactor = weightPenaltyFactor
    )
    
    result
  }
  
  // weird function that somehow maps the dimension of previous layer to
  // the dimension of next layer, given the compression factor
  def nextLayerDimension(compressionFactor: Double, currentDim: Int): Int = {
    max(1, 
      (if (compressionFactor < 1)
        (x: Double) => min(x, currentDim - 1)
      else 
        (x: Double) => x)(currentDim * compressionFactor)
    ).toInt
  }
  
  // inverts the above function wrt. to compressionFactor.
  // actually, it's pretty similar to root-function, but all the 
  // ceilings make an analytical computation impossible
  // def findCompressionFactor(
  //   numLayers: Int, 
  //   dataDim: Int,
  //   compressionDim: Int
  // ): Double = {
  //   
  //   var steps = 1
  //   while (steps < 100000) {
  //     for (x <- (0 until steps).map(_ / steps)) {
  //       var currDim = dataDim
  //       for (l <- 0 until numLayers) {
  //         currDim = nextLayerDimension(x, currDim)
  //       }
  //       if (currDim == compressionDim) {
  //         return x
  //       }
  //     }
  //     steps *= 2
  //   }
  //   throw new ArithmeticException(
  //     "Could not find inverse for " + numLayers + " " + compressionDim
  //   )
  // }
  
  // def main(args: Array[String]): Unit = {
  //   val c = findCompressionFactor(2, 14, 5)
  //   println(Stream.iterate(14)(nextLayerDimension(c, _)).take(10).toList)
  // }
}