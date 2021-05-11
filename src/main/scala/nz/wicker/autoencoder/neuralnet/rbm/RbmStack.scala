package nz.wicker.autoencoder.neuralnet.rbm

import java.awt.Graphics2D
import java.awt.RenderingHints
import java.awt.image.BufferedImage

import nz.wicker.autoencoder.math.matrix.Mat
import nz.wicker.autoencoder.neuralnet.FullBipartiteConnection
import nz.wicker.autoencoder.neuralnet.autoencoder.Autoencoder
import nz.wicker.autoencoder.visualization.PartiallyTrainedRbmStack
import nz.wicker.autoencoder.visualization.TrainingObserver
import nz.wicker.autoencoder.visualization.Visualizable

/**
 * Encapsulates training procedures for stacks of Rbm's.
 * The Rbm's must be stored in the following way: the 
 * innermost Rbm of the future autoencoder (or: the top Rbm of the stack) must
 * be the last element of the list. The Rbm at the bottom, which is clamped
 * to the input, must be the first element. It's assumed that all dimensions
 * of the Rbm's are compatible. Please keep in mind that the signals are 
 * stored in row-vectors, which are propagated by multiplication from the right
 * by the weight matrices.
 * 
 */
class RbmStack(rbms: List[Rbm]) extends Visualizable {

  /**
   * Trains the whole stack of RBM's with Gibb's sampling-like methods
   */
  def train(
    data: Mat, 
    trainingStrategies: List[RbmTrainingStrategy],
    trainingObservers: List[TrainingObserver]
  ): RbmStack = {
    import RbmStack._
    def rec(
      alreadyTrained: List[Rbm],
      untrained: List[Rbm], 
      remainingStrategies: List[RbmTrainingStrategy],
      data: Mat
    ): List[Rbm] = {
      
      for (obs <- trainingObservers) {
        obs.notify(PartiallyTrainedRbmStack(this), true)
      }
      
      (alreadyTrained, untrained) match {
        case (all, Nil) => { 
          all.reverse
        }
        case (trained, rbm :: tail) => {
          val currentStrategy = remainingStrategies.head
          
          val newTrainedRbm = currentStrategy.train(
            rbm, 
            data, 
            trainingObservers
          )
          
          val nextLevelData = newTrainedRbm(data)
          rec(
            newTrainedRbm :: trained, 
            tail, 
            remainingStrategies.tail, 
            nextLevelData
          )
        }
      }
    }
    
    new RbmStack(rec(Nil, rbms, trainingStrategies, data))
  }
  
  /**
   * Unfolds this RbmStack into an Autoencoder. The autoencoder has to
   * be trained further by standard backpropagation methods
   */
  def unfold: Autoencoder = {
    val bottomHalf = 
      rbms.head.layers ++ rbms.tail.flatMap{_.layers.tail}
    val topHalf = 
      rbms.reverse.flatMap{_.layers.reverse.tail}.map{_.reverseLayer}
    new Autoencoder(bottomHalf ++ topHalf)
  }
  
  override def toString = {
    rbms.map{_.toString}.mkString("RbmStack[\n  ","\n  ","\n]") 
  }
  
  override def toImage = {
    val layerImages = rbms.map{_.toImage}
    val heights = layerImages.map(_.getHeight)
    val unitHeight = heights.sum
    val totalPadding = unitHeight / 10
    val singlePadding = scala.math.min(1, totalPadding / (rbms.size + 1))
    val h = singlePadding * (rbms.size + 1) + unitHeight
    val maxLayerWidth = layerImages.map{_.getWidth}.max
    val w = 2 * singlePadding + maxLayerWidth
    val offsets = heights.scanLeft(singlePadding)(_ + _ + singlePadding)
    
    val img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB)
    val g = img.getGraphics.asInstanceOf[Graphics2D]
    g.setRenderingHint(
      RenderingHints.KEY_ANTIALIASING, 
      RenderingHints.VALUE_ANTIALIAS_ON
    )
    
    for ((layerImg, offset) <- layerImages zip offsets) {
      g.drawImage(
        layerImg, 
        singlePadding + (maxLayerWidth - layerImg.getWidth) / 2,
        offset,
        layerImg.getWidth,
        layerImg.getHeight,
        null
      )
    }
    
    img
  }
}

object RbmStack {
  def apply(unitLayers: List[RbmLayer]) = {
    val rbms = (for ((v, h) <- unitLayers.tail zip unitLayers) yield {
      new Rbm(
        v, 
        new FullBipartiteConnection(
          v.parameters.width, 
          h.parameters.width
        ),
        h
      )
    }).toList
    new RbmStack(rbms)
  }
}