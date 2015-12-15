package org.kramerlab.autoencoder.neuralnet

import org.kramerlab.autoencoder.math.matrix._
import org.kramerlab.autoencoder.math.matrix.Mat
import org.kramerlab.autoencoder.math.structure.VectorSpace
import org.kramerlab.autoencoder.visualization.Visualizable
import java.awt.Image
import java.awt.image.BufferedImage

/**
 * Represents a single layer of a neural network. The sort of layer meant 
 * here does not necessary contain any neurons, instead we think of a layer
 * as of a filter in a long pipe: it can contain neurons that are activated 
 * with some activation function, but it can also contain only connections 
 * between the layer below and the layer above. 
 * 
 * Common to all layers is that they are parameterized by something that 
 * is isomorphic to Real^n, that they
 * know how to transform their input into output (or transmit the signal in
 * opposite direction), and to calculate entries of the gradient of the error
 * function that correspond to their parameters, given partial derivatives wrt.
 * their output passed from above. With other words: each layer knows how to 
 * propagate all signals in feed-forward manner, and how to propagate errors 
 * backwards. 
 */
trait Layer extends VectorSpace[Layer] with Visualizable {
 
  // signal propagation methods
  
  /**
   * Returns the output given the input. This method can cache data
   * that could be useful on the second pass of the backpropagation.
   *
   * The input contains one example in each row, the output shall have the
   * same layout.
   */
  def propagate(input: Mat): Mat

  /**
   * Returns the result of signal propagation in reverse direction
   */
  def reversePropagate(output: Mat): Mat

  /**
   * Returns the gradient (Layer-valued) and the backpropagated
   * error, which is passed to the layer below.
   * 
   * This method can rely on the fact that the `propagate` method
   * already has been called in the first pass.
   *
   * @param backpropagatedError error propagated from above, formatted 
   *   the same way (one row for each example) as input and output
   * @returns gradient (Layer-valued) and the next backpropagated error
   */
  def gradAndBackpropagationError(backpropagatedError: Mat): (Layer, Mat)
  
  /**
   * Creates a new independent layer that has the same type as this one, 
   * but propagates the information in reverse direction
   */
  def reverseLayer: Layer

  def inputDimension: Int
  def outputDimension: Int
  
  /**
   * Optionally, one can specify how to reshape the neuron activities for 
   * visualization (height, width).
   */
  def activityShape: Option[(Int, Int)] = None
  
  /**
   * Color map for the activities
   */
  def activityColorscheme: Double => Int = 
    org.kramerlab.autoencoder.visualization.defaultColorscheme
    
  def visualizeActivity(activity: Mat): BufferedImage = {
    activityShape match {
      case None => activity.toImage(activityColorscheme)
      case Some((h, w)) => {
        val numExamples = activity.height
        val result = 
          new BufferedImage(h, w * numExamples, BufferedImage.TYPE_INT_RGB)
        val g = result.getGraphics()
        for (r <- 0 until activity.width) {
          val img = activity(r ::: (r + 1), 0 ::: end).
            reshape(h, w).
            toImage(activityColorscheme)
          g.drawImage(img, w * r, 0, w, h, null)
        }
        result
      }
    }
  }
}