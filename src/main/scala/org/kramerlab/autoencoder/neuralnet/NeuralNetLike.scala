package org.kramerlab.autoencoder.neuralnet

import java.awt.Graphics2D

import java.awt.RenderingHints
import java.awt.image.BufferedImage
import org.kramerlab.autoencoder.math.matrix.Mat
import org.kramerlab.autoencoder.math.matrix._
import org.kramerlab.autoencoder.math.optimization._
import org.kramerlab.autoencoder.math.optimization.EarlyStopping
import org.kramerlab.autoencoder.math.structure.VectorSpace
import org.kramerlab.autoencoder.visualization.Visualizable
import org.kramerlab.autoencoder.visualization.TrainingObserver
import org.kramerlab.autoencoder.visualization.Observer
import org.kramerlab.autoencoder.visualization.VisualizableIntermediateResult
import scala.math.{min, max}
import java.awt.Color

/**
 * Implementation trait for all subclasses of neural net.
 * Implements everything that is necessary to perform 
 * parameter optimization (feed forward, backpropagation,
 * minimization with non-linear conjugate gradient or 
 * similar algorithm).
 *
 * For those not familiar with the "-Like"-implementation
 * trait pattern: it's main purpose is to ensure that 
 * the results of optimization have the right type for 
 * all subclasses of neural net (so that one gets an 
 * optimized `Autoencoder` after running the minimization,
 * not just some abstract neural net). For this, it keeps the 
 * information about the concrete representation type 
 * `Repr`.
 * 
 * Subclasses that inherit the backpropagation functionality
 * from this implementation trait have to provide a `build` 
 * method, which takes a list of `Layer`s and returns the 
 * right flavor of neural net built from those layers.
 */
trait NeuralNetLike[+Repr <: NeuralNetLike[Repr]] 
  extends (Mat => Mat)
  with Visualizable {

  self: Repr =>

  /**
   * Enumerates layers of this (linear) neural net.
   *
   * TODO: generalize it to arbitrary directed acyclic graphs,
   * what's so special about lists?...
   */
  def layers: List[Layer]
 
  /**
   * Propagates the input from the visible layer up to the top layer
   */
  override def apply(input: Mat): Mat = {
    layers.tail.foldLeft(input) {(in, layer) => layer.propagate(in)}
  } 
  
  def activities(input: Mat): List[Mat] = {
    layers.tail.scanLeft(input){(in, layer) => layer.propagate(in)}
  }

  /**
   * Propagates the output from top layer down to the visible layer
   */
  def reverse(output: Mat): Mat = {
    layers.reverse.tail.foldLeft(output) {
	  (out, layer) => layer.reversePropagate(out)
	}
  }
  /**
   * Builds a neural net of the right type and of the right shape
   * out of specified layers.
   *
   * Note that this method depends on instance, not just a class:
   * fore example the `Autoencoder` has to know what it's 'central'
   * Layer is.
   */
  def build(layers: List[Layer]): Repr

  /** 
   * Performs optimization of all parameters of the neural network
   * using the specified `input` and `output`, the specified method
   * to define an error function (defaults to `SquareErrorFunctionFactory`).
   *
   * Standard feed-forward algorithm is used for evaluation of the function,
   * backpropagation is used for calculation of the gradient.
   */
  def optimize(
    input: Mat, 
	output: Mat,
	errorFunctionFactory: DifferentiableErrorFunctionFactory[Mat] = 
	  SquareErrorFunctionFactory,
	relativeValidationSetSize: Double,
	maxEvals: Int,
	trainingObservers: List[TrainingObserver]
  ): Repr = {
    
    import NeuralNetLike._
    
    if (relativeValidationSetSize == 0) {
      throw new Error("Training without a validation set is currently not " +
      		"supported, but can be added easily.")
      
      //// just optimize as long as possible without a validation set
      //val errorFunction = errorFunctionFactory(output)
      //import NeuralNetLike._
      //val minimizableFunction: DifferentiableFunction[ParameterVector[Repr]] = 
      //  differentiableComposition[Repr](input, errorFunction)
      //
      //val adjustedObservers = trainingObservers.map{ obs => 
      //  new Observer[ParameterVector[Repr]] {
      //    override def notify(
      //      r: ParameterVector[Repr], 
      //      important: Boolean
      //    ): Unit = {
      //      val firstHundredLines = 
      //        input(0 ::: min(input.height, 100), 0 ::: end)
      //      firstHundredLines.shuffleRows()
      //      val sample = firstHundredLines(
      //        0 ::: min(firstHundredLines.height, 10), 
      //        0 ::: end
      //      )
      //      r.net.dataSample = Some(sample)
      //      obs.notify(VisualizableIntermediateResult(r.net), important)
      //    }
      //  }
      //}
      //
      //val minimizer = CG_Rasmussen2_WithTermination
      //minimizer.minimize(
      //  minimizableFunction, 
      //  ParameterVector(this), 
      //  adjustedObservers
      //).net
    } else {
      // use holdout, split the input and output into the actual training set
      // and the validation set
      val permutation = 
        org.kramerlab.autoencoder.math.random.permutation(input.height)
      
      val shuffledInput = input.clone
      val shuffledOutput = output.clone
        
      shuffledInput.permutateRows(permutation)
      shuffledOutput.permutateRows(permutation)
      val validationHeight = (input.height * relativeValidationSetSize).toInt
      val validationInput = shuffledInput(0 ::: validationHeight, :::)
      val validationOutput = shuffledOutput(0 ::: validationHeight, :::)
      val trainingInput = shuffledInput(validationHeight ::: end, :::)
      val trainingOutput = shuffledOutput(validationHeight ::: end, :::)
      
      // create the function that is to minimize, and the function that
      // is evaluated to test the fitness on validation set
      val errorFunction = errorFunctionFactory(trainingOutput)
      val minimizableFunction: DifferentiableFunction[ParameterVector[Repr]] = 
        differentiableComposition[Repr](trainingInput, errorFunction)
    
      val errorOnValidationSet: DifferentiableFunction[ParameterVector[Repr]] = 
        differentiableComposition(
          validationInput, 
          errorFunctionFactory(validationOutput)
        )
      
      // create early stopping strategy
      val earlyStopping = new EarlyStopping(
          (x:ParameterVector[Repr]) => -errorOnValidationSet(x),
          numberOfInitialSteps = 64,
          fitnessReevaluationInterval = 16,
          maxStepsWithoutNewRecord = 512,
          maxStepsWithoutImprovement = 128
      )
      
      // combine the early stopping with another termination criterion that
      // limits the total number of function evaluations
      val evalsLimit = new LimitNumberOfEvaluations(maxEvals)
      val terminationCriterion = evalsLimit | earlyStopping
      
      // adjusting the observers: boring wrapping-unwrapping, nothing 
      // substantial
      val adjustedObservers = trainingObservers.map{ obs => 
        new Observer[ParameterVector[Repr]] {
          override def notify(
            r: ParameterVector[Repr], 
            important: Boolean
          ): Unit = {
            val firstHundredLines = 
              input(0 ::: min(input.height, 100), 0 ::: end)
            firstHundredLines.shuffleRows()
            val sample = firstHundredLines(
              0 ::: min(firstHundredLines.height, 10), 
              0 ::: end
            )
            r.net.dataSample = Some(sample)
            obs.notify(VisualizableIntermediateResult(r.net), important)
          }
        }
      }
      
      // configure the minimizer
      val minimizer = new CG_Rasmussen2_WithTermination
      
      // launch the machinery
      minimizer.minimize[ParameterVector[Repr], Double](
        minimizableFunction, 
        ParameterVector(this), 
        terminationCriterion, 
        earlyStopping, 
        adjustedObservers
      ).net
    }
  }
  
  override def toImage = {
    val layerImages = layers.map{_.toImage}
    val activityImages = dataSample match {
      case Some(d) => {
        val neuralActivities = activities(d)
        (neuralActivities zip layers).map{ case (activity, layer) =>
          layer.visualizeActivity(activity)
        }
      }
      case None => layerImages.map{x => 
        new BufferedImage(1, 1, BufferedImage.TYPE_INT_RGB)
      }
    }
    val layerHeights = layerImages.map(_.getHeight)
    val activityHeights = activityImages.map{_.getHeight}
    val heights = for ((lh, ah) <- layerHeights zip activityHeights) yield {
      math.max(lh, ah)
    }
    val unitHeight = heights.sum
    val totalPadding = unitHeight / 10
    val singlePadding = math.max(12, totalPadding / (layers.size + 1))
    val borderWidth = singlePadding / 3
    val h = singlePadding * (layers.size + 1) + unitHeight
    val maxLayerWidth = (for ((l,a) <- layerImages zip activityImages) yield {
      l.getWidth + a.getWidth + borderWidth
    }).max
    val w = 3 * singlePadding + maxLayerWidth
    val offsets = heights.scanLeft(singlePadding)(_ + _ + singlePadding)
    
    val img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB)
    val g = img.getGraphics.asInstanceOf[Graphics2D]
    g.setRenderingHint(
      RenderingHints.KEY_ANTIALIASING, 
      RenderingHints.VALUE_ANTIALIAS_ON
    )
    g.setColor(Color.BLACK)
    g.fillRect(0, 0, w, h);
    g.setColor(Color.DARK_GRAY)
    
    for (
      ((layerImg, activityImg), offset) <- 
        (layerImages zip activityImages) zip offsets
    ) {
      
      val leftImageStartX = singlePadding + 
        (maxLayerWidth - 
          layerImg.getWidth - 
          activityImg.getWidth - 
          borderWidth
        ) / 2 
      
      g.fillRoundRect(
        leftImageStartX - borderWidth,
        offset - borderWidth,
        layerImg.getWidth + activityImg.getWidth + 3 * borderWidth,
        math.max(layerImg.getHeight, activityImg.getHeight) + 2 * borderWidth,
        borderWidth,
        borderWidth
      )
      g.drawImage(
        layerImg, 
        leftImageStartX,
        offset,
        layerImg.getWidth,
        layerImg.getHeight,
        null
      )
      g.drawImage(
        activityImg, 
        leftImageStartX + layerImg.getWidth + borderWidth,
        offset,
        activityImg.getWidth,
        activityImg.getHeight,
        null
      )
      
    }

    img
  }
  
  var dataSample: Option[Mat] = None
  
  /** 
   *  Assumes that this is a "usual" neural net with alternating unit and
   *  connection layers and prepends an affine linear transformation to it.
   *  
   *  Why the heck did I implement biased layers at all, why didn't I stuff 
   *  all this cruft into something like "AffineLinearTransform" or so... Damn
   */
  def prependAffineLinearTransformation(factor: Mat, offset: Mat): Repr = {
    val zerothIrrelevantLayer = layers(0)
    val firstConnectionLayer = layers(1).asInstanceOf[FullBipartiteConnection]
    val secondUnitLayer = layers(2).asInstanceOf[BiasedUnitLayer]
    val transformedOffset = firstConnectionLayer.propagate(offset)
    val modifiedBias = secondUnitLayer.parameters + transformedOffset
    val modifiedWeights = factor * firstConnectionLayer.parameters
    val newConnectionLayer = firstConnectionLayer.build(modifiedWeights)
    val newUnitLayer = secondUnitLayer.build(modifiedBias)
    val newLayers = 
      zerothIrrelevantLayer ::
      newConnectionLayer ::
      newUnitLayer ::
      layers.drop(3)
    build(newLayers)
  }
}

object NeuralNetLike {

  /**
   * An ad hoc vector space structure on the parameters of the layers
   * of the neural nets.
   */
  case class ParameterVector[Repr<: NeuralNetLike[Repr]](net: Repr)
    extends VectorSpace[ParameterVector[Repr]] {
   
    private def binOp(
	  f: (Layer, Layer) => Layer
    )(other: ParameterVector[Repr]): ParameterVector[Repr] = {
      new ParameterVector(net.build(
	    for((a,b) <- this.net.layers zip other.net.layers) yield f(a, b)
	  ))
	}

	private def unOp(f: Layer => Layer): ParameterVector[Repr] = {
      new ParameterVector(net.build(net.layers.map(f)))
	}

    override def +(other: ParameterVector[Repr]) = binOp(_ + _)(other)
    override def -(other: ParameterVector[Repr]) = binOp(_ - _)(other)
    override def *(d: Double) = unOp(_ * d)
    override def /(d: Double) = unOp(_ / d)
    override def unary_- = unOp(-_)
	override def zero = unOp(_.zero)
    override def dot(other: ParameterVector[Repr]) = {
      (for((a, b) <- net.layers zip other.net.layers) yield (a dot b)).sum
    }

    override def isNaN = net.layers.exists(_.isNaN)
    override def isInfinite = net.layers.exists(_.isInfinite)
    override def isInvalid = net.layers.exists(_.isInvalid)
    
	override def toString = net.toString
  }

  /**
   * Given a fixed data and a differentiable error 
   * function for matrix-valued outputs, returns a
   * differentiable function that takes wrapped neural nets 
   * as arguments, and returns neural net valued 
   * gradients (again wrapped as `ParameterVector`).
   * 
   * For this, the argument neural net is composed 
   * with the error function and applied to the fixed data.
   *
   * Symbolically, if `x` is our neural net, `d` is the data, 
   * and `E` our error function, this method returns the function
   * `x => E(x(d))`, which is evaluated in usual feed-forward manner.
   * The gradient of this function is evaluated with the classical 
   * backpropagation algorithm. The whole composition might look 
   * somewhat odd, but keep in mind that we want a function that 
   * takes neural nets as inputs and returns the errors on data as 
   * outputs, and furthermore calculates neural-net-valued gradients,
   * which simply store the gradients wrt. parameters of the neural
   * net in a data structure that looks exactly like the neural net itself.
   */
  def differentiableComposition[Repr <: NeuralNetLike[Repr]](
    data: Mat, 
	errorFunction: DifferentiableFunction[Mat]):
	DifferentiableFunction[ParameterVector[Repr]] = {

    new DifferentiableFunction[ParameterVector[Repr]] {
      override def apply(x: ParameterVector[Repr]) = errorFunction(x.net(data))
	  override def grad(x: ParameterVector[Repr]) = valueAndGrad(x)._2
	  override def valueAndGrad(x: ParameterVector[Repr]) = {
        def rec(
		  remainingLayers: List[Layer],
          input: Mat
		): (Double, Mat, List[Layer]) = {
          
          if (remainingLayers.isEmpty) {
            val (error, errorGrad) = errorFunction.valueAndGrad(input)
            (error, errorGrad, Nil)
          } else {
            val currentLayer :: tail = remainingLayers
            val nextInput = currentLayer.propagate(input)
            val (error, backpropFromTop, upperLevelGradients) = 
              rec(tail, nextInput)
            val (currentGrad, nextBackpropError) = 
              currentLayer.gradAndBackpropagationError(backpropFromTop)
            (error, nextBackpropError, currentGrad :: upperLevelGradients)
          }
        }
          
        val visible :: hidden = x.net.layers
        val (error, _, hiddenGrads) = rec(hidden, data)
        (error, new ParameterVector(x.net.build(visible.zero :: hiddenGrads)))
	  }
	}
  }
}
