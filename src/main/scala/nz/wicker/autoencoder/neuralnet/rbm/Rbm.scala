package nz.wicker.autoencoder.neuralnet.rbm

import nz.wicker.autoencoder.math.matrix.Mat
import nz.wicker.autoencoder.neuralnet._
import nz.wicker.autoencoder.math.matrix._
import scala.math.{min}
import nz.wicker.autoencoder.visualization.TrainingObserver
import nz.wicker.autoencoder.visualization.PartiallyTrainedRbm
import nz.wicker.autoencoder.math.optimization.TerminationCriterion
import nz.wicker.autoencoder.math.optimization.ResultSelector

/**
 * Represents a restricted Boltzmann machine, which consists of two layers
 * of neurons connected with a full bipartite graph.
 */
class Rbm(
  val visible: RbmLayer,
  val connection: FullBipartiteConnection,
  val hidden: RbmLayer
) extends NeuralNet(List(visible, connection, hidden))
  with NeuralNetLike[Rbm] {
  
  override def build(ls: List[Layer]): Rbm = ls match {
    case (v: RbmLayer) :: 
	     (c: FullBipartiteConnection) :: 
         (h: RbmLayer) :: Nil => {
           new Rbm(v, c, h)
    }
	case _ => 
	  throw new IllegalArgumentException("Cannot build an Rbm from " + ls)
  }
  
  /**
   * Given hidden activation, samples a hidden state, 
   * calculates visible activation, and samples visible state.
   *
   * If `sampleVisibleStatesDeterministically` is set to `true`,
   * activations of visible neurons are returned directly, no
   * random sampling occurs for visible neurons in this case.
   */
  def confabulate(
    hiddenActivation: Mat,
	sampleVisibleUnitsDeterministically: Boolean = false
  ): Mat = {
    val hiddenSample = hidden.sample(hiddenActivation)
	val visibleActivation = this.reverse(hiddenSample)
	if (sampleVisibleUnitsDeterministically) {
	  visibleActivation
	} else {
	  visible.sample(visibleActivation)
	}
  }

  /**
   * Performs Gibbs sampling `steps` times, starting with 
   * the input clamped to the visible layer.
   * 
   * If `steps` is zero, the visible input together with 
   * hidden layer activation is returned (useful for collecting positive
   * statistics in contrastive divergence).
   *
   * If `steps` is greater than zero, returns visible reconstruction and 
   * exact hidden layer activations after specified number of steps.
   *
   * If `sampleVisibleUnitsDeterministically` is set to `true`, then 
   * activations of the visible units are used instead of random samples.
   *
   * The hidden units are always updated randomly.
   */
  def gibbsSampling(
    visibleStates: Mat, 
	steps: Int = 1, 
	sampleVisibleUnitsDeterministically: Boolean = false
  ): (Mat, Mat) = {
    
    val hiddenActivations = this(visibleStates)
    
    if (steps == 0) {
	  (visibleStates, hiddenActivations)
	} else {
	  val visibleSamples = confabulate(
	    hiddenActivations,
		sampleVisibleUnitsDeterministically
	  )
	  gibbsSampling(
	    visibleSamples, 
		steps - 1, 
		sampleVisibleUnitsDeterministically
	  )
	}
  }
  
  /**
   * Extracts 
   * average visible unit state,
   * average hidden unit activation, and
   * average products of visible states and hidden activations
   * from 
   * the samples of visible units and 
   * hidden activations.
   *
   * The dimensions of extracted matrices correspond to
   * dimensions of visible biases, weight Mat, and hidden
   * biases respectively.
   */
  protected def extractStatistics(
    visibleStates: Mat, 
	hiddenActivations: Mat
  ): (Mat, Mat, Mat) = {
    val n = visibleStates.height // numberOfSamplesInMinibatch 
	val averageVisibleStates = visibleStates.sumRows / n
	val averageHiddenActivations = hiddenActivations.sumRows / n
	val averageProducts = visibleStates.transpose * hiddenActivations / n

	(averageVisibleStates, averageHiddenActivations, averageProducts)
  }

  /**
   * Calculates (very coarse) approximations of 
   * partial derivatives of the logarithmized probability of
   * the minibatch w.r.t. biases of the unit layers and 
   * weights of the connection layer.
   * 
   * @return tuple containing three matrices that correspond to
   *         derivatives w.r.t. biases of visible units, weights,
   *         and hidden units respectively.
   */
  protected def contrastiveDivergence(
    minibatch: Mat, 
	steps: Int = 1, 
	sampleVisibleUnitsDeterministically: Boolean = false
  ): (Mat, Mat, Mat) = {
    
    val (positiveVisibleStates, positiveHiddenActivations) =
	  gibbsSampling(minibatch, 0, sampleVisibleUnitsDeterministically)
	val (negativeVisibleSamples, negativeHiddenActivations) =
	  gibbsSampling(
	    confabulate(
		  positiveHiddenActivations, 
		  sampleVisibleUnitsDeterministically
		),
		steps - 1,
		sampleVisibleUnitsDeterministically
	  ) 

    val (
	  positiveAverageVisibleStates,
	  positiveAverageHiddenActivations,
	  positiveAverageProducts 
	) = extractStatistics(positiveVisibleStates, positiveHiddenActivations)

	val (
	  negativeAverageVisibleSamples,
	  negativeAverageHiddenActivations,
	  negativeAverageProducts
	) = extractStatistics(negativeVisibleSamples, negativeHiddenActivations)

    (positiveAverageVisibleStates - negativeAverageVisibleSamples,
	 positiveAverageProducts - negativeAverageProducts,
	 positiveAverageHiddenActivations - negativeAverageHiddenActivations)
  }

  /**
   * Trains this Rbm with the data contained in the minibatches with 
   * parameters specified in the configuration.
   * 
   * Returns the training data processed by the trained Rbm
   */
  def train[Fitness: Ordering](
    trainingSet: Mat,
    configuration: RbmTrainingConfiguration,
    trainingObservers: List[TrainingObserver] = Nil,
    terminationCriterion: TerminationCriterion[Rbm, Int],
    resultSelector: ResultSelector[Rbm, Fitness]
  ): Rbm = {
    
    resultSelector.consider(this)
    
	// velocities for biases and weights
	var visibleVelocity = visible.parameters.zero
	var hiddenVelocity = hidden.parameters.zero
	var weightVelocity = connection.parameters.zero

	// run through all minibatches `epoch` number of times
	val startTime = System.currentTimeMillis
	var lastMessageTime = Long.MinValue / 10
	var epoch = 0
	
	while(!terminationCriterion(this, epoch)) {
 
	  val currentMomentum = configuration.momentum(epoch)

	  trainingSet.shuffleRows()
	  val minibatches = Rbm.cutDataIntoMinibatches(
	    trainingSet, 
	    configuration.minibatchSize
	  )
	  
	  var minibatchIndex = 0
	  for (minibatch <- minibatches) {
	    
	    // TODO: move it all into some observer
	    // val currentTime = System.currentTimeMillis
	    // 
	    // if (currentTime - lastMessageTime > 60000) {
	    //   println(
	    //     "Epoch: " + epoch + 
	    //     " minibatch: " + minibatchIndex + 
	    //     " time passed: " + (currentTime - startTime) / 60000 + " min"
	    //   )
	    //   lastMessageTime = currentTime
	    // }
	    
	    
        if (connection.parameters.l2Norm.isNaN) {
          throw new Exception(
            "Weights exploded in epoch = " + epoch + 
            " batch = " + minibatchIndex)
        }
        
	    val (visibleGrad, weightsGrad, hiddenGrad) = 
		  contrastiveDivergence(
		    minibatch, 
		    configuration.gibbsSamplingSteps(epoch),
		    configuration.sampleVisibleUnitsDeterministically
		  )
		
	    // the gradient is used as a sort of "acceleration" in an 
		// environment with friction. It changes the velocities. The 
		// velocities in turn are used to change the actual weights and 
		// biases.
		  
		weightVelocity = 
		  weightVelocity * currentMomentum +
		  (weightsGrad - configuration.weightPenalty(connection.parameters)) *
		  configuration.learningRate
		  
		visibleVelocity = 
		  visibleVelocity * currentMomentum + 
          (visibleGrad - configuration.weightPenalty(visible.parameters)) * 
          configuration.learningRate
          
        hiddenVelocity = 
          hiddenVelocity * currentMomentum +
          (hiddenGrad - configuration.weightPenalty(hidden.parameters)) *
          configuration.learningRate
          
        connection.parameters += weightVelocity
        visible.parameters += visibleVelocity
        hidden.parameters += hiddenVelocity

        val avgVisibleBias = 
          visible.parameters.transpose.sumRows / visible.parameters.width
        val avgHiddenBias = 
          hidden.parameters.transpose.sumRows / hidden.parameters.width
        
        for (obs <- trainingObservers) {
          val firstHundredRows = 
            trainingSet(0 ::: min(100, trainingSet.height), 0 ::: end)
          firstHundredRows.shuffleRows()
          this.dataSample = Some(firstHundredRows(0 ::: 10, 0 ::: end)) 
          obs.notify(PartiallyTrainedRbm(this), false)
        }
	    
	    minibatchIndex += 1
      }
	  
	  resultSelector.consider{this.clone}
	  epoch += 1
	}
    
    resultSelector.result
  }
  
  override def toString = {
      "RBM[visible: " + visible.parameters.height + "x" + 
      visible.parameters.width + " (" + 
      visible.getClass.getSimpleName + ")" +
      " connections: " + connection.parameters.height + "x" + 
      connection.parameters.width +
      " hidden:" +  
      hidden.parameters.height + "x" + 
      hidden.parameters.width + " (" + 
      hidden.getClass.getSimpleName + ")]"
  }
  
  override def clone: Rbm = new Rbm(visible.copy, connection.copy, hidden.copy)
  
  def reinitialize(config: RbmTrainingConfiguration): Rbm = {
    new Rbm(
      visible.reinitialize(config.initialBiasScaling),
      connection.reinitialize(config.initialWeightScaling),
      hidden.reinitialize(config.initialBiasScaling)
    )
  }
}

object Rbm {
  def cutDataIntoMinibatches(data: Mat, minibatchSize: Int): List[Mat] = {
    data.shuffleRows()
    def rec(remainingRows: Int): List[Mat] = {
      if (remainingRows <= minibatchSize) {
        List(data(0 ::: remainingRows, 0 ::: data.width))
      } else {
        data(
          (remainingRows - minibatchSize) ::: remainingRows, 
          0 ::: data.width
        ) :: 
        rec(remainingRows - minibatchSize)
      }
    }
    rec(data.height)
  }
}