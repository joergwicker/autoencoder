package nz.wicker

import scala.Array.canBuildFrom
import nz.wicker.autoencoder.math.matrix.Mat
import nz.wicker.autoencoder.math.optimization.CG_Rasmussen2
import nz.wicker.autoencoder.neuralnet.FullBipartiteConnection
import nz.wicker.autoencoder.neuralnet.autoencoder.Autoencoder
import nz.wicker.autoencoder.neuralnet.rbm.BernoulliUnitLayer
import nz.wicker.autoencoder.neuralnet.rbm.DefaultRbmTrainingConfiguration
import nz.wicker.autoencoder.neuralnet.rbm.GaussianUnitLayer
import nz.wicker.autoencoder.neuralnet.rbm.Rbm
import nz.wicker.autoencoder.neuralnet.rbm.RbmStack
import nz.wicker.autoencoder.neuralnet.rbm.RbmTrainingConfiguration
import nz.wicker.autoencoder.visualization.TrainingObserver
import scala.math._
import nz.wicker.autoencoder.math.optimization.Minimizer
import nz.wicker.autoencoder.math.optimization.SquareErrorFunctionFactory
import nz.wicker.autoencoder.math.optimization.DifferentiableErrorFunctionFactory
import nz.wicker.autoencoder.experiments.Metaparameters
import nz.wicker.autoencoder.neuralnet.rbm.RbmTrainingStrategy
import nz.wicker.autoencoder.neuralnet.rbm.RandomRetryTrainingStrategy
import nz.wicker.autoencoder.neuralnet.rbm.CompetitiveRetryTrainingStrategy
import nz.wicker.autoencoder.neuralnet.rbm.TournamentTrainingStrategy
import nz.wicker.autoencoder.math.optimization.CrossEntropyErrorFunctionFactory
import scala.collection.JavaConversions
import nz.wicker.autoencoder.neuralnet.rbm.ConstantConfigurationFixedEpochsTrainingStrategy

package object autoencoder {

  val Linear = 0
  val Sigmoid = 1
  
  /**
   * Empty list of observers for convenient method calls from java-code
   */
  val NoObservers: List[TrainingObserver] = Nil
  
  /**
   * Bunch of different pretraining strategy factories
   */
  val NoPretraining: () => RbmTrainingStrategy = { () =>
    new RbmTrainingStrategy() {
      def train(
        rbm: Rbm, 
        data: Mat, 
        trainingObservers: List[TrainingObserver]
      ): Rbm = rbm
    }
  }
  
  val HintonsMiraculousStrategy: () => RbmTrainingStrategy = { () =>
    new ConstantConfigurationFixedEpochsTrainingStrategy(
      new DefaultRbmTrainingConfiguration()   
    )
  }
  
  val RandomRetryStrategy: () => RbmTrainingStrategy = { () =>
    new RandomRetryTrainingStrategy(4, 0.33)
  } 
  
  val TournamentStrategy: () => RbmTrainingStrategy = { () => 
    new TournamentTrainingStrategy(8, 12, 0.5, 0.33)
  }
  
  /**
   * Trains a single autoencoder with the algorithm proposed by Hinton.
   * 
   * @param data input data with one instance per row
   * @param compressionDimension dimension of the central layer
   * @param numberOfHiddenLayers number of hidden layers between the input and 
   *     the central bottleneck
   * @param useL2Error whether to use L2 or Cross-Entropy error. If you aren't
   *                   sure what you need, set it to `true`
   * @param trainingStrategyFactory pick one of the predefined. If you don't 
   *     know which one you need: pick `HintonsMiraculousStrategy` if you need
   *     it a little faster, or `TournamentStrategy` if you need it a little 
   *     more accurate
   * @param observers list of training observers that can be used to display
   *     information about the training progress. Use `NoObservers` if you 
   *     don't need it.
   */
  def trainAutoencoder(
    data: Mat, 
    compressionDimension: Int,
    numberOfHiddenLayers: Int,
    useL2Error: Boolean,
    trainingStrategyFactory: () => RbmTrainingStrategy,
    observers: List[TrainingObserver]
  ): Autoencoder = {
    
    val params = new Metaparameters(
      inputDim = data.width,
      compressionDim = compressionDimension,
      numHidLayers = numberOfHiddenLayers,
      linearCentralLayer = false
    )
    
    val errorFunctionFactory = if (useL2Error) {
      SquareErrorFunctionFactory
    } else {
      CrossEntropyErrorFunctionFactory
    }
    
    trainAutoencoder(
      params.layerTypes,
      params.layerDims,
      data,
      List.fill(params.numHidLayers + 1) { trainingStrategyFactory() },
      errorFunctionFactory,
      5000,
      observers
    )
  }
  
  // helper method with more arguments
  private def trainAutoencoder(
    layerTypes: Array[Int], 
    layerDims: Array[Int],
    data: Mat,
    rbmTrainingStrategies: List[RbmTrainingStrategy] = 
      Nil,
    errorFunctionFactory: DifferentiableErrorFunctionFactory[Mat] = 
      SquareErrorFunctionFactory,
    maxEvals: Int, 
    trainingObservers: List[TrainingObserver]
  ): Autoencoder = {
    
    val layerDescriptions = layerTypes zip layerDims
    val rbms = 
      (for ( ((visType, visDim), (hidType, hidDim)) <- 
            layerDescriptions zip layerDescriptions.tail) yield {
        new Rbm(
          mkLayer(visType, visDim),
          new FullBipartiteConnection(visDim, hidDim),
          mkLayer(hidType, hidDim)
        )
      }).toList
    val stack = new RbmStack(rbms)
    val trainedStack = 
      stack.train(data, rbmTrainingStrategies, trainingObservers)
    val autoencoder = trainedStack.unfold

    // input: Matrix, 
    // output: Matrix,
    // errorFunctionFactory: DifferentiableErrorFunctionFactory = 
    //   SquareErrorFunctionFactory,
    // relativeValidationSetSize: Double,
    // maxEvals: Int,
    // trainingObservers: List[TrainingObserver]
    
    val fineTunedAutoencoder = autoencoder.optimize(
      data, 
      data,
      errorFunctionFactory,
      0.33,
      maxEvals,
      trainingObservers
    )
    fineTunedAutoencoder
  }
  
  protected[autoencoder] def mkLayer(
      layerType: Int, 
      layerDim: Int
  ) = layerType match {
    case Linear => new GaussianUnitLayer(layerDim)
    case Sigmoid => new BernoulliUnitLayer(layerDim)
  }
  
  /**
   * Our deviation from Hinton's training procedure,
   * based on the idea of successive fine-tuning of "almost-isomorphisms".
   */
  def deepAutoencoderStream(
    layerType: Int,
    maxDepth: Int,
    hiddenLayerDims: collection.Seq[Int], 
    data: Mat,
    useL2Error: Boolean,
    pretrainingStrategyFactory: () => RbmTrainingStrategy,
    finetuneInnerLayers: Boolean,
    trainingObservers: List[TrainingObserver]
  ): Stream[Autoencoder] = {
    
    val errorFunctionFactory = if (useL2Error) {
      SquareErrorFunctionFactory
    } else {
      CrossEntropyErrorFunctionFactory
    }
    
    val inputLayer = mkLayer(layerType, data.width)
    val trivialAutoencoder = new Autoencoder(List(inputLayer))
    
    
    hiddenLayerDims.toStream.scanLeft(trivialAutoencoder){ (a, d) =>
      val nextAutoencoder = a.unfoldCentralLayer(
        layerType, 
        d, 
        pretrainingStrategyFactory(), 
        data, 
        errorFunctionFactory, 
        finetuneInnerLayers,
        trainingObservers
      )
      
      nextAutoencoder
    }.tail.take(maxDepth)
    
  }
  
  /**
   * Same as `deepAutoencoderStream`, but with a Java-Iterable as return type.
   */
  def deepAutoencoderStream_java(
    layerType: Int,
    maxDepth: Int,
    compressionFactor: Double,
    data: Mat,
    useL2Error: Boolean,
    pretrainingStrategyFactory: () => RbmTrainingStrategy,
    finetuneInnerLayers: Boolean, 
    trainingObservers: List[TrainingObserver]
  ) = JavaConversions.asJavaIterable(
    deepAutoencoderStream(
      layerType: Int,
      maxDepth: Int,
      Stream.iterate(data.width)(
        Metaparameters.nextLayerDimension(compressionFactor, _)
      ).tail.take(maxDepth),
      data: Mat,
      useL2Error: Boolean,
      pretrainingStrategyFactory,
      finetuneInnerLayers,
      trainingObservers: List[TrainingObserver]
    )  
  ) 
  
  /**
   * Trains a single Autoencoder using our Autoencoder-Stream strategy.
   * 
   * @param data input data with one instance per row
   * @param compressionDimension dimension of the central layer
   * @param numberOfHiddenLayers number of hidden layers between the input and 
   *     the central bottleneck
   * @param useL2Error whether to use L2 or Cross-Entropy error. If you aren't
   *                   sure what you need, set it to `true`
   * @param trainingStrategyFactory pick one of the predefined. If you don't 
   *     know which one you need: pick `HintonsMiraculousStrategy` if you need
   *     it a little faster, or `TournamentStrategy` if you need it a little 
   *     more accurate
   * @param observers list of training observers that can be used to display
   *     information about the training progress. Use `NoObservers` if you 
   *     don't need it.
   */
  def trainAutoencoder_Stream(
    data: Mat, 
    compressionDimension: Int,
    numberOfHiddenLayers: Int,
    useL2Error: Boolean,
    trainingStrategyFactory: () => RbmTrainingStrategy,
    observers: List[TrainingObserver]
  ): Autoencoder = {
    
    val params = new Metaparameters(
      inputDim = data.width,
      compressionDim = compressionDimension,
      numHidLayers = numberOfHiddenLayers,
      linearCentralLayer = false
    )
    
    deepAutoencoderStream(
      layerType = Sigmoid,
      maxDepth = numberOfHiddenLayers,
      hiddenLayerDims = params.layerDims, 
      data: Mat,
      useL2Error: Boolean,
      pretrainingStrategyFactory = trainingStrategyFactory,
      finetuneInnerLayers = true,
      trainingObservers = observers
    ).last
  }
  
  /**
   * This method returns dimensions of layers for specified input dimension,
   * hidden layer dimension, number of layers, and a parameter alpha, which
   * determines, the "convexity" of the [layer-index -> layer-size] function
   * (alpha = 1 corresponds to linear
   * interpolation between number of visible and number of hidden units,
   * alpha < 1 corresponds to "slim" networks, alpha > 1 corresponds to "fat"
   * networks).
   */
  def layerDims(numVis: Int, numHid: Int, n: Int, alpha: Double): List[Int] = {
    (for (k <- 0 to n) yield {
      (numHid + (numVis - numHid) * pow(
         1 - pow(k.toDouble/n, alpha), 
         1d/alpha
       )
      ).round.toInt
    }).toList
  }
  
  /**
   * Sets number of threads in the thread pool for all parallel 
   * collections globally.
   */
  def setParallelismGlobally(numThreads: Int): Unit = {
    val parPkgObj = scala.collection.parallel.`package`
    val defaultTaskSupportField = parPkgObj.getClass.getDeclaredFields.find{
      _.getName == "defaultTaskSupport"
    }.get
    
    defaultTaskSupportField.setAccessible(true)
    defaultTaskSupportField.set(
      parPkgObj, 
      new scala.collection.parallel.ForkJoinTaskSupport(
        new scala.concurrent.forkjoin.ForkJoinPool(numThreads)
      ) 
    )
  }
  
}