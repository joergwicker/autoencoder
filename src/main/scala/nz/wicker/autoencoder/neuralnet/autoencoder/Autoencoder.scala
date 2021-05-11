package nz.wicker.autoencoder.neuralnet.autoencoder

import nz.wicker.autoencoder.math.matrix.Mat
import nz.wicker.autoencoder.math.optimization.DifferentiableErrorFunctionFactory
import nz.wicker.autoencoder.neuralnet.Layer
import nz.wicker.autoencoder.neuralnet.NeuralNet
import nz.wicker.autoencoder.neuralnet.NeuralNetLike
import nz.wicker.autoencoder.neuralnet.rbm.RbmTrainingStrategy
import nz.wicker.autoencoder.visualization.TrainingObserver
import nz.wicker.autoencoder.neuralnet.rbm.RbmLayer
import nz.wicker.autoencoder.neuralnet.rbm.Rbm
import nz.wicker.autoencoder.neuralnet.FullBipartiteConnection
import nz.wicker.autoencoder.neuralnet.rbm.RbmStack
import nz.wicker.autoencoder.math.optimization.SquareErrorFunctionFactory

/**
 * Stack of Rbm's that approximates the identity function
 */
class Autoencoder(override val layers: List[Layer]) 
  extends NeuralNetLike[Autoencoder] with Serializable {

  override def build(layers: List[Layer]): Autoencoder = new Autoencoder(layers)
  
  val compressor = new NeuralNet(layers.take((layers.size + 1) / 2))
  val decompressor = new NeuralNet(layers.drop(layers.size / 2))
  
  def compress(uncompressed: Mat) = compressor(uncompressed)
  def decompress(compressed: Mat) = decompressor(compressed)
  
  def compressionDimension = compressor.layers.last.outputDimension
  
  /**
   * creates a new autoencoder with an additional central layer
   */
  def unfoldCentralLayer(
    newCentralLayerType: Int,
    newCentralLayerDim: Int,
    rbmTrainingStrategy: RbmTrainingStrategy,
    data: Mat,
    errorFunctionFactory: DifferentiableErrorFunctionFactory[Mat],
    fineTuneInnerLayers: Boolean,
    trainingObservers: List[TrainingObserver]
  ): Autoencoder = {
    // compress the data for the training of central autoencoder
    val compressedData = compress(data)
    
    val (diagonal, offsets) = if (layers.size > 1) {
      // find factors and offsets that map the data onto the full interval [0, 1]
      val mins = compressedData.foldRows(Double.PositiveInfinity){_ min _}
      val maxs = compressedData.foldRows(Double.NegativeInfinity){_ max _}
      val ranges = maxs - mins
      val nonzeroRanges = ranges.map{_ max 0.000001}
      val factors = nonzeroRanges.map{x => 1 / x}
      (Mat.diag(factors), -mins :/ nonzeroRanges)
    } else {
      (Mat.eye(compressedData.width), Mat.empty(1, compressedData.width, 0d))
    }
    
    // rescale the data so that it becomes more suitable for the RBM training
    val rescaledData = 
      compressedData * diagonal + Mat.ones(compressedData.height, 1) * offsets
    
    // initialize layers of the innermost sub-autoencoder 
    // (only the dimensions are relevant, other values are reinitialized
    // by the training strategy anyway)
    val oppositeVisible =
      compressor.layers.last.asInstanceOf[RbmLayer].reinitialize(0)
    val newCentral = nz.wicker.autoencoder.mkLayer(
      newCentralLayerType, 
      newCentralLayerDim
    )
    val connection = new FullBipartiteConnection(
      oppositeVisible.parameters.width, 
      newCentral.parameters.width
    )
    val rbm =
      new Rbm(
        oppositeVisible,
        connection,
        newCentral
      )
    
    // pre-train innermost autoencoder
    val trainedRbm = rbmTrainingStrategy.train(
      rbm, 
      rescaledData, 
      trainingObservers
    )
    
    // fine-tune innermost autoencoder
    val centralAutoencoder = new RbmStack(List(trainedRbm)).unfold
    
    val optimizedCentralAutoencoder = if (fineTuneInnerLayers) {
      centralAutoencoder.optimize(
        rescaledData, 
        rescaledData, 
        SquareErrorFunctionFactory, 
        0.33, 
        5000, 
        trainingObservers
      )
    } else {
      // just work with the unoptimized central autoencoder
      centralAutoencoder
    } 
    
    val (rescaledInnerAutoencoder, rescaledDecompressor) = 
      if (layers.size > 1) {
        // prepend the original affine transform to the new inner autoencoder and
        // it's inverse to the old decoder. This should prevent our inner RBM's 
        // from being braindead constant functions
        val rescaledInnerAutoencoder = 
          optimizedCentralAutoencoder.prependAffineLinearTransformation(
            diagonal, 
            offsets
          )
        
        val inverseDiagonal = diagonal.map{x => if (x == 0d) 0d else 1 / x}
        val rescaledDecompressor = 
          this.decompressor.prependAffineLinearTransformation(
            inverseDiagonal, 
            - offsets * inverseDiagonal
          )
        (rescaledInnerAutoencoder, rescaledDecompressor)
      } else {
        // do nothing. Cannot affect data preprocessing, it's outside of the
        // autoencoder itself.
        (optimizedCentralAutoencoder, this.decompressor)
      }
    
    // stuff the inner autoencoder between the already present layers
    val newLayers = 
      this.compressor.layers ++
      rescaledInnerAutoencoder.layers.tail ++ 
      rescaledDecompressor.layers.tail
    
    val moreAbstractAutoencoder = new Autoencoder(newLayers)  
    
    // fine tune the more abstract autoencoder as a whole
    val result = moreAbstractAutoencoder.optimize(
      data, 
      data, 
      errorFunctionFactory, 
      0.33, 
      5000,
      trainingObservers
    )
    
    result
  }
}