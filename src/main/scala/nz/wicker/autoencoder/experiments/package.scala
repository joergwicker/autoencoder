package nz.wicker.autoencoder

import nz.wicker.autoencoder.neuralnet.autoencoder.Autoencoder
import nz.wicker.autoencoder.math.matrix.Mat
import nz.wicker.autoencoder.thresholding._

package object experiments {

  
  def calculateRelativeError(
    autoencoder: Autoencoder, 
    data: Mat
  ) = {
  
    val compression = autoencoder.compress(data)
    val reconstruction = autoencoder.decompress(compression)
    val thresholds = findOptimalColumnThresholds(reconstruction, data)
    val binaryReconstruction = binarize(reconstruction, thresholds)
    
    val binaryDifference = binaryReconstruction - data
    val zeroToOneErrors = binaryDifference.filter{_ > 0}.normSq
    val oneToZeroErrors = binaryDifference.filter{_ < 0}.normSq
    val binaryError = binaryDifference.normSq
    
    binaryError / (data.width * data.height)
  }
}