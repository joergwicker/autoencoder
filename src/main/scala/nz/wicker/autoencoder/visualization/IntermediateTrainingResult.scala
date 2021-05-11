package nz.wicker.autoencoder.visualization

import nz.wicker.autoencoder.neuralnet.rbm.RbmStack
import nz.wicker.autoencoder.neuralnet.autoencoder.Autoencoder
import nz.wicker.autoencoder.neuralnet.rbm.Rbm
import nz.wicker.autoencoder.neuralnet.NeuralNetLike
import nz.wicker.autoencoder.neuralnet.NeuralNet

sealed trait IntermediateTrainingResult

case class PartiallyTrainedRbmStack(stack: RbmStack) 
  extends IntermediateTrainingResult
case class PartiallyTrainedRbm(rbm: Rbm)
  extends IntermediateTrainingResult
case class PartiallyTrainedAutoencoder(autoencoder: Autoencoder)
  extends IntermediateTrainingResult
case class VisualizableIntermediateResult(v: Visualizable)
  extends IntermediateTrainingResult