package org.kramerlab.autoencoder.visualization

import org.kramerlab.autoencoder.neuralnet.rbm.RbmStack
import org.kramerlab.autoencoder.neuralnet.autoencoder.Autoencoder
import org.kramerlab.autoencoder.neuralnet.rbm.Rbm
import org.kramerlab.autoencoder.neuralnet.NeuralNetLike
import org.kramerlab.autoencoder.neuralnet.NeuralNet

sealed trait IntermediateTrainingResult

case class PartiallyTrainedRbmStack(stack: RbmStack) 
  extends IntermediateTrainingResult
case class PartiallyTrainedRbm(rbm: Rbm)
  extends IntermediateTrainingResult
case class PartiallyTrainedAutoencoder(autoencoder: Autoencoder)
  extends IntermediateTrainingResult
case class VisualizableIntermediateResult(v: Visualizable)
  extends IntermediateTrainingResult