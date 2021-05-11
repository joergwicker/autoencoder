package nz.wicker.autoencoder.neuralnet.rbm

import nz.wicker.autoencoder.math.matrix.Mat
import nz.wicker.autoencoder.visualization.TrainingObserver

trait RbmTrainingStrategy {
  def train(rbm: Rbm, data: Mat, trainingObservers: List[TrainingObserver]): Rbm
}