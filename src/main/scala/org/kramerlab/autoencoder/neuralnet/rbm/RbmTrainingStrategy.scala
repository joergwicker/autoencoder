package org.kramerlab.autoencoder.neuralnet.rbm

import org.kramerlab.autoencoder.math.matrix.Mat
import org.kramerlab.autoencoder.visualization.TrainingObserver

trait RbmTrainingStrategy {
  def train(rbm: Rbm, data: Mat, trainingObservers: List[TrainingObserver]): Rbm
}