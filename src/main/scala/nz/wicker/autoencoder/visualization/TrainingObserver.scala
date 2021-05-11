package nz.wicker.autoencoder.visualization

trait TrainingObserver extends Observer[IntermediateTrainingResult] {
  def notify(currentState: IntermediateTrainingResult, important: Boolean): Unit
}