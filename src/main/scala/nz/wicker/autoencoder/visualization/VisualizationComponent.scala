package nz.wicker.autoencoder.visualization

import scala.swing.Component
import java.awt.Graphics2D
import java.awt.Image

class VisualizationComponent extends Component with TrainingObserver {

  val skipSteps = 20
  val importantEventCooldown = 32
  var lastImportantStep = 0
  var currentStep = 0
  var currentImage: Image = _
  
  override def paint(g: Graphics2D) {
    val (w, h) = {val dim = this.size; (dim.getWidth(), dim.getHeight())}
    
    g.drawImage(currentImage, 0, 0, w.toInt, h.toInt, null)
  }
  
  override def notify(
    intermediateResult: IntermediateTrainingResult, 
    important: Boolean
  ) {
    if (important) {
      lastImportantStep = currentStep
    }
    if (
      currentStep % skipSteps == 0 || 
      (currentStep - lastImportantStep) < importantEventCooldown
    ) {

      val img = intermediateResult match {
        case PartiallyTrainedRbmStack(stack) => stack.toImage
        case PartiallyTrainedRbm(rbm) => rbm.toImage
        case PartiallyTrainedAutoencoder(autoencoder) => autoencoder.toImage
        case VisualizableIntermediateResult(v) => v.toImage
      }
      
      currentImage = img
      repaint()
    }
    currentStep += 1
  }
}
