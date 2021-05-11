package nz.wicker.autoencoder.visualization

import java.awt.image.BufferedImage
import java.awt.Color
import java.awt.Graphics2D

trait Visualizable {
  def toImage(colormap: Double => Int): BufferedImage = toImage
  def toImage: BufferedImage
  def toImage(w: Int, h: Int): BufferedImage = {
    val result = new BufferedImage(w, h, BufferedImage.TYPE_INT_ARGB)
    val original = toImage
    val g = result.getGraphics().asInstanceOf[Graphics2D]
    g.setRenderingHint(
      java.awt.RenderingHints.KEY_ANTIALIASING,
      java.awt.RenderingHints.VALUE_ANTIALIAS_ON
    )
    g.drawImage(original, 0, 0, original.getWidth, original.getHeight, null)
    result
  }
}