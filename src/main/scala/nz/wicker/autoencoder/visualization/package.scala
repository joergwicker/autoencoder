package nz.wicker.autoencoder

import java.awt.Dimension
import java.awt.Graphics2D
import java.awt.Image
import java.awt.image.BufferedImage

import scala.math._
import scala.swing.Component
import scala.swing.Frame

import nz.wicker.autoencoder.math.matrix.Mat

package object visualization {

  val defaultColorscheme: (Double => Int) = (x: Double) => {
    val blue = (127 * (1 - cos(10 * x))).toInt
    if (x < 0) {
      val red = (128 * (1 - exp(x))).toInt
      0xFF000000 + red << 16 + blue
    } else if (x > 1) {
      val green = (128 * (1 - exp(-(x - 1)))).toInt
      0xFF000000 + green << 8 + blue
    } else {
      val intensity = (255 * x).toInt
      0xFF000000 + intensity * 0x010100 + blue
    }
  }
  
  def draw(
    mat: Mat, 
    colorscheme: Double => Int = defaultColorscheme
  ): BufferedImage = {
    val w = mat.width
    val h = mat.height
    val img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB)
    
    mat.foreachWithIndex { (r, c, x) => 
      val rgb = colorscheme(x)
      img.setRGB(c, r, rgb)
    }
    
    img
  }
  
  def show(img: Image, plotTitle: String): Unit = {
    val frame = new Frame {
        title = plotTitle
        contents = new Component {
          override def paint(g: Graphics2D): Unit = {
            val (w, h) = {
              val dim = this.size
              (dim.getWidth().toInt, dim.getHeight().toInt)
            }
            g.drawImage(img, 0, 0, w, h, null)
          }
        }
      }
      
      frame.size = new Dimension(img.getWidth(null), img.getHeight(null))
      frame.location = new scala.swing.Point(
        (scala.math.random * 1000).toInt, 
        (scala.math.random * 800).toInt
      )
      frame.visible = true
  }
}