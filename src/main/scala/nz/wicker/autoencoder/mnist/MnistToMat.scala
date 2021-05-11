package nz.wicker.autoencoder.mnist

import nz.wicker.autoencoder.math.matrix.Mat
import java.io.BufferedReader
import java.io.FileReader
import java.awt.image.BufferedImage
import scala.math.min
import nz.wicker.autoencoder.math.matrix._
import java.io.BufferedInputStream
import java.io.FileInputStream
import java.io.DataInputStream

/**
 * This component loads images from MNIST format and returns them as dense 
 * matrix (one row per image). The values are scaled to the interval [0, 1]
 * (0 is white, 1 is black). 
 */
object MnistToMat {

  def loadMnistFile(pathToFile: String, maxImages: Int = Int.MaxValue): Mat = {
    val reader = new DataInputStream(new FileInputStream(pathToFile))
    val magicNumber = reader.readInt() // ignore it...
    println("magic number: " + magicNumber)
    val numImages = reader.readInt()
    val numRows = min(numImages, maxImages)
    println("number of images: " + numImages)
    val imageWidth = reader.readInt()
    println("width: " + imageWidth)
    val imageHeight = reader.readInt()
    println("height: " + imageHeight)
    val numCols = imageWidth * imageHeight
    
    val result = Mat.empty(numRows, numCols)
    for (r <- 0 until numRows) {
      val buff = new Array[Byte](numCols)
      reader.read(buff)
      for (c <- 0 until numCols) {
        result(r, c) = buff(c) / 255d
      }
    }
    
    reader.close()
    
    result
  }
  
  def rowToImage(row: Mat, height: Int, width: Int): BufferedImage = {
    row.reshape(height, width).toImage{ x =>
      val c = (255 * (1 - x)).toInt
      0xFF000000 + c * 0x00010101
    }
  }
  
  // just a little test. Adjust the path if you want to see what the data looks
  // like
  def main(args: Array[String]): Unit = {
    println("Showing example digits")
    val pathToMnistImageFile = 
      "/home/tyukiand/Projects/autoencoder/train-images-idx3-ubyte"
    val data = loadMnistFile(pathToMnistImageFile, 100)
    for (r <- 0 until data.height) {
      val row = data(r ::: (r + 1), 0 ::: end)
      val img = rowToImage(row, 28, 28)
      nz.wicker.autoencoder.visualization.show(img, "example digit")
    }
  }
}