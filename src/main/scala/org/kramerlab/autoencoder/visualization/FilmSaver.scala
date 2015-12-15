package org.kramerlab.autoencoder.visualization

import java.awt.Color

/**
 * Instances of this class can be used to write streams of png images to the
 * hard drive.
 */
class FilmSaver(
  val width: Int = 1024,
  val height: Int = 800,
  val backgroundColor: Color = Color.BLACK
) {
  // TODO: ok, maybe finish it later, it would be extraordinary helpful to see
  // what's happening while the simulation runs
  
  
}

/*
package edu.jgu.num.md.visualization

import edu.jgu.num.md.Particle
import java.awt.Graphics2D
import java.awt.image.BufferedImage
import edu.jgu.num.md.geometry.SimulationRegion
import javax.imageio._
import java.io.File
import scala.actors.Actor._

class FilmSaver(
  epochMask: Int,
  outputFileNamePrefix: String,
  numberOfSimulationSteps: Int,
  height: Int,
  width: Int
) extends Visualization 
  with ((Int, Int, SimulationRegion, Array[Particle]) => Unit) {

  override def apply(
    epoch: Int, 
    simulationStep: Int, 
    simulationRegion: SimulationRegion,
    particles: Array[Particle]
  ): Unit = {
    if ((epoch & epochMask) > 0 && simulationStep % numberOfSimulationSteps == 0) {
      actor {
        val image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
        paint(
          image.getGraphics.asInstanceOf[Graphics2D], 
          particles,
          simulationRegion,
          height,
          width
        )
        val outputFile = new File(
          outputFileNamePrefix + "_" + 
          simulationStep.formatted("%08d") + ".png"
        )
        if (!outputFile.exists) {
          outputFile.createNewFile()
        }
        ImageIO.write(image, "png", outputFile)
      }
    }
  }
}
*/