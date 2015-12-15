package org.kramerlab.autoencoder

import org.kramerlab.autoencoder.math.matrix.Mat
import scala.io.Source
import weka.core.Instances
import weka.core.Attribute
import weka.core.DenseInstance

package object wekacompatibility {

  def readBooleanArff(fileName: String): Mat = {
    if (isSparse(fileName)) {
      readSparseBooleanArff(fileName)
    } else {
      readDenseBooleanArff(fileName)
    }
  }
  
  def isSparse(fileName: String): Boolean = {
    for (line <- Source.fromFile(fileName).getLines) {
      if (line.startsWith("{")) return true
    }
    return false
  }
  
  def readSparseBooleanArff(fileName: String): Mat = {
    var lines = List[List[Int]]()
    var width = 0
    
    for (line <- Source.fromFile(fileName).getLines) {
      if (line.startsWith("{")) {
        val lineWithoutParens = line.replace("{", "").replace("}", "")
        if (lineWithoutParens.contains(',')) {
          val lineEntries = lineWithoutParens.split(",")
          val firstEntriesExtracted = lineEntries.map{
            _.trim.takeWhile{_ != ' '}
          }
          val nonzeroIndices = firstEntriesExtracted.map{_.toInt}.toList
          lines = nonzeroIndices :: lines
        } else {
          lines = Nil :: lines
        }
      } else if (line.startsWith("@attribute")) {
        width += 1
      }
    }
    
    lines = lines.reverse
    val height = lines.size
    
    val mat = new Mat(height, width)
    
    for ((r, line) <- Stream.from(0) zip lines; c <- line) {
      mat(r, c) = 1d
    }
    
    mat
  }
  
  def readDenseBooleanArff(fileName: String): Mat = {
    var lines = List[List[Int]]()
    var width = 0
    
    for (line <- Source.fromFile(fileName).getLines) {
      if (!line.trim.isEmpty && !line.startsWith("@")) {
        val lineEntries = line.split(",")
        val nonzeroIndices = (for (
          index <- 0 until lineEntries.size 
          if (lineEntries(index).trim == "1")
        ) yield index).toList
        lines = nonzeroIndices :: lines
      } else if (line.startsWith("@attribute")) {
        width += 1
      }
    }
    
    lines = lines.reverse
    val height = lines.size
    
    val mat = new Mat(height, width)
    
    for ((r, line) <- Stream.from(0) zip lines; c <- line) {
      mat(r, c) = 1d
    }
    
    mat
  }
  
  def instancesToMat(instances: Instances): Mat = {
    val width = instances.numAttributes
    val height = instances.numInstances
    val result = Mat.createDirty(height, width, 0)
    for (r <- 0 until height) {
      val instance = instances.get(r)
      for (i <- 0 until instance.numValues()) {
        val c = instance.index(i)
        val value = instance.valueSparse(i)
        result(r, c) = value
      }
    }
    result
  }
  
  def matToInstances(mat: Mat): Instances = {
    //val nominalValues = new java.util.ArrayList[String]
    //nominalValues.add("0")
    //nominalValues.add("1")
    
    val attributes = new java.util.ArrayList[Attribute]
    for (c <- 0 until mat.width) {
      attributes.add(new Attribute("col " + c))
    }
    
    val result = new Instances("matrix", attributes, mat.height)
    for (r <- 0 until mat.height) {
      val entries = Array.ofDim[Double](mat.width)
      val inst = new DenseInstance(1, entries)
      for (c <- 0 until mat.width) {
        inst.setValue(c, mat(r, c))
      }
      result.add(inst)
    } 
    
    result
  }
  
}