package org.kramerlab.autoencoder.math.optimization

import org.jfree.chart.plot.XYPlot
import org.jfree.chart.ChartPanel
import org.jfree.chart.JFreeChart
import org.jfree.ui.ApplicationFrame
import org.jfree.data.xy.DefaultXYDataset
import org.kramerlab.autoencoder.math.matrix.Mat
import org.kramerlab.autoencoder.math.optimization.{
  DifferentiableFunction => DF, 
  _
}
import org.jfree.chart.axis.NumberAxis
import scala.collection.mutable.ListBuffer
import org.jfree.chart.renderer.xy.XYSplineRenderer
import org.kramerlab.autoencoder.math.matrix.Mat
import scala.math._

/**
 * Comparison of various re-implementations of conjugate gradient descent and
 * a bunch of other, much simpler, optimization algorithms
 */

object CgComparison {
  type Vec = Array[Double]
  
  abstract class DifferentiableFunction extends (Vec => (Double, Vec)) {
    val history = new ListBuffer[(Vec, Double)]
    def valueAndGrad(x: Vec): (Double, Vec)
    
    override def apply(x: Vec) = {
      val result = valueAndGrad(x)
      history += {(x, result._1)}
      result
    }
  }   
   
  /**
   * Example 20.2 from Hanke-Bourgeois's book
   */
  class BananaFunction(val a: Double = 100, val b: Double = 1) 
    extends DifferentiableFunction {
    override def valueAndGrad(v: Vec) = {
      val x = v(0)
      val y = v(1)
      val distFromParabola = y - x * x
      val distFromMin = x - b
      val value = 
        a * a * distFromParabola * distFromParabola + distFromMin * distFromMin
        
      val grad = Array(
        -4 * a * a * distFromParabola * x + 2 * distFromMin,
        2 * a * a * distFromParabola
      )
      
      (value, grad)
    }
  }
  
  def add(a: Array[Double], b: Array[Double]) = 
    (a zip b).map{case (x, y) => x + y}
  
  def mul(a: Array[Double], d: Double) = a.map{_ * d}
  def normSq(a: Array[Double]) = a.map{x => x * x}.sum
  
  def main(args: Array[String]): Unit = {
    val startValue = Array(-0.5, -0.4)

    val pointset = new DefaultXYDataset
    
    // add series to dataset
    def addSeries(name: String, arr: Array[Double]): Unit = {
      val maxPoints = 500
      val truncated = arr.take(maxPoints).map{x => log10(x)}
      pointset.addSeries(
        name, 
        Array(
          (0 until truncated.size).map{_.toDouble}.toArray, 
          truncated
        )
      )
    }
    
    // stupid grid search
    {
      val banana = new BananaFunction 
      val grid = (-200 to 200).map{_ / 100d}
      var argmin = Array(0d ,0d)
      var min = Double.PositiveInfinity
      for (x <- grid; y <- grid) {
         val value =  banana(Array(x, y))._1
         if (value < min) {
           min = value
           argmin = Array(x, y)
         }
      }
      println("Grid search result: f(" + 
        argmin.mkString("[", ",", "]") + 
        ")= " + min
      )
    }
    
    // basic gradient descent (Hanke p. 180)
    {
      val banana = new BananaFunction
      val maxIters = 51000
      var currentPosition = startValue
      var currentValue = banana(startValue)._1
      var currentDirection = mul(banana(startValue)._2, -1d)
      val mu = 0.5
      var i = 0
      while (i < maxIters) {
        i += 1
        var alpha = 1d
        var nextPosition = add(currentPosition, mul(currentDirection, alpha)) 
        var nextValue = banana(nextPosition)._1
        while (
          nextValue > currentValue - mu * alpha * normSq(currentDirection)
        ) {
          alpha /= 2
          nextPosition = add(currentPosition, mul(currentDirection, alpha))
          nextValue = banana(nextPosition)._1
        }
        currentPosition = nextPosition
        currentValue = nextValue
        currentDirection = mul(banana(currentPosition)._2, -1)
      }
      
      println("Basic Gradient Descent: f(" + 
        currentPosition.mkString("[", ",", "]") + 
        ") = " + currentValue
      )
      
      addSeries("Gradient descent", banana.history.grouped(100).map(_.last._2).toArray)
      
    }
    
    // My implementation of Hager/Zhang's version of CG
    {
      val cg = 
        new ConjugateGradientDescent_HagerZhang(new ConjugateGradientDescent_HagerZhangConfiguration)
      
      val b = new BananaFunction
      val f = new DF[Mat] {
        override def apply(x: Mat): Double = {
          b(Array(x(0, 0), x(1, 0)))._1
        }
        override def grad(x: Mat): Mat = {
          val gradAsArray = b(Array(x(0, 0), x(1, 0)))._2
          Mat(2, 1)(gradAsArray(0), gradAsArray(1))
        }
      }
      
      val x = cg.minimize(f, Mat(2, 1)(startValue(0), startValue(1)))
      
      println("My CG result: f(" + x(0, 0) + ", " + x(1, 0) + ")= " + f(x))
      
      addSeries("MyCg", b.history.map(_._2).toArray)
    }
    
    // nice re-implementation of minimize.m that doesn't seem to work for some
    // reason?
    {
      val cg = 
        new NonlinearConjugateGradientDescent_Rasmussen
      
      // let's just play with interpolation-extrapolation a little bit
      val zeroOfStandardParabola = cg.interpolate((-1,1,-2), (1,1,2), 100)
      println("The zero of the standard parabola: " + zeroOfStandardParabola)
      
      val oneDivSqrt3 = cg.interpolate((0,0,-1), (2, 6, 11), 100)
      println("Something about " + sqrt(1d / 3) + " : " + oneDivSqrt3)
      
      val b = new BananaFunction
      val f = new DF[Mat] {
        override def apply(x: Mat): Double = {
          b(Array(x(0, 0), x(1, 0)))._1
        }
        override def grad(x: Mat): Mat = {
          val gradAsArray = b(Array(x(0, 0), x(1, 0)))._2
          Mat(2, 1)(gradAsArray(0), gradAsArray(1))
        }
      }
      
      val x = cg.minimize(f, Mat(2, 1)(startValue(0), startValue(1)))
      
      println("2nd Rasmussen's reimplementation result: f(" + x(0, 0) + ", " + 
          x(1, 0) + ")= " + f(x))
      
      addSeries("2nd Rasmussen", b.history.map(_._2).toArray)
    }
    
    {
      val cg = 
        new CG_Rasmussen2(maxIters = 500)
      
      val b = new BananaFunction
      val f = new DF[Mat] {
        override def apply(x: Mat): Double = {
          b(Array(x(0, 0), x(1, 0)))._1
        }
        override def grad(x: Mat): Mat = {
          val gradAsArray = b(Array(x(0, 0), x(1, 0)))._2
          Mat(2, 1)(gradAsArray(0), gradAsArray(1))
        }
      }
      
      val x = cg.minimize(f, Mat(2, 1)(startValue(0), startValue(1)), Nil)
      
      println("CG_Rasmussen2 : f(" + x(0, 0) + ", " + 
          x(1, 0) + ")= " + f(x))
      
      addSeries("2nd Rasmussen", b.history.map(_._2).toArray)
    }
    
    val splineRenderer = new XYSplineRenderer()
    val xAxis = new NumberAxis("steps")
    val yAxis = new NumberAxis("log_10(error)")
    val plot = new XYPlot(pointset, xAxis, yAxis, splineRenderer)
    val chart = new JFreeChart(plot)
    val frame = new ApplicationFrame("CG Performance comparison")
    val chartPanel = new ChartPanel(chart)
    frame.setContentPane(chartPanel)
    frame.pack()
    frame.setVisible(true)
  }
}