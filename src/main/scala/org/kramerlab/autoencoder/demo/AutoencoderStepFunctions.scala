//package org.kramerlab.autoencoder.demo
//
//import org.jfree.chart.plot.XYPlot
//import org.jfree.chart.ChartPanel
//import org.kramerlab.autoencoder.neuralnet.rbm.Rbm
//import org.jfree.chart.JFreeChart
//import org.jfree.ui.ApplicationFrame
//import org.jfree.data.xy.DefaultXYDataset
//import org.kramerlab.autoencoder.math.matrix.Mat
//import org.jfree.chart.axis.NumberAxis
//import org.jfree.chart.renderer.xy.XYSplineRenderer
//import org.kramerlab.autoencoder.neuralnet.FullBipartiteConnection
//import org.kramerlab.autoencoder.neuralnet.rbm.RbmStack
//import org.kramerlab.autoencoder.neuralnet.rbm.BernoulliUnitLayer
//import org.kramerlab.autoencoder.neuralnet.rbm.DefaultRbmTrainingConfiguration
//import org.kramerlab.autoencoder.{trainAutoencoder, Linear, Sigmoid}
//import org.kramerlab.autoencoder.math.matrix.Matrix
//
//object AutoencoderStepFunctions {
//    
//  def randomStepFunction(steps: Int): (Double => Double) = {
//    val onOff = (0 to steps).map{ x => if (math.random > 0.5) 1d else 0d }
//    (x => onOff((x * steps).floor.toInt))
//  }
//  
//  def interval(dataDim: Int) = (for (i <- 0 until dataDim) yield {
//    i.toDouble / dataDim
//  }).toArray
//  
//  def arrToRow(arr: Array[Double]) = {
//    val res = new Mat(1, arr.size, 0)
//    for (i <- 0 until arr.size) res(0, i) = arr(i)
//    res
//  }
//  
//  def rowToArr(row: Matrix) = {
//    val res = new Array[Double](row.width)
//    for (i <- 0 until row.width) {
//      res(i) = row(0, i)
//    }
//    res
//  }
//  
//  def main(args: Array[String]) {
//    
//    // generate data
//    val steps = 10
//    val visDim = 100
//    val hidDim = 50
//    val numberOfExamples = 5000
//    val data = new Mat(numberOfExamples, visDim, 0)
//    for (r <- 0 until numberOfExamples) {
//      val function = randomStepFunction(steps)
//      val rowData = interval(visDim) map function
//      for (c <- 0 until visDim) {
//        data(r, c) = rowData(c)
//      }
//    }
//    
//    // generate a whole autoencoder with, say, 3 levels
//    val autoencoder = trainAutoencoder(
//      Array(Sigmoid, Sigmoid, Sigmoid),
//      Array(visDim, 64, 16),
//      data
//    )
//    
//    // create random examples and see the reconstructions produced after
//    // compression and decompression
//    
//    for (i <- 0 until 5) {
//      val f = randomStepFunction(steps)
//      val input = arrToRow(interval(visDim) map f)
//      val compressed = autoencoder.compress(input)
//      val output = autoencoder.decompress(compressed)
//      val xValues = interval(visDim)
//      val pointset = new DefaultXYDataset()
//      pointset.addSeries("input", Array(xValues, rowToArr(input)))
//      pointset.addSeries("output", Array(xValues, rowToArr(output)))
//      val splineRenderer = new XYSplineRenderer()
//      val xAxis = new NumberAxis("foo")
//      val yAxis = new NumberAxis("bar")
//      val plot = new XYPlot(pointset, xAxis, yAxis, splineRenderer)
//      val chart = new JFreeChart(plot)
//      val frame = new ApplicationFrame("Example...")
//      val chartPanel = new ChartPanel(chart)
//      frame.setContentPane(chartPanel)
//      frame.pack()
//      frame.setVisible(true)
//    } 
//  }
//}