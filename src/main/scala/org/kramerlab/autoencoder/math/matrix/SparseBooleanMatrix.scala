//package org.kramerlab.autoencoder.math.matrix
//
//import scala.collection.mutable.ListBuffer
//
///**
// * Instances of this class represent a special subset of real valued
// * matrices that have only 0 and 1 as entries, and are sparse.
// */
//class SparseBooleanMatrix(height: Int, width: Int) 
//  extends Matrix(height, width, 0d) {
//
//  val entries = Array.fill(height){new ListBuffer[Int]()}
//  
//  override def update(r: Int, c: Int, e: Double): Unit = {
//    if (e == 0) {
//      entries(r) -= c
//    } else if (e == 1) {
//      val row = entries(r)
//      if (!row.contains(c)) {
//        row += c
//      } 
//    } else {
//      throw new IllegalArgumentException(
//        "Cannot insert enties unequal to 0 or 1 in sparse boolean matrix (" +
//        "tried to insert " + e + ")"
//      )
//    }
//  }
//  
//  override def apply(r: Int, c: Int): Double = {
//    if (entries(r).contains(c)) 1d else 0d
//  }
//
//  override def apply(rows: RangeSelector, cols: RangeSelector): SparseBooleanMatrix = {
//    throw new UnsupportedOperationException("this class is a wrack, just don't use it")
//  }
//  
//  private def parallel_colCached_*(that: Matrix): Matrix = {
//    Matrix.requireMultipliableDimensions(this, that)
//    val w = that.width      // dimensions of result matrix
//    val h = this.height     
//    val compat = this.width // the dimension that decides about compatibility
//    that match {
//      case mat: Mat => {
//        val result = new Mat(h, w)
//        val column = new Array[Double](compat)   
//        var c = 0
//        while (c < w) {
//          for (r <- (0 until compat).par) {
//            column(r) = that(r, c)
//          }
//          // synchronize after copying the column...
//          for (r <- (0 until compat).par) {
//            var sum = 0d
//            for (i <- entries(r)) 
//              sum += column(i)
//            result(r, c) = sum
//          }
//        }
//        result
//      }
//      case _ => throw new UnsupportedOperationException(
//        "Cannot handle multiplication of sparse boolean matrices with " +
//        "anything except dense matrices for now, sorry."
//      )
//    }
//  }
//  
//  override def *(other: Matrix) = parallel_colCached_*(other)
//  
//  override def shuffleRows(): Unit = throw new UnsupportedOperationException
//  
//  override def isNaN = false
//  override def isInfinite = false
//  override def isInvalid = false
//}