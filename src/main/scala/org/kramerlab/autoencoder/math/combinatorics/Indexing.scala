package org.kramerlab.autoencoder.math.combinatorics

import scala.collection.immutable.Stream.consWrapper

object Indexing {
  
  /**
   * Constructs a sequence that looks like this for [0, 1]:
   * 0.5
   * 0.25 0.75
   * 0.125 0.357 0.625 0.875 ...
   *
   */
  def intervalRefinement(start: Double, end: Double): Stream[Double] = {
    val mid = (start + end) / 2
    mid #:: interleaveStreams(
      intervalRefinement(start, mid),
      intervalRefinement(mid, end)
    )
  }
  
  def interleaveStreams[X](a: Stream[X], b: Stream[X]): Stream[X] = {
    a.head #:: b.head #:: interleaveStreams(a.tail, b.tail)
  }
  
  /**
   * Permutates the range in a "bisecting" way, which allows to get better 
   * and better approximations of a function, without evaluating too many
   * values.
   */
  def rangeRefinement(start: Int, end: Int): List[Int] = {
    def interleaveLists[X](a: List[X], b: List[X]): List[X] = {
      (a, b) match {
        case (Nil, _) => b
        case (_, Nil) => a
        case (x :: xt, y :: yt) => x :: y :: interleaveLists(xt, yt)
      }
    }
    if (end - start == 1) {
      List(start)
    } else if ((end - start) == 2) {
      List(start, start + 1)
    } else {
      val mid = (start + end) / 2
      mid :: interleaveLists(
        rangeRefinement(start, mid), 
        rangeRefinement(mid + 1, end)
      )
    }
  }
  
  /**
   * goes through cartesian product of two potentially infinite sequences in
   * the following way:
   * 
   *  1  2  5 10
   *  3  4  6 11
   *  7  8  9 12
   * 13 14 15 16
   * 
   * etc.
   */
  def cartesianProduct[X, Y](x: Stream[X], y: Stream[Y]): Stream[(X, Y)] = {
    interleaveStreams(
      triangle(x, y), 
      triangle(y.tail, x).map{case (a, b) => (b, a)}
    ) 
  }
  
  case class CartesianProductOps[X](a: Stream[X]) {
    def x[Y](b: Stream[Y]) = cartesianProduct(a, b);
  }
  
  implicit def toCartesianProductOps[X](a: Stream[X]) = 
    CartesianProductOps(a) 
  
  def triangle[X, Y](x: Stream[X], y: Stream[Y]): Stream[(X, Y)] = {
    def rec(xs: Stream[X], usedYs: List[Y], ys: Stream[Y]): Stream[(X, Y)] = {
      if (xs.isEmpty) {
        Stream.empty[(X, Y)]
      } else {
        val xHead = xs.head
        if(ys.isEmpty) {
          usedYs.map{y => (xHead, y)} ++: rec(xs.tail, usedYs, ys)
        } else {
          val nextUsedYs = ys.head :: usedYs
          nextUsedYs.map{y => (xHead, y)}.toStream #::: 
          rec(xs.tail, nextUsedYs, ys.tail)
        }
      }
    } 
    rec(x, Nil, y)
  }
}