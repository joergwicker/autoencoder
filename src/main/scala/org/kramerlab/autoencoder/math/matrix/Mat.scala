package org.kramerlab.autoencoder.math.matrix

import scala.math._
import org.kramerlab.autoencoder.math.structure.VectorSpace
import org.kramerlab.autoencoder.visualization.Visualizable

/**
 * Implementation of dense matrices in row major format with double values.
 * 
 * This is a minimalistic implementation of a dense double matrix. The main goal
 * is to write an implementation that is too simple to break and too simple too
 * introduce some sneaky autoboxing issues somewhere. It has no complex 
 * dependencies on any libraries, and is supposed to be used for testing and
 * performance measurement of more advanced implementations. 
 * 
 * Notice: Although it's stupid and ugly, it still beats the crap out of 
 * single-threaded java libraries like EJML... :/ 
 * 
 * Replace it by CUDA or something more performant later.
 */

class Mat protected (
  val height: Int,
  val width: Int,
  val defaultValue: Double,
  val entries: Array[Array[Double]]
) extends VectorSpace[Mat] with Visualizable with Serializable {
  
  def this(height: Int, width: Int, defaultValue: Double) = 
    this(height, width, defaultValue, Array.ofDim[Double](height, width))

  def this(height: Int, width: Int) =
    this(height, width, 0d, Array.ofDim[Double](height, width))

  def this() = this(0, 0)
    
  def apply(r: Int, c: Int): Double = entries(r)(c)
  def apply(i: Int): Double = {
    if (height == 1) {
      this(0, i)
    } else if (width == 1) {
      this(i, 0)
    } else {
      throw new IllegalArgumentException(
        "Tried to access entry of a matrix with single index = " + i + ", but" + 
        " matrix has size " + height + " x " + width + " and is neither " +
        "a column, nor a row vector."
      )
    }
  }
  def update(r: Int, c: Int, d: Double) { entries(r)(c) = d }

  def toArray: Array[Array[Double]] = entries
  
  def apply(rows: RangeSelector, cols: RangeSelector): Mat = {
    val (startRow, endRow) = rows(height)
    val (startCol, endCol) = cols(width)
    val h = endRow - startRow
    val w = endCol - startCol
    val res = Mat.createDirty(h, w, defaultValue)
    var r = startRow
    while (r < endRow) {
      var c = startCol
      val resRow = res.entries(r - startRow)
      val thisRow = this.entries(r)
      while (c < endCol) {
        resRow(c - startCol) = thisRow(c) 
        c += 1
      }
      r += 1
    }
    res
  }
  
  def update(
    rows: RangeSelector, 
    cols: RangeSelector, 
    other: Mat
  ): Unit = {
    val (startRow, endRow) = rows(height)
    val (startCol, endCol) = cols(width)
    val h = endRow - startRow
    val w = endCol - startCol
    var r = 0
    var c = 0
    while (r < h) {
      c = 0
      while (c < w) {
        entries(startRow + r)(startCol + c) = other.entries(r)(c)
        c += 1
      }
      r += 1
    }
  }
  
  /**
   * Maps entries of the matrix with the specified function `f`.
   */
  def map(f: Double => Double): Mat = map_parallel(f)
  
  def map_naive(f: Double => Double): Mat = {
    val result = new Mat(height, width, f(defaultValue))
	var r = 0
	while (r < height) {
	  val row = entries(r)
	  val resultRow = result.entries(r)
	  var c = 0
	  while (c < width) {
		resultRow(c) = f(row(c))
	    c += 1
	  }
	  r += 1
	}
	result
  }

  def map_parallel(f: Double => Double): Mat = {
    val result = new Mat(height, width, f(defaultValue))
    val resEntries = result.entries
    
    val nThreads = scala.collection.parallel.defaultTaskSupport.parallelismLevel
    
    if (height > width) {
      val stripeSize = (height + nThreads - 1) / nThreads
      for (stripe <- (0 until nThreads).par) {
        var r = stripeSize * stripe
        val maxR = min(height, stripeSize * (stripe + 1))
        while (r < maxR) {
          val resRow = resEntries(r)
          val thisRow = entries(r)
          var c = 0
          while (c < width) {
            resRow(c) = f(thisRow(c))
            c += 1
          }
          r += 1
        }
      }
    } else {
      val stripeSize = (width + nThreads - 1) / nThreads
      for (stripe <- (0 until nThreads).par) {
        val startC = stripe * stripeSize
        val endC = min(stripeSize * (stripe + 1), width)
        var r = 0
        while (r < height) {
          val resRow = resEntries(r)
          val thisRow = entries(r)
          var c = startC
          while (c < endC) {
            resRow(c) = f(thisRow(c))
            c += 1
          }
          r += 1
        }
      }
    }
    
    result
  }
  
   def mapWithIndex(f: (Int, Int, Double) => Double): Mat = {
    val result = Mat.createDirty(height, width, f(0,0,defaultValue))
	var r = 0
	while (r < height) {
	  val row = entries(r)
	  val resultRow = result.entries(r)
	  var c = 0
	  while (c < width) {
		resultRow(c) = f(r,c,row(c))
	    c += 1
	  }
	  r += 1
	}
	result
  }

  /**
   * Removes all values from the matrix that do not fulfill the predicate `p`.
   * The removed values are replaced by the default value.
   */
   def filter(p: Double => Boolean): Mat = {
    val result = new Mat(height, width, defaultValue)
	var r = 0
	while (r < height) {
	  val row = entries(r)
	  val resultRow = result.entries(r)
	  var c = 0
	  while (c < width) {
	    val e = row(c)
		if (p(e)) resultRow(c) = e
		c += 1
	  }
	  r += 1
	}
	result
  }


  /**
   * Inverses the control over the process of iteration through
   * the entries of the matrix. 
   * 
   * It would be interesting to compare which impact such an indirection has
   * on performance: if would eliminate tons of boilerplate code. Notice 
   * that this iteration strategy is not applied in the most potentially 
   * time critical methods (map, filter, ...) of this class, although it
   * obviously would make the code much more concise. We do not use it,
   * because we really want to measure the performance of the 
   * implementations
   * that are as low level as possible. However, this method is used when
   * the performance is irrelevant, as in `toString` for example.
   */
   def foreachWithIndex[U](f: (Int, Int, Double) => U) {
    var r = 0
	while (r < height) {
	  val row = entries(r)
	  var c = 0
	  while (c < width) {
	    f(r, c, row(c))
		c += 1
	  }
	  r += 1
	}
  }

  /**
   * Executes the specified procedure `f` for each entry of this matrix
   */
   def foreach[U](f: Double => U) {
    var r = 0
	while (r < height) {
	  val row = entries(r)
	  var c = 0
	  while (c < width) {
	    f(row(c))
		c += 1
	  }
	  r += 1
	}
  }

   def exists(p: Double => Boolean): Boolean = {
    var r = 0
	while (r < height) {
	  val row = entries(r)
	  var c = 0
      while (c < width) {
        if (p(row(c)))  {
          return true
		}
		c += 1
	  }
	  r += 1
	}
	false
  }

   def existsWithIndex(p: (Int, Int, Double) => Boolean): Boolean = {
    var r = 0
	while (r < height) {
	  val row = entries(r)
	  var c = 0
      while (c < width) {
        if (p(r, c, row(c)))  {
          return true
		}
		c += 1
	  }
	  r += 1
	}
	false
  }

   def forall(p: Double => Boolean): Boolean = {
    var r = 0
	while (r < height) {
	  val row = entries(r)
	  var c = 0
	  while (c < width) {
	    if (!p(row(c))) {
	      return false
		}
		c += 1
	  }
	  r += 1
	}
	true
  }

   def forallWithIndex(p: (Int, Int, Double) => Boolean): Boolean = {
    var r = 0
	while (r < height) {
	  val row = entries(r)
	  var c = 0
      while (c < width) {
        if (!p(r, c, row(c)))  {
          return false
		}
		c += 1
	  }
	  r += 1
	}
	true
  }

  /**
   * Transposes the matrix. 
   * This method creates a completely new matrix, all values are copied
   */
   def transpose: Mat = {
    val result = new Mat(width, height, defaultValue)
    var r = 0
	while (r < height) {
	  val row = entries(r)
	  var c = 0
	  while (c < width) {
	    result.entries(c)(r) = row(c)
		c += 1
	  }
	  r += 1
	}
    result
  }
  
  def size = (height, width)
  
  /**
   * Reshapes the matrix row-wise
   */
   def reshape(newHeight: Int, newWidth: Int): Mat = {
    require(newHeight * newWidth == height * width, 
	  "Cannot reshape " + size + "-matrix to " + 
	  (newHeight, newWidth) + " matrix.")
    val result = new Mat(newHeight, newWidth, defaultValue)
	var r = 0
	var newR = 0
	var newC = 0
	while (r < height) {
	  val row = entries(r)
	  var c = 0
	  while (c < width) {
	    result.entries(newR)(newC) = row(c)
		c += 1
		newC += 1
		if (newC == newWidth) {
		  newC = 0
		  newR += 1
		}
	  }
	  r += 1
	}
	result
  }

  def shuffleRows(): Unit = permutateRows(
    org.kramerlab.autoencoder.math.random.permutation(height)
  )
  
   def permutateRows(permutation: Array[Int]): Unit = {
    val alreadyMoved = Array.fill(height){ false }
    var r = 0
    while (r < height) {
      if (!alreadyMoved(r)) {
        // move everything in this cycle
        var floating = entries(r) // pick the element at r-th position
        var next = permutation(r) // see where it goes
        do {
          val temp = entries(next)
          entries(next) = floating
          floating = temp
          alreadyMoved(next) = true
          next = permutation(next)
        } while (next != r)
        entries(next) = floating
      }
      r += 1
    }
  }
  
   def sameEntries(other: Mat): Boolean = {
    (size == other.size) && {
      forallWithIndex {
        (r, c, e) => 
		other(r, c) == e
	  }
	}
  }

  /**
   * Creates a multiline string representation of this matrix.
   * All entries are indented to the right, the exponentials of
   * the entries are left independent (no global rescaling occurs).
   * It doesn't look very nice, but it's good enough for debugging.
   */
  override def toString = {
    val unindented = Array.ofDim[String](height, width)
	val columnWidths = Array.fill[Int](width){1}
    foreachWithIndex { (r, c, e) =>
	  val str = e.toString
	  unindented(r)(c) = str
	  columnWidths(c) = max(columnWidths(c), str.size)
	}
	val pseudoline = Array.fill[String](width){""}
	val withPseudolines = (pseudoline +: unindented) :+ pseudoline
	  
	val indented = for (row <- withPseudolines) yield {
	  for ((str, i) <- row.zipWithIndex) yield {
	    " " * (columnWidths(i) - str.size) + str
	  }	
	}
	val lines = indented.map { arr => arr.mkString("| ", " ", " |") }
	lines(0) = "\n _" + lines(0).substring(2, lines(0).size - 2) + "_ "
	val n = lines.size - 1
	lines(n) = " -" + lines(n).substring(2, lines(n).size - 2) + "- \n"
    
	lines.mkString("\n")
  }

  def foldRows(start: Double)(f: (Double, Double) => Double): Mat = {
    val result = Mat.fill(1, width, defaultValue){(i,j) => start}
    var r = 0
    while (r < height) {
      var c = 0
      while (c < width) {
        result(0, c) = f(result(c), this(r, c))
        c += 1
      } 
      r += 1
    }
    result
  }
  
  /*-------------------------Algebraic structure-----------------------------*/
  
  private def multiplyInto(d: Double, target: Mat): Mat = {
	var r = 0
	while (r < height) {
	  var c = 0
	  val row = entries(r)
	  val targetRow = target.entries(r)
	  while (c < width) {
	    targetRow(c) = row(c) * d
		c+= 1
	  }
	  r += 1
	}
	target
  }

  def *=(d: Double): Unit = {
    multiplyInto(d, this)
  }

   def *(d: Double): Mat = {
    multiplyInto(d, Mat.createDirty(height, width, defaultValue))
  }

  private def divideInto(d: Double, target: Mat): Mat = {
    var r = 0
	while (r < height) {
	  var c = 0
	  val row = entries(r)
	  val targetRow = target.entries(r)
	  while (c < width) {
	    targetRow(c) = row(c) / d
		c+= 1
	  }
	  r += 1
	}
	target
  }

  def /=(d: Double): Unit = divideInto(d, this)
  
  override def /(d: Double): Mat = {
    divideInto(d, Mat.createDirty(height, width, defaultValue))
  }

  private def addInto(other: Mat, target: Mat): Mat = {
    Mat.requireSameDimensions(this, other, "addition")
	Mat.requireSameDimensions(this, target, "addition")
    var r = 0
	while (r < height) {
	  var c = 0
	  val row = entries(r)
	  val targetRow = target.entries(r)
	  val otherRow = other.entries(r)
	  while (c < width) {
	    targetRow(c) = row(c) + otherRow(c)
		c+= 1
	  }
	  r += 1
	}   
	target
  }

  def +(other: Mat): Mat = {  
    addInto(other, Mat.createDirty(height, width, defaultValue))
  } 

   def +=(other: Mat): Unit = addInto(other, this)

  private def subtractInto(other: Mat, target: Mat): Mat = {
    Mat.requireSameDimensions(this, other, "addition")
	Mat.requireSameDimensions(this, target, "addition")
    var r = 0
	while (r < height) {
	  var c = 0
	  val row = entries(r)
	  val targetRow = target.entries(r)
	  while (c < width) {
	    targetRow(c) = row(c) - other(r, c)
		c+= 1
	  }
	  r += 1
	}   
	target
  }

  override def -(other: Mat): Mat = 
    subtractInto(other, Mat.createDirty(height, width, defaultValue))

  def -=(other: Mat): Unit = subtractInto(other, this)

  /** 
   * Replaces all entries by their additive inverses
   */
  def negate() {
    var r = 0
	while (r < height) {
	  var c = 0
	  val row = entries(c)
	  while (c < width) {
	    row(c) = -row(c)
	    c += 1
	  }
	  r += 1
	}
  }

  private def pointwiseMultiplyInto(other: Mat, target: Mat): Mat = {
    Mat.requireSameDimensions(this, other, "pointwise multiplication")
	Mat.requireSameDimensions(this, target, "pointwise multiplication")
    var r = 0
	while (r < height) {
	  var c = 0
	  val row = entries(r)
	  val targetRow = target.entries(r)
	  while (c < width) {
	    targetRow(c) = row(c) * other(r, c)
		c+= 1
	  }
	  r += 1
	}   
	target
  }

  /**
   * Multiplies this matrix with other matrix pointwise
   */
   def :*(other: Mat): Mat = 
    pointwiseMultiplyInto(other, Mat.createDirty(height, width, defaultValue))

  private def pointwiseDivideInto(other: Mat, target: Mat): Mat = {
    Mat.requireSameDimensions(this, other, "pointwise division")
    Mat.requireSameDimensions(this, target, "pointwise division")
    var r = 0
    while (r < height) {
      var c = 0
      val row = entries(r)
      val targetRow = target.entries(r)
      while (c < width) {
        targetRow(c) = row(c) / other(r, c)
        c+= 1
      }
      r += 1
    }   
    target
  }
    
   def :/(other: Mat): Mat = 
    pointwiseDivideInto(other, Mat.createDirty(height, width, defaultValue))
    
  /**
   * Multiplies this matrix with other matrix pointwise and writes the 
   * result into this matrix
   */
  def :*=(other: Mat): Mat = pointwiseMultiplyInto(other, this)

  /**
   * Unary prefix minus
   */
   def unary_- : Mat = this * -1d
  
  /**
   * Naive implementation of the dense matrix multiplication
   */
  def naive_*(other: Mat): Mat = {
    Mat.requireMultipliableDimensions(this, other)
	val result = Mat.createDirty(this.height, other.width, 0d)
	var r = 0
	while (r < height) {
	  val row = entries(r)
	  val resultRow = result.entries(r)
	  var c = 0
	  while (c < other.width) {
	    var k = 0
		while (k < width) {
		  resultRow(c) += row(k) * other.entries(k)(c)
		  k += 1
		}
		c += 1
	  }
	  r += 1
	}
	result
  }

  /**
   * Tiled matrix multiply with tile size = 16
   */
  def tiled_*(other: Mat): Mat = {
    
    Mat.requireMultipliableDimensions(this, other)
    val TileSize = 52
    val gridHeight = (this.height + TileSize - 1) / TileSize
    val gridWidth = (other.width + TileSize - 1) / TileSize
    val gridCompat = (this.width + TileSize - 1) / TileSize
    val result = Mat.empty(this.height, other.width, 0d)

    val tile = Array.ofDim[Double](TileSize, TileSize)
    val temp_1 = Array.ofDim[Double](TileSize, TileSize)
    val temp_2 = Array.ofDim[Double](TileSize, TileSize)
    
    var gridRow = 0
    while (gridRow < gridHeight) {
      val blockHeight = min(TileSize, height - gridRow * TileSize)
      var gridCol = 0
      while (gridCol < gridWidth) {
        val blockWidth = min(TileSize, other.width - gridCol * TileSize)
        // clear the scratchpad
        var r = 0
        while (r < blockHeight) {
          var c = 0
          while (c < blockWidth) {
            tile(r)(c) = 0
            c += 1
          }
          r += 1
        }
        
        var gridK = 0
        while (gridK < gridCompat) {
          
          // the multiplied blocks have dimensions 
          // blockHeight x blockCompat
          // blockCompat x blockHeight
          val blockCompat = min(TileSize, width - gridK * TileSize)
          
          // upper left corners of relevant blocks
          val startRow_1 = gridRow * TileSize 
          val startCol_1 = gridK * TileSize
          val startRow_2 = gridK * TileSize
          val startCol_2 = gridCol * TileSize
          
          // load the submatrices
          var r = 0
          while (r < blockHeight) {
            var c = 0
            while (c < blockCompat) {
              temp_1(r)(c) = this(startRow_1 + r, startCol_1 + c)
              c += 1
            }
            r += 1
          }
          
          r = 0
          while (r < blockCompat) {
            var c = 0
            while (c < blockWidth) {
              temp_2(r)(c) = other(startRow_2 + r, startCol_2 + c)
              c += 1
            }
            r += 1
          }
          
          // perform naive matrix-multiply on submatrix, add results into
          // the scratchpad-tile
          // TODO: calculate the global row and column in some proper way
          r = 0
          while (r < blockHeight) {
            var c = 0
            while (c < blockWidth) {
              var sum = 0d
              var k = 0
              while (k < blockCompat) {
                sum += temp_1(r)(k) * temp_2(k)(c)
                k += 1
              }
              tile(r)(c) += sum
              c += 1
            }
            r += 1
          }
          
          // copy the scratchpad content to the result matrix
          r = 0
          while (r < blockHeight) {
            var c = 0
            while (c < blockWidth) {
              result(startRow_1 + r, startCol_2 + c) = tile(r)(c)
              c += 1
            }
            r += 1
          } 
          
          gridK += 1
        }
        gridCol += 1
      }
      gridRow += 1
    }
    result
  }

  
  private def threadLocalScratchpad(size: Int) = 
    new ThreadLocal[Array[Array[Double]]] {
       override protected def initialValue = 
        Array.ofDim[Double](size, size) 
    }
  
  /**
   * Tiled matrix multiply
   */
  def tiled_parallel_*(other: Mat): Mat = {
    
    Mat.requireMultipliableDimensions(this, other)
    val TileSize = 52
    val gridHeight = (this.height + TileSize - 1) / TileSize
    val gridWidth = (other.width + TileSize - 1) / TileSize
    val gridCompat = (this.width + TileSize - 1) / TileSize
    val result = Mat.empty(this.height, other.width, 0d)
    
    val thisEntries = this.entries
    val otherEntries = other.entries
    val resultEntries = result.entries
    
    val grid = for (i <- 0 until gridHeight; j <- 0 until gridWidth) yield {
      (i, j)
    }
    
    val threadLocalTile = threadLocalScratchpad(TileSize)
    val threadLocalTemp_1 = threadLocalScratchpad(TileSize)
    val threadLocalTemp_2 = threadLocalScratchpad(TileSize)
    val cleanRow = new Array[Double](TileSize)
    
    for ((gridRow, gridCol) <- grid.par) {
      val blockHeight = min(TileSize, height - gridRow * TileSize)
      val tile = threadLocalTile.get
      var r = 0
      while (r < TileSize) {
        System.arraycopy(cleanRow, 0, tile(r), 0, TileSize)
        r += 1
      }
      // val tile = Array.ofDim[Double](TileSize, TileSize)
      val temp_1 = threadLocalTemp_1.get 
      val temp_2 = threadLocalTemp_2.get
      
      val blockWidth = min(TileSize, other.width - gridCol * TileSize)
      
      var gridK = 0
      while (gridK < gridCompat) {
        
        // the multiplied blocks have dimensions 
        // blockHeight x blockCompat
        // blockCompat x blockHeight
        val blockCompat = min(TileSize, width - gridK * TileSize)
        
        // upper left corners of relevant blocks
        val startRow_1 = gridRow * TileSize 
        val startCol_1 = gridK * TileSize
        val startRow_2 = gridK * TileSize
        val startCol_2 = gridCol * TileSize
        
        // load the submatrices
        var r = 0
        while (r < blockHeight) {
          val temp_1row = temp_1(r)
          val thisRow = thisEntries(startRow_1 + r)
          var c = 0
          while (c < blockCompat) {
            temp_1row(c) = thisRow(startCol_1 + c)
            c += 1
          }
          r += 1
        }
        
        r = 0
        while (r < blockCompat) {
          val temp_2row = temp_2(r)
          val otherRow = otherEntries(startRow_2 + r)
          var c = 0
          while (c < blockWidth) {
            temp_2row(c) = otherRow(startCol_2 + c)
            c += 1
          }
          r += 1
        }
        
        // perform naive matrix-multiply on submatrix, add results into
        // the scratchpad-tile
        r = 0
        while (r < blockHeight) {
          val temp_1row = temp_1(r)
          var c = 0
          while (c < blockWidth) {
            var sum = 0d
            var k = 0
            while (k < blockCompat) {
              sum += temp_1row(k) * temp_2(k)(c)
              k += 1
            }
            tile(r)(c) += sum
            c += 1
          }
          r += 1
        }
        
        // copy the scratchpad content to the result matrix
        r = 0
        while (r < blockHeight) {
          val resRow = resultEntries(startRow_1 + r)
          var c = 0
          while (c < blockWidth) {
            resRow(startCol_2 + c) = tile(r)(c)
            c += 1
          }
          r += 1
        } 
        
        gridK += 1
      }
    }
    result
  }
  
  /**
   * matrix multiplication
   */
  def *(other: Mat): Mat = this tiled_parallel_* other
  
  /**
   * Creates new matrix of same dimension of this one, 
   * filled with zeros, and with a zero as default value
   */
   def zero = Mat.empty(height, width, 0d)

  /**
   * Returns a new matrix that looks like 
   * multiplication of constant-1-row-vector from 
   * the left
   */
   def sumRows: Mat = {
    val resEntries = new Array[Double](width)
	var r = 0
	while (r < height) {
	  var c = 0
	  val row = entries(r)
	  while(c < width) {
	    resEntries(c) += row(c)
		c += 1
	  }
	  r += 1
	}
	new Mat(1, width, defaultValue, Array(resEntries))
  }

  /**
   * Calculates the squared Frobenius norm of the matrix
   */
   def l2NormSq: Double = {
    var res = 0d
	var r = 0
	while (r < height) {
	  var c = 0
	  val row = entries(r)
	  while (c < width) {
	    val e = row(c)
		res += e * e
		c += 1
      }
	  r += 1
	}
	res
  }

  def l2Norm = math.sqrt(l2NormSq)
  
  /**
   * Scalar product: sum of the results of the 
   * pointwise multiplication
   */
   def dot(other: Mat): Double = {
    var res = 0d
	var r = 0
	while (r < height) {
	  var c = 0
	  val row = entries(r)
	  while (c < width) {
	    val e = row(c)
		val otherE = other(r, c)
		res += e * otherE
		c += 1
      }
	  r += 1
	}
	res
  }

  def luDecomposition: (Mat, Array[Int]) = {
    require(this.width == this.height) 
    val result = this.clone
    val n = this.width // number of iterations required
    val permutation = (0 until n).toArray
    
    var c = 0 // index of the currently eliminated column
    while (c < n) {
      // search the pivot element 
      var pivotIndex = -1
      var bestPivotAbsToRowNormRatio = 0d
      var potentialPivotIndex = c
      while (potentialPivotIndex < n) {
        var rowL1Norm = 0d
        var cc = c
        while (cc < n) {
          rowL1Norm += abs(result(potentialPivotIndex, cc))
          cc += 1
        }
        val pivotToRowNormRatio = 
          abs(result(potentialPivotIndex, c)) / rowL1Norm
        if (pivotToRowNormRatio > bestPivotAbsToRowNormRatio) {
          bestPivotAbsToRowNormRatio = pivotToRowNormRatio
          pivotIndex = potentialPivotIndex
        }
        potentialPivotIndex += 1
      }
      
      // throw an exception if no acceptable pivot found
      if (pivotIndex == -1) {
        throw new IllegalArgumentException(
          "Cannot perform LU-decomposition on singular matrix"
        )
      }
      
      // swap rows 
      val temp = result.entries(pivotIndex)
      result.entries(pivotIndex) = result.entries(c)
      result.entries(c) = temp
      val tempIndex = permutation(pivotIndex)
      permutation(pivotIndex) = permutation(c)
      permutation(c) = tempIndex
      
      // eliminate lower right submatrix
      var r = c + 1
      while (r < n) {
        val ratio = result(r, c) / result(c, c)
        var cc = c + 1
        while (cc < n) {
          result(r, cc) -= result(c, cc) * ratio
          cc += 1
        }
        result(r, c) = ratio
        r += 1
      }
      
      c += 1
    }
    
    (result, permutation)
  } 
  
  /**
   * Solving dense linear equations with pivoted LU-decomposition
   */
  def \(other: Mat): Mat = {
    require(this.width == this.height && this.width == other.height)
    val n = this.width
    val (lu, permutation) = luDecomposition
        
    // permutate the input matrix 
    val permutatedEntries = new Array[Array[Double]](other.height)
    var i = 0
    while (i < n) {
      permutatedEntries(i) = other.entries(permutation(i))
      i += 1
    }
    
    val permutatedInputMatrix = new Mat(n, other.width, 0d, permutatedEntries)
    
    val result = permutatedInputMatrix.clone
    for (c <- (0 until other.width).par) {
      // solve Lx = y by forward substitution
      var r = 0
      while (r < n) {
        var cc = 0
        while (cc < r) {
          result(r, c) -= result(cc, c) * lu(r, cc)
          cc += 1
        }
        r += 1
      }
      
      // solve Rz = x by backward substitution
      r = n - 1
      while (r >= 0) {
        var cc = r + 1
        while (cc < n) {
          result(r, c) -= result(cc, c) * lu(r, cc)
          cc += 1
        }
        result(r, c) /= lu(r, r)
        r -= 1
      }
    }
    
    result
  }
  
  /**
   * Cloning this matrix
   */
  override def clone: Mat = {
    val result = new Mat(height, width, defaultValue)
    var r = 0
    while (r < height) {
      val row = entries(r)
      val resultRow = result.entries(r)
      var c = 0
      while (c < width) {
        resultRow(c) = row(c)
        c += 1
      }
      r += 1
    }
    result
  } 

  def isNaN = exists{_.isNaN}
  def isInfinite = exists{_.isInfinite}
  override def isInvalid = exists{x => x.isNaN || x.isInfinite}
    
  override def hashCode: Int = {
    ((width * height + 37) >>> height + height * 79) >>> width +
    (for (i <- 0 until 10; j <- 0 until 10) yield 
       entries(i * 179 % height)(j * 573 % width)
    ).sum.hashCode
  }
  
  override def equals(other: Any) = other match {
    case m: Mat => {
      if (m.height == this.height && m.width == this.width) {
        forallWithIndex{(r, c, e) => m(r, c) == e }
      } else {
        false
      }
    }
    case _ => false
  }
  
  override def toImage(colormap: Double => Int) = {
    org.kramerlab.autoencoder.visualization.draw(this, colormap)
  }
  
  def toImage = {
    org.kramerlab.autoencoder.visualization.draw(this)
  }
}

object Mat {

  def empty(height: Int, width: Int, defaultValue: Double): Mat = { 
    if (defaultValue == 0) {
      new Mat(height, width, 0, Array.ofDim[Double](height, width))
    } else {
      new Mat(height, width, defaultValue, 
          Array.fill(height, width){defaultValue})
    }
  }

  def empty(height: Int, width: Int): Mat = empty(height, width, 0d)

  case class MatInitializer(height: Int, width: Int, defaultValue: Double) {
    def this(height: Int, width: Int) = this(height,width,0d)
	
	def apply[X <% Double](entries: Traversable[X]): Mat = {
      require(entries.size == height * width, 
  	  "Cannot build a " + (height, width) + "-matrix" +
  	  " from " + entries.size + " values")
  
      val result: Mat = createDirty(height, width, defaultValue)
      var r = 0
      var c = 0
      for (e <- entries) {
        result(r, c) = e
        c += 1
        if (c == width) {
          c = 0
          r += 1
        }
      }
      result
    }

	def apply(entries: Double*): Mat = {
	  apply(entries.asInstanceOf[Traversable[Double]])
	}

  }

  def apply(
    height: Int,
    width: Int,
    defaultValue: Double
  ): MatInitializer = 
	new MatInitializer(height, width, defaultValue)

  def apply(height: Int, width: Int): MatInitializer = apply(height, width, 0d)
  
  /**
   * This method creates a matrix of correct dimension, with correct default 
   * value,
   * but without any guarantees on content of the entries array.
   */
  def createDirty(height: Int, width: Int, defaultValue: Double): Mat = 
    new Mat(height, width, defaultValue, Array.ofDim[Double](height, width))

  def fill(height: Int, width: Int, defaultValue: Double)
    (f: (Int, Int) => Double): Mat = {

	val result = createDirty(height, width, defaultValue)
	// obviously could expressed by map, optimally by from an
	// immutable empty virtual matrix that does not occupy space
	// alternavely by inplace-map on a dirty matrix
	// TODO: this is pure laziness, against the convention of
	// this implementation to duplicate stupid code as often as possible!
	result.mapWithIndex{(i,j,x) => f(i,j)}
  }
  
  def fill(height: Int, width: Int)(f: (Int, Int) => Double): Mat = {
    fill(height, width, 0d)(f)
  }

  def fill(
    height: Int, width: Int, 
	defaultValue: Double, constant: Double): Mat = 
    new Mat(height, width, defaultValue, 
	  Array.fill(height, width){constant})
  
  /* this doesn't work: requires something like a MatrixFiller
  def fill(height: Int, width: Int)(f: () => Double): Mat = {
    fill(height, width, 0d){(x,y) => f()}
  }
  */

  def ones(height: Int, width: Int): Mat = fill(height, width, 0d, 1d)

  def diag(entries: Mat): Mat = {
    val n = max(entries.width, entries.height)
    Mat.fill(n, n, 0d) {(i, j) => if (i == j) entries(i) else 0d}
  }
  
  def eye(n: Int): Mat = diag(Mat.fill(1, n, 0, 1d))
  
  private[Mat] def requireSameDimensions(
    a: Mat, b: Mat, 
	operationDescription: String) {
    
	require(a.size == b.size, "Matrix dimensions are incompatible. For " +
	  operationDescription + " the dimensions must coincide, but are " +
	  a.size + "!=" + b.size)
  }

  private[Mat] def requireMultipliableDimensions(a: Mat, b: Mat) {
    require(a.width == b.height, "Matrix dimensions incompatible for " +
	  "multiplication. The width of the first matrix does not coincide " +
	  "with the height of the second matrix: " + a.size + " " + b.size)
  }
  
  def main(args: Array[String]) {
    val a = Mat(2, 2)(1, 2, 3, 4)
    val b = Mat(2, 3)(13, 13, 19, 29, 33, 43)
    val c = a \ b
    println(c)
    
    val rnd = Mat.fill(8, 8, 0d){(i, j) => math.random * 100}
    val rightSide = Mat.fill(8, 1, 0d){(i, j) => sin(i)}
    val x = rnd \ rightSide
    val reconstruction = rnd * x
    println((rightSide - reconstruction).l2Norm)
    
    val size = 8
    val permutationSubject = Mat.fill(size, 1, 0d){(i, j) => i}
    val permutation = org.kramerlab.autoencoder.math.random.permutation(size)
    permutationSubject.permutateRows(permutation)
    println(permutation.mkString("[", ",", "]"))
    println(permutationSubject)
  }
  
}