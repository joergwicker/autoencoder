package org.kramerlab.autoencoder.math

package object matrix {

  implicit def integerToConstantIndexSelector(i: Int) = ConstantIndexSelector(i)
  implicit def integerToConstantRangeSelector(c: Int) = c ::: (c + 1)
  
  object EndSelector extends IndexSelector {
    override def apply(dim: Int) = dim
  }
  
  object MiddleSelector extends IndexSelector {
    override def apply(dim: Int) = dim / 2
  }
  
  object AllSelector extends RangeSelector {
    override def apply(dim: Int) = (0, dim)
  }
  
  val end : IndexSelector = EndSelector
  val middle : IndexSelector = MiddleSelector
  val ::: : RangeSelector = AllSelector
}