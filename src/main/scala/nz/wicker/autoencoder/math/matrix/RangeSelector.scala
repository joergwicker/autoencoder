package nz.wicker.autoencoder.math.matrix

trait RangeSelector extends (Int => (Int, Int))

trait IndexSelector extends (Int => Int) {
  def ::: (start: IndexSelector) = TwoEndRangeSelector(start, this)
  def ::: (i: Int) = TwoEndRangeSelector(ConstantIndexSelector(i), this)
}

case class ConstantIndexSelector(i: Int) extends IndexSelector {
  override def apply(dim: Int) = i
}

case class TwoEndRangeSelector(a: IndexSelector, b: IndexSelector) 
  extends RangeSelector {
  override def apply(dim: Int) = (a(dim), b(dim))
}