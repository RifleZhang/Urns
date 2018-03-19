package type;

/**
 * <p>Title: </p>
 *
 * <p>Description: </p>
 *
 * <p>Copyright: Copyright (c) 2004</p>
 *
 * <p>Company: </p>
 *
 * @author not attributable
 * @version 1.0
 */



public class UrnParameters
{
  public double c;
  public double exptruth;
  public double e;
  public ExtractionUrnCounts urnTotals;
  public UrnParameters()
  {}


  public UrnParameters(double cIn, double expIn)
  {
    c = cIn;
    exptruth = expIn;
  }
}
