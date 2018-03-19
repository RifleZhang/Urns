package type;

import cern.jet.stat.Gamma;

//little class that holds doubles that get very large or very small:
public class BigDouble
{
  private double d;
  private int exp;


  public BigDouble(double m)
  {
    d = m;
    rescale();
  }

  public BigDouble(BigDouble m)
  {
    d = m.d;
    exp = m.exp;
  }

  public BigDouble times(BigDouble m)
  {
    BigDouble db = new BigDouble(m);
    db.timesEquals(this);
    return db;
  }

  public BigDouble times(double m)
  {
    return times(new BigDouble(m));
  }

  public BigDouble divide(BigDouble m)
  {
    BigDouble db = new BigDouble(this);
    db.divideEquals(m);
    return db;
  }

  public BigDouble divide(double m)
  {
    return divide(new BigDouble(m));
  }

  public BigDouble plus(BigDouble m)
  {
    BigDouble db = new BigDouble(m);
    db.plusEquals(this);
    return db;
  }

  public BigDouble plus(double m)
  {
    return plus(new BigDouble(m));
  }

  public BigDouble minus(BigDouble m)
  {
    BigDouble db = new BigDouble(this);
    db.minusEquals(m);
    return db;
  }

  public BigDouble minus(double m)
  {
    return minus(new BigDouble(m));
  }

  //updates the bigdouble to be m times bigger
  public void timesEquals(BigDouble m)
  {
    d *= m.d;
    exp += m.exp;
    rescale();
  }

  public void timesEquals(double m)
  {
    timesEquals(new BigDouble(m));
  }

  public void plusEquals(BigDouble m)
  {
    if ( Math.abs(m.exp - exp) > 20 && this.d != 0)
        return;
    if(this.d == 0) {d = m.d; exp = m.exp; return;}
    d += (m.d * Math.pow(10, m.exp - exp));
    rescale();
  }

  public void plusEquals(double m)
  {
    plusEquals(new BigDouble(m));
  }

  public void minusEquals(BigDouble m)
  {
    if ( Math.abs(m.exp - exp) > 20  &&this.d != 0)
        return;
    if(this.d == 0) {d = -m.d; exp = m.exp; return;}
    d -= (m.d * Math.pow(10, m.exp - exp));
    rescale();
  }

  public void minusEquals(double m)
  {
    minusEquals(new BigDouble(m));
  }

  public void divideEquals(BigDouble m)
  {
    d /= m.d;
    exp -= m.exp;
    rescale();
  }

  public void divideEquals(double m)
  {
    divideEquals(new BigDouble(m));
  }

  //returns natural log of x
  public static double lnSmall(BigDouble x) {
    return x.exp * Math.log(x.d);
  }

  //returns natural log of x
  public static BigDouble ln(BigDouble x)
  {
    return new BigDouble(x.exp * Math.log(x.d));
    //not handling overflow here (should never happen)
  }

  //returns x to the y
  public static BigDouble pow(double x, double y)
  {
    if(y==0) return new BigDouble(1.0);
    double decimalpart = y - Math.floor(y);
    int iters = Math.abs((int) Math.floor(y));
    BigDouble db = new BigDouble(Math.pow(x, decimalpart));
    db = new BigDouble(x);
    int powstaken = 1;
    while(powstaken < Math.abs(y))
    {
      db.timesEquals(db);
      powstaken *= 2;
    }
    double powsleft = y - (double)powstaken;
    if(y < 0) { db = new BigDouble(1).divide(db); powsleft = y + powstaken;}
    if(Math.abs(powsleft) > 1)
      return db.times(BigDouble.pow(x, powsleft));
    else
      return db.times(Math.pow(x, powsleft));
//    for (int i = 0; i < iters; i++) {
//      if (y < 0)
//        db.divideEquals(x);
//      else
//        db.timesEquals(x);
//    }
//    return db;
  }

  public static BigDouble factorial(int i)
  {
    if(i <= 1) return new BigDouble(1.0);
    return gamma(i+1);
  }


//  public static BigDouble gamma(double a)
//  {
//       return BigDouble.pow(Math.E, Gamma.logGamma(a));
//  }

  public static BigDouble gamma(double a)
  {

    double decimalpart = a - Math.floor(a);
    double start = 1.0;
    if (decimalpart != 0) start = Gamma.gamma(decimalpart);
    int iters = Math.abs( (int) Math.floor(a));
    BigDouble db = new BigDouble(start);
    int i = 0;
    if(decimalpart == 0) i = 1;
    for(; i<iters;i++)
    {
      if (a < 0)
        db.divideEquals((i + 1)*-1.0+decimalpart);
      else
        db.timesEquals(i+decimalpart);
    }
    return db;
  }

//  public static BigDouble upperGamma(double a, double x)
//  {
//    if(a < 0)
//    {
//      BigDouble uG = upperGamma(a+1, x);
//      uG.minusEquals(BigDouble.pow(Math.E, -x).times(BigDouble.pow(x, a)));
//      return uG.divide(a);
//    }
//    return new BigDouble(Gamma.incompleteGammaComplement(a, x)).times(gamma(a));
//  }

  public static BigDouble upperGamma(double a, double x)
  {
    if (a < 0) {
      double aOrig = a;
      BigDouble adj = new BigDouble(0);
      BigDouble aProd = new BigDouble(1);
      while (a < 0) {
        aProd.timesEquals(a);
        adj.plusEquals(BigDouble.pow(x, a).divide(aProd));
        a++;
      }
      BigDouble uG = upperGamma(a, x).divide(aProd);
      uG.minusEquals(BigDouble.pow(Math.E, -x).times(adj));
      return uG;
    }
    return new BigDouble(Gamma.incompleteGammaComplement(a, x)).times(gamma(a));
  }

  //gets double back into canonical form, which has
  //d between 1 and 10.
  private void rescale()
  {
    if(d == 0) {exp = 0; return;}
    int dexp = (int) Math.floor(Math.log(Math.abs(d))/Math.log(10));
    double newd = d/(Math.pow(10, dexp));
    d = newd;
    exp = exp + dexp;
  }

  public boolean expIsLargerThan(double m)
  {
    return exp > m;
  }

  public boolean isZero()
  {
    return d == 0.0;
  }

  public double toDouble()
  {
    return d*Math.pow(10, exp);
  }

  public String toString()
  {
    return d + "E" + exp;
  }

}
