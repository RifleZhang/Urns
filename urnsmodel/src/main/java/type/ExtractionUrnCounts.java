package type;

public class ExtractionUrnCounts
{
  private String _args_xml;
  private int _left_counts[];
  private int _right_counts[];

  public ExtractionUrnCounts(String argsXml, int leftCounts[],
                             int rightCounts[])
  {
    _args_xml = argsXml;
    _left_counts = leftCounts;
    _right_counts = rightCounts;
  }


  public int sumCounts()
  {
    return sumLeftCounts() + sumRightCounts();
  }


  public int sumLeftCounts()
  {
    int count = 0;
    for (int i = 0; i < _left_counts.length; i++) {
      count += _left_counts[i];
    }
    return count;
  }


  public int sumRightCounts()
  {
    int count = 0;
    for (int i = 0; i < _right_counts.length; i++) {
      count += _right_counts[i];
    }
    return count;
  }


  public void setLeftCounts(int leftCounts[])
  {
    this._left_counts = leftCounts;
  }
  public int [] getLeftCounts()
  {
    return _left_counts;
  }
  public void setRightCounts(int rightCounts[])
  {
    this._right_counts = rightCounts;
  }
  public int [] getRightCounts()
  {
    return _right_counts;
  }
  public void setArgsXml(String argsXml)
  {
    this._args_xml = argsXml;
  }
  public String getArgsXml()
  {
    return _args_xml;
  }

  public String toString()
  {
    String s = "";
    for (int i = 0; i < _left_counts.length; i++) {
      s += _left_counts[i] + "-";
    }
    for (int i = 0; i < _right_counts.length; i++) {
      s += _right_counts[i] + "-";
    }
    return s;
  }
}

