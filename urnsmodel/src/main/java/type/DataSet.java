package type;

import java.util.ArrayList;

/**
 * Created by riflezhang on 4/17/17.
 */
public class DataSet {
    public ArrayList<DataPoint> dataPoints;
    public double zC, zE, precision;
    public int lenC, lenE;

    public DataSet(ArrayList<DataPoint> dataPoints) {
        this.dataPoints = dataPoints;
    }

    public void printData() {
        for(int i=0; i<dataPoints.size(); i++)
            System.out.println(dataPoints.get(i).toString());
    }

    public void print(double val, String s) {
        System.out.print(val + s);
    }

    public double CalPrecision() {
        double right = 0, wrong = 0;
        for(int i=0; i<dataPoints.size(); i++) {
            DataPoint dp = dataPoints.get(i);
            if (dp.label == 1) {
                right++;
            }
            else
                wrong++;
        }
        return right/(right + wrong);
    }

    public void calProb() {

    }

}
