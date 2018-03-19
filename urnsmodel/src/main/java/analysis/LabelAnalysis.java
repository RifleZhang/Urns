package analysis;

import type.DataPoint;
import type.DataSet;
import tool.Parser;

import java.util.ArrayList;

/**
 * Created by riflezhang on 4/17/17.
 */
public class LabelAnalysis {
    public ArrayList<DataPoint> labeledData;
    public DataSet ds;

    public double CalPrecision() {
        double right = 0, wrong = 0;
        for(int i=0; i<ds.dataPoints.size(); i++) {
            DataPoint dp = ds.dataPoints.get(i);
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

    public void analyzeLabel(String name) {
        labeledData = Parser.parseLabeled(name);
        ds = new DataSet(labeledData);

        ds.precision = CalPrecision();
        ds.print(ds.precision, "\n");
    }

    public static void main (String argv[]) {
        LabelAnalysis la = new LabelAnalysis();
        la.analyzeLabel("citydata.txt");
    }
}
