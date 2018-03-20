package type;

public class SyntheticFeature {
    // vector input: logLen, sigletonRate/log, low5, mean, median, top10, top50, top100, top1000
    // vector output: logLenC, zC

    public static int featureCount = 9;
    public static String[] featureNames = {
            "logLen",
            "singleton",
            "low5",
            "mean",
            "median",
            "top10",
            "top50",
            "top100",
            "top1000"
    };
    public static String[] labelNames = {
            "logLenC",
            "expTruth"
    };

    public double[] feature;
    public double[] label;

    public SyntheticFeature(double[] feature) {
        this.feature = feature;
    }

    public SyntheticFeature(double[] feature, double logLenC, double expTruth) {
        this.feature = feature;
        this.label = new double[2];
        this.label[0] = logLenC;
        this.label[1] = expTruth;
    }

}
