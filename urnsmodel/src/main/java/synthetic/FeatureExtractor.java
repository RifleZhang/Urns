package synthetic;

import type.Extraction;

public class FeatureExtractor {
    // vector input: logLen, sigletonRate/log, low5, mean, median, top10, top50, top100, top1000
    // vector output: logLenC, zC

    // TODO: use enum

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

    public static double[] extractFeature(Extraction e) {
        double[] feature = new double[featureCount];
        int[] extractions = e.extractions;
        int n = 0;
        for (int num : extractions) {
            n += num;
        }
        int len;
        double top10, top50, top100, top1000;
        double singleton, low5 = 0;
        double median, mean;
        len = extractions.length;
        singleton = 0;
        for (int i = 0; i < len; i++)
            if (extractions[i] == 1)
                singleton++;
        singleton /= (double)n;
        median = extractions[len / 2];
        mean = (double) n / (double) len;


        top10 = top50 = top100 = top1000 = 1.0;
        double tot = 0;
        for (int i=0; i<len; i++) {
            tot += (double)extractions[i];
            if (extractions[i] <= 5)
                low5++;
            if (i == 10)
                top10 = tot/(double)n;
            if (i == 50)
                top50 = tot/(double)n;
            if (i == 100)
                top100 = tot/(double)n;
            if (i == 1000)
                top1000 = tot/(double)n;
        }
        low5 /= (double)n;

        feature[0] = Math.log(len);
        feature[1] = singleton;
        feature[2] = low5;
        feature[3] = mean;
        feature[4] = median;
        feature[5] = top10;
        feature[6] = top50;
        feature[7] = top100;
        feature[8] = top1000;
        return feature;
    }

    public static Extraction[] extractFeatures(Extraction[] extractions) {
        int len = extractions.length;
        for (int i=0; i<len; i++) {
            extractions[i].features = extractFeature(extractions[i]);
        }
        return extractions;
    }
}
