package synthetic;

import type.Extraction;
import type.SyntheticParam;
import type.SyntheticFeature;

import java.util.Arrays;

public class SyntheticTraining {
    public final static double eps = 1e-1;
    public SyntheticParam params;
    public int[] C;
    public int[] E;
    public Extraction ex;
    public double[] feature;
    public double expTruth;
    public double logLenC;

    SyntheticTraining(SyntheticParam params) {
        this.params = params;
        this.expTruth = params.zC;
        this.logLenC = Math.log(params.lenC);
    }

    public void reverse(int[] a) {
        int len = a.length;
        int tmp;
        for (int i = 1; i <= len / 2; i++) {
            tmp = a[i];
            a[i] = a[len - i];
            a[len - i] = tmp;
        }
    }

    void generateExtraction() {
        int lenC = C.length - 1, lenE = E.length - 1;
        Arrays.sort(C, 1, lenC + 1);
        reverse(C);
        Arrays.sort(E, 1, lenE + 1);
        reverse(E);

        int lc, le;
        int[] extractions = new int[lenC + lenE + 1];
        int[] labels = new int[lenC + lenE + 1];
        lc = le = 1;
        int t = 0;
        while ((lc <= lenC && C[lc] > 0 + eps) || (le <= lenE && E[le] > 0 + eps)) {
            t++;
            if (lc > lenC || C[lc] < 0 + eps) {
                extractions[t] = E[le];
                le++;
                continue;
            }
            if (le > lenE || E[le] < 0 + eps) {
                extractions[t] = C[lc];
                labels[t] = 1;
                lc++;
                continue;
            }

            if (C[lc] > E[le]) {
                extractions[t] = C[lc];
                labels[t] = 1;
                lc++;
            } else {
                extractions[t] = E[le];
                le++;
            }
        }
        ex = new Extraction();
        ex.extractions = Arrays.copyOfRange(extractions, 1, t+1);
        ex.labels = Arrays.copyOfRange(labels, 1,t+1);
    }

    int bsearch (double[] sum, double p) {
        int l = 1;
        int r = sum.length + 1;
        while (l + 1 < r) {
            int mid = (l+r)/2;
            if (sum[mid] > p)
                r = mid;
            else
                l = mid;
        }
        return l;
    }

    void generateCAndE() {
        C = new int[params.lenC + 1];
        E = new int[params.lenE + 1];

        C[0] = 0;
        E[0] = 0;

        params.reRandN();

        for (int i=0; i<params.N; i++) {
            if (Math.random() < params.precision) {
                double p = Math.random();
                int j = bsearch(params.pc, p);
                C[j]++;
            }
            else {
                double p = Math.random();
                int j = bsearch(params.pe, p);
                E[j]++;
            }
        }
    }

    public SyntheticFeature generateTrainingData () {
        generateCAndE();
        generateExtraction();
        double[] feature = FeatureExtractor.extractFeature(ex);
        SyntheticFeature f = new SyntheticFeature(feature, logLenC, expTruth);
        return f;
    }
}
