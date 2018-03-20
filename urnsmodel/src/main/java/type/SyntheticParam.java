package type;

import tool.Parser;

import java.util.Random;


public class SyntheticParam {
    //public static double[] precisions = {0.9, 0.85, 0.8};
    // fix precision to be 0.36, which is calculated by hand-labeled data
    // TODO: Since the precision is smaller, should make the curve of E flatter
    public final static double precision = 0.36;
    // TODO: think about range of C
    public final static int rangeC = 5000;
    public final static double eps = 1e-6;
    final static double perturb = 0.1;

    public final static int preloadSize = 500;
    public final static int[] N_dist = new int[preloadSize];
    public final static int[] lenW_dist = new int[preloadSize];

    public final static double[] sizeC;

    static {
        Parser.preload(N_dist, lenW_dist);
        sizeC = new double[rangeC + 1];
        double tot = 0;
        for (int i = 5; i <= rangeC; i++) {
            sizeC[i] = 1 / (double) i;
            tot += sizeC[i];
        }
        for (int i = 5; i <= rangeC; i++) {
            sizeC[i] /= tot;
        }
    }

    public double distProb;
    public int lenC; // lenC in [5, rangeC]
    public int lenE; // lenW - lenC
    public int lenW; // generated by pre-loaded distribution
    public double zC; // zC in [0.8, 1.8]
    public double zE; // zE min (zC, 1)
    public int N; // generated by pre-loaded distribution
    // c and e prob distribution
    public double[] pc;
    public double[] pe;

    void norM(double[] a) {
        double tot = 0;
        int len = a.length;
        for (int i = 0; i < len; i++)
            tot += a[i];
        for (int i = 0; i < len; i++)
            a[i] = a[i] / tot;
    }

    void cumulate (double[] prob) {
        int len = prob.length;
        for (int i=1; i<len; i++) {
            prob[i] += prob[i-1];
        }
    }

    // using discrete approximation for prob
    void genProbabilityDistribution() {
        pc = new double[lenC+1]; // saves the frequency of extractions of targets
        pe = new double[lenE+1]; // saves the frequency of extractions of errors

        pc[0] = 0;
        for (int i = 1; i <= lenC; i++) {
            pc[i] = Math.pow(i, -zC);
        }
        norM(pc);
        cumulate(pc);

        pe[0] = 0;
        for (int i = 1; i <= lenE; i++) {
            pe[i] = Math.pow(i, -zE);
        }
        norM(pe);
        cumulate(pe);
    }

    double perturb(double p) {
        Random r = new Random();
        double ret = r.nextGaussian() * perturb + p;
        ret = Math.max(ret, 0);
        ret = Math.min(ret, 1-eps);
        return ret;
    }

    /**
     * lenW, n and lenC are all correlated for now.
     * lenC doesn't matter that match because lenC << lenW
     */
    public void initParam() {
        int idx;
        idx = (int)(distProb * N_dist.length);
        lenW = lenW_dist[idx];

        double p = perturb(distProb);
        idx = (int) (p * N_dist.length);
        N = N_dist[idx];

        zC = 0.8 + Math.random() + eps;
        zE = Math.min(zC, 1.0001);

        //double tmp = Math.random();
        p = perturb(distProb);
        idx = 5;
        while (p > sizeC[idx] && idx < rangeC) {
            p -= sizeC[idx];
            idx++;
        }
        lenC = idx;

        lenE = lenW - lenC;
    }

    public void reRandN() {
        int idx = (int) (Math.random() * N_dist.length);
        N = N_dist[idx];
    }

    public SyntheticParam(double p) {
        this.distProb = p;
        initParam();
        genProbabilityDistribution();
    }

    public SyntheticParam() {
        this.distProb = Math.random();
        initParam();
        genProbabilityDistribution();
    }

    public static void main(String args[]) {
        SyntheticParam param = new SyntheticParam();
        System.out.println("n: " + param.N);
        System.out.println("prob: " + param.distProb);
        System.out.println("lenW: " + param.lenW);
        System.out.println("lenE: " + param.lenE);
        System.out.println("lenC: " + param.lenC);
        //System.out.println((int)((1-eps) * 500));

    }
}
