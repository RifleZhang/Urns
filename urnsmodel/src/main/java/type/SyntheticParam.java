package type;

public class SyntheticParam {
    //public static double[] precisions = {0.9, 0.85, 0.8};
    // fix precision to be 0.36, which is calculated by hand-labeled data
    public final static double precision = 0.36;
    public final static int rangeC = 10000;
    public final static double eps = 1e-1;
    // TODO: N and E can be generate by distribution
    public final static int[] sizeN = {10, 50, 250, 1000, 5000, 20000, 50000, 100000};
    // current 1/2 50k, 1/4 100k, 1/8 150k, 1/8 200k
    public final static double[] lenE_dist = {0.5, 0.25, 0.125, 0.125};
    public final static int[] lenE_choices = {50000, 100000, 150000, 200000};


    public final static double[] sizeC;
    static {
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

    public int lenC; // lenC in [5, rangeC]
    public int lenE; // gen by above criteria
    public double zC; // zC in [0.8, 1.8]
    public double zE; // zE min (zC, 1)
    public int N; // n 10, 50, 250, 1000, 5000, 20000, 50000, 100000
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

    // TODO: fix math for distribution
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

    public void initParam() {
        double select = Math.random();
        int idx;
        idx = -1;
        while ((select -= lenE_dist[++idx]) > 0);
        lenE = lenE_choices[idx];

        idx = (int) (Math.random() * sizeN.length);
        N = sizeN[idx];
        zC = 0.8 + Math.random() * (1.0 + eps);
        zE = Math.min(zC, 1.0001);
        double tmp = Math.random();
        idx = 5;

        while (tmp > sizeC[idx] && idx < rangeC) {
            tmp -= sizeC[idx];
            idx++;
        }
        lenC = idx;
    }

    public void reRandN() {
        int idx = (int) (Math.random() * sizeN.length);
        N = sizeN[idx];
    }

    public SyntheticParam() {
        initParam();
        genProbabilityDistribution();
    }


    public static void main(String args[]) {
        SyntheticParam param = new SyntheticParam();

    }
}
