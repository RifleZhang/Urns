package synthetic;

import java.io.*;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Arrays;

import tool.GenerateWekaData;
import type.SyntheticParam;
import type.SyntheticFeature;
import urns.CalProbability;
import weka.core.*;

import java.util.Scanner;

/**
 * Created by riflezhang on 2/23/17.
 */
public class SyntheticMain {
    public static double[] precisions = {0.9, 0.85, 0.8};
    public static int[] sizen = {10, 50, 250, 1000, 5000, 20000, 50000, 100000};
    public static int rangeC = 100000;
    public static double eps = 1e-6;
    public static double precision = 0.36;
    public static int[] lenEs = {500000, 1000000, 1500000, 2000000};

    public double[] pC;
    public int lenC; // lenC in [5, 1000000]
    public int lenE;
    public double zC; // zC in [0.8, 1.8]
    public double zE; // zE min (zC, 1)
    public int n; // n 10, 50, 250, 1000, 5000, 20000, 50000, 100000
    public double[] pc;
    public double[] pe;
    public int[] C;
    public int[] E;
    public int[] extractionCounts;
    public ArrayList<Attribute> atts;
    public double[] probs;
    public short[] mark;
    public Instances training;
    public PrintWriter writer;
    public PrintWriter writer2;

    SyntheticMain() {

    }

    public void norM(double[] a, double p) {
        int len = (int) a[0];
        double tot = 0;
        for (int i = 1; i <= len; i++)
            tot += a[i];
        for (int i = 1; i <= len; i++)
            a[i] = a[i] / tot * p;
    }

    public void genP(int n, double precision, int lenE, int lenC, double zC, double zE) {
        this.n = n;
        this.precision = precision;
        this.lenE = lenE;
        this.lenC = lenC;
        this.zC = zC;
        this.zE = zE;

        pc = new double[lenC + 1]; // saves the frequency of extractions of targets
        pe = new double[lenE + 1]; // saves the frequency of extractions of errors

        pc[0] = lenC;
        pe[0] = lenE;

        for (int i = 1; i <= lenC; i++) {
            int j = 1 + (int) (Math.random() * lenC); // range 1 to lenC inclusive
            pc[i] = 1 / Math.pow(j, zC);
        }
        norM(pc, 1);

        for (int i = 1; i < lenE; i++) {
            int j = 1 + (int) (Math.random() * lenE);
            pe[i] = 1 / Math.pow(j, zE);
        }
        norM(pe, 1);
    }

    public int bsearch (double[] sum, double p) {
        int l = 0, r = (int)sum[0] + 1;
        while (l + 1 < r) {
            int mid = (l+r)/2;
            if (sum[mid] > p)
                r = mid;

            else
                l = mid;
        }
        return r;
    }

    public void generateData() {
        C = new int[lenC + 1];
        E = new int[lenE + 1];

        C[0] = lenC;
        E[0] = lenE;

        assert(n != 0);
        int idx = (int) (Math.random() * sizen.length);
        n = sizen[idx];

        double[] sumc = new double[lenC+1];
        double[] sume = new double[lenE+1];

        for (int i=1; i<=lenC; i++)
            sumc[i] = sumc[i-1] + pc[i];
        for (int j=1; j<=lenE; j++)
            sume[j] = sume[j-1] + pe[j];
        sumc[0] = lenC;
        sume[0] = lenE;

        for (int i=0; i<n; i++) {
            if (Math.random() < precision) {
                double p = Math.random();
                int j = bsearch(sumc, p);
                C[j]++;
            }
            else {
                double p = Math.random();
                int j = bsearch(sume, p);
                E[j]++;
            }
        }

    }

    public void generateProbability() {
        double precision; // precision 0.9, 0.85, 0.8
        int lenE = 100000; // fix lenE to be 1,000,000
        int lenC;// lenC in [5, 1000000]
        double zC;// zC in [0.8, 1.8]
        double zE;// zE min (zC, 1)
        int n;// n 10, 50, 250, 1000, 5000, 20000, 50000, 100000

        int idx;
        precision = this.precision;

        idx = (int) (Math.random() * sizen.length);
        n = sizen[idx];
        zC = 0.8 + Math.random() * (1.0 + eps);
        zE = Math.min(zC, 1.0001);
        double tmp = Math.random();
        idx = 5;

        while (tmp > pC[idx] && idx < rangeC) {
            tmp -= pC[idx];
            idx++;
        }
        lenC = idx;
        genP(n, precision, lenE, lenC, zC, zE);
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

    public void process() {
        int lc, le;
        Arrays.sort(C, 1, lenC + 1);
        reverse(C);
        Arrays.sort(E, 1, lenE + 1);
        reverse(E);

        extractionCounts = new int[lenC + lenE + 1];
        mark = new short[lenC + lenE + 1];
        lc = le = 1;
        int t = 0;
        while ((lc <= lenC && C[lc] > 0 + eps) || (le <= lenE && E[le] > 0 + eps)) {
            t++;
            if (lc > lenC || C[lc] < 0 + eps) {
                extractionCounts[t] = E[le];
                le++;
                continue;
            }
            if (le > lenE || E[le] < 0 + eps) {
                extractionCounts[t] = C[lc];
                mark[t] = 1;
                lc++;
                continue;
            }

            if (C[lc] > E[le]) {
                extractionCounts[t] = C[lc];
                mark[t] = 1;
                lc++;
            } else {
                extractionCounts[t] = E[le];
                le++;
            }
        }
        extractionCounts[0] = t;
    }

    public void buildTraining() {
        // vector input: logLen, sigletonRate/log, low5, mean, median, top10, top50, top100, top1000
        // vector output: logLenC, zC
        int len;
        double top10, top50, top100, top1000;
        double singleton, low5 = 0;
        double median, mean;
        len = extractionCounts[0];
        singleton = 0;
        for (int i = 1; i <= len; i++)
            if (extractionCounts[i] == 1)
                singleton++;
        singleton /= (double)n;
        median = extractionCounts[len / 2];
        mean = (double) n / (double) len;


        top10 = top50 = top100 = top1000 = 1.0;
        double tot = 0;
        for (int i=1; i<=len; i++) {
            tot += (double)extractionCounts[i];
            if (extractionCounts[i] <= 5)
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
        /*
        Instance ie = new DenseInstance(8);
        ie.setValue(atts.get(0), Math.log((double) len));
        ie.setValue(atts.get(1), Math.log((double) singleton));
        ie.setValue(atts.get(2), mean);
        ie.setValue(atts.get(3), median);
        training.add(ie);
        */

        NumberFormat formatter = new DecimalFormat("#0.0000");
        writer.print(formatter.format(Math.log(len)) + "," + formatter.format(singleton) + "," + formatter.format(low5) + ","
                + formatter.format(mean) + "," + median + ","
        + formatter.format(top10) + "," + formatter.format(top50) + "," + formatter.format(top100) + "," + formatter.format(top1000) + ",");

        writer.print(formatter.format(Math.log(lenC)) + "," + formatter.format(zC) + "\n");
        //writer.print("?\n");
        /*
        int c1, c2;
        c1 = (int)Math.log(lenC) + 1;
        c1 = Math.min(c1, 10);
        c2 = (int)((zC - 0.8 + eps)/0.1) + 1;
        c2 = Math.min (c2, 10);
        writer.println("," + c1 + "," + c2);
        */

    }

    public void printExtraction() {
        int len = extractionCounts[0];
        writer2.print(len + " ");
        for (int i=1; i<=len; i++)
            writer2.print(extractionCounts[i] + " ");
        for (int i=1; i<=len; i++)
            writer2.print(mark[i] + " ");
        writer2.print("\n");

    }

    public void urnsWithEM() {
        int len = extractionCounts[0];

        int[][] extractions = new int[len][1];
        double [] precs = new double[] {0.9};

        for(int i=0;i<extractions.length; i++) {
            extractions[i][0] = extractionCounts[i+1];
        }
        CalProbability cal = new CalProbability();

        try {
            probs = cal.probabilitiesForCounts(extractions, precs);
        } catch (Exception e) {
            e.printStackTrace();
        }
        //for(int i=0; i<probs.length; i++)
        //    System.out.println(extractions[i][0] + "\t" + probs[i]);
        System.out.println (analysis());

    }

    public void assignProbs() {
        int len = extractionCounts[0];
        probs = new double[len];

        assert (len != 0);
        //System.out.println(zC + " " + zE + " " + lenC + " " + lenE + " " + precision + " " + n);
        for (int i=1; i<=len; i++) {
            probs[i-1] = CalProbability.oneUrnProbability(extractionCounts[i], zC, zE, (double)lenC, (double)lenE, precision, n);
        }

        //for (int i=0; i<len; i++)
        //   System.out.println(extractionCounts[i+1] + " " + probs[i]);
        System.out.println (analysis());
    }

    public double analysis() {
        int len = extractionCounts[0];
        double correct= 0;
        for (int i=1; i<=len; i++) {
            if (probs[i-1] > 0.5 ) {
                if (mark[i] == 1)
                    correct += 1.0;
            }
            else {
                if (mark[i] == 0)
                    correct += 1.0;
            }
        }
        //System.out.println (correct + " " + len);
        return correct/(double)len;
    }


    public void pre() {
        int ri = rangeC;
        pC = new double[ri + 1];
        double tot = 0;
        for (int i = 5; i <= ri; i++) {
            pC[i] = 1 / (double) i;
            tot += pC[i];
        }
        for (int i = 5; i <= ri; i++) {
            pC[i] /= tot;
            //System.out.print (g.sizeC[i] + " ");
        }

    }

    public void runAnalysis() {
        pre();
        generateProbability();
        generateData();
        process();
        printData();
        urnsWithEM();
        assignProbs();
        System.out.print(analysis());
    }

    public void printData() {
        System.out.println(n);
        System.out.print("|C| = " + lenC + ": ");

        for (int i = 1; i <= lenC; i++)
            System.out.print(" " + C[i]);

        System.out.print("\n");
        System.out.print("|E| = " + lenE + ": ");
        for (int i = 1; i <= lenE; i++)
            System.out.print(" " + E[i]);

        System.out.println("");
        int len = extractionCounts[0];
        System.out.print("|ExtractionCounts| = " + len + ": ");
        for (int i = 1; i <= len; i++)
            System.out.print(" " + extractionCounts[i]);
        System.out.print("\n");
    }

    public void runML(int set, double[] tc, double[] tzc) {
        try {
            Scanner scanner = new Scanner(new File("data2.txt"));
            for (int j=0; j<set; j++) {
                int len = 0;
                if (scanner.hasNextInt())
                    len = scanner.nextInt();
                extractionCounts = new int[len+1];
                extractionCounts[0] = len;
                for (int i = 1; i <= len; i++)
                    extractionCounts[i] = scanner.nextInt();
                mark = new short[len+1];
                mark[0] = (short)len;
                for (int i=1; i<=len; i++)
                    mark[i] = (short)scanner.nextInt();
                zC = tzc[j];
                zE = Math.min(zC, 1.0001);
                lenC = (int)Math.exp(tc[j]);
                lenE = 100000;
                precision = 0.9;
                n = 0;
                for (int i=1; i<=len; i++)
                    n += extractionCounts[i];
                assignProbs();
            }

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    public static SyntheticFeature[] genData(int bundle, int num) {
        SyntheticFeature[] features = new SyntheticFeature[bundle * num];
        int t = 0;
        for (int i = 0; i < bundle; i++) {
            if ((i+1) % 20 == 0) {
                System.out.println(i+1);
            }
            SyntheticParam param = new SyntheticParam();
            for (int j=0; j<num; j++) {
                SyntheticTraining train = new SyntheticTraining(param);
                features[t++] = train.generateTrainingData();
            }
        }
        return features;
    }

    public static void main(String args[]) throws FileNotFoundException {
        SyntheticFeature[] features = genData(800, 3);
        GenerateWekaData.printWekaTrainingDataToFile(features, 0, "TrainingLogLenC.arff");
        GenerateWekaData.printWekaTrainingDataToFile(features, 1, "TrainingExpTruth.arff");
    }
}
