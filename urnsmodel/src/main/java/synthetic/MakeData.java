package synthetic;

import com.sun.tools.javac.util.ArrayUtils;
import urns.AssignUrnProbabilities;

import java.util.Arrays;
import java.util.Collections;

public class MakeData {

    public int numParam = 5; // len, largest num, singleton, mean, median,
    static double eps = 1e-6;
    public int[][] genData(int len) {
        int[][] extractionCounts = new int[len][1];
        for(int i=0;i<extractionCounts.length; i++) {
            extractionCounts[i][0] = Math.max(1, (int)Math.round(1000.0/(double)i + Math.random()*20 - 15));
        }
        return extractionCounts;
    }

    public int[] getParam (int[] ex) {
        int len = ex.length;

        int[] p = new int[numParam];
        p[0] = ex.length;
        Arrays.sort(ex);
        p[1] = ex[len-1];
        int i, j;
        for (i=0; ex[i] == 0; i++);
        for (j = i; ex[j] ==1; j++);
        p[2] = j-i;

        int tot = 0;
        for (i=0; i<len; i++)
            tot += ex[i];
        p[3] = tot/len;
        p[4] = ex[len/2];

        return p;
    }

    public double estimate (int[] C, double zC) {
        int len = C[0];
       // System.out.println(len + " " + C.length);
        double hn = 0.0;
        for (int i=1; i<=len; i++) {
            hn += 1/(Math.pow(i, zC));
        }
        double [] logP = new double[len+1];
        for (int i=1; i<=len; i++) {
            logP[i] = -zC * Math.log(i) - Math.log(hn);
        }

        double logPr = 0.0;
        for (int i=1; i<=len; i++)
            logPr += C[i] * logP[i];

        return logPr;
    }

    public double cal (int[] C) {
        double l = 0.1, r = 3.0;
        while (l < r - eps) {
            double mid1 = (l + r) / 2;
            double mid2 = (mid1 + r) / 2;
            double ans1, ans2;
            ans1 = estimate(C, mid1);
            ans2 = estimate(C, mid2);

            if (ans1 < ans2)
                l = mid1;
            else
                r = mid2;
        }
        return l;
    }

    public double test () {
        int[] C = {10, 26486, 12053, 5052, 3033, 2536, 2391, 1444, 1220, 1152, 1039};
        double ll = estimate(C, 1.45041);
        System.out.println(ll);
        double zC = cal(C);
        System.out.println(zC);
        return zC;
    }

    void makeData() {

    }

    public static void main(String args[]) throws Exception
    {

        //The following example shows how to use the main method for Urns,
        //probabilitiesForCounts(), in the single-urn case.
        int len = 1000;
        AssignUrnProbabilities a = new AssignUrnProbabilities();
        MakeData m = new MakeData();
        m.test();

        int[][] extractionCounts;
        double [] precisions = new double[] {0.9};
        extractionCounts = m.genData(len);

        int[] ex = new int[len];
        for (int i=0; i<len; i++)
            ex[i] = extractionCounts[i][0];

        int[] param = m.getParam (ex);

        double [] probEstimates = a.probabilitiesForCounts(extractionCounts, precisions);
        int [] C = new int[len+10];
        int t = 0;

        for(int i=0; i<probEstimates.length; i++) {
            if (probEstimates[i] > precisions[0])
                C[++t] = extractionCounts[i][0];
        }
        C[0] = t;
        double zC = m.cal(C);
        System.out.print(zC);
        for (int i=0; i<param.length; i++)
            System.out.print (" " + param[i]);
        System.out.println("\n");
    }
}
