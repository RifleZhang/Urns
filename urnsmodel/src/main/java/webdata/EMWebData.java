package webdata;

import type.Extraction;
import tool.Parser;
import urns.CalProbability;

import java.io.IOException;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.text.NumberFormat;

public class EMWebData {
    // Sync with handlabel data
    public static double pval = 0.35;

    public static double[] assignEM(int[] extractions) {
        int len = extractions.length;
        int[][] EMExtractions = new int[len][1];
        double [] precs = new double[] {pval};

        for(int i=0;i<extractions.length; i++) {
            EMExtractions[i][0] = extractions[i];
        }
        CalProbability cal = new CalProbability();

        double[] prob = new double[len];
        try {
            prob = cal.probabilitiesForCounts(EMExtractions, precs);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return prob;
    }

    public static Extraction[] runEM(String filename, int n) {
        Extraction[] extractions = Parser.readExtractionFromFile(filename, n);

        final long startTime = System.currentTimeMillis();
        for (int i=0; i<n; i++) {
            System.out.println(i);
            extractions[i].EMProb = assignEM(extractions[i].extractions);
        }
        final long endTime = System.currentTimeMillis();
        System.out.println((double)(endTime - startTime) / 1000.0);

        return extractions;
    }

    public static void main(String args[]) throws IOException {
        int n = 50;
        Extraction[] ex = runEM("urnFreq.txt", n);
        PrintWriter writer;
        writer = new PrintWriter("EMProbs.txt");
        for (int i=0; i<n; i++) {
            writer.print(ex[i].className + " ");
            double[] prob = ex[i].EMProb;
            int len = prob.length;
            writer.print(len);
            writer.print("\n");
            NumberFormat formatter = new DecimalFormat("#0.000");
            for (int j=0; j<len; j++) {
                writer.print(formatter.format(prob[j]) + " ");
            }
            writer.print("\n");
        }
    }
}
