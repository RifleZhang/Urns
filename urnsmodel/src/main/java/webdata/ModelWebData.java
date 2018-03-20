package webdata;

import type.Extraction;
import type.SyntheticParam;
import type.UrnParameters;
import tool.Parser;
import urns.CalProbability;

import java.io.IOException;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.text.NumberFormat;

public class ModelWebData {

    public static double[] assignModel(int[] extractions, UrnParameters param) {
        int len = extractions.length;
        double[] probs = new double[len];
        double lenC = Math.exp(param.c);
        double expC = param.exptruth;
        double expE = Math.min(expC, 1.0001);
        // TODO: need to sync with generation of data from SyntheticMain
        double lenE = 100000; // should it be 1M or 100K, check

        double precision = SyntheticParam.precision;

        assert (len != 0);

        int totalCount = 0;
        for (int i=0; i<len; i++) {
            totalCount += extractions[i];
        }

        for (int i=0; i<len; i++) {
            probs[i] = CalProbability.oneUrnProbability(
                    extractions[i],
                    expC,
                    expE,
                    lenC,
                    lenE,
                    precision,
                    totalCount
            );
        }
        return probs;
    }

    public static Extraction[] runModel(String filename, int n, String cFile, String expFile) {
        Extraction[] extractions = Parser.readExtractionFromFile(filename, n);
        UrnParameters[] params = Parser.readUrnParams(n, cFile, expFile);

        final long startTime = System.currentTimeMillis();
        for (int i=0; i<n; i++) {
            System.out.println(i);
            extractions[i].ModelProb = assignModel(extractions[i].extractions, params[i]);
        }
        final long endTime = System.currentTimeMillis();
        System.out.println("Run urn model time:" + (double)(endTime - startTime) / 1000.0);

        return extractions;
    }

    public static void main(String args[]) throws IOException {
        int n = 50;
        Extraction[] ex = runModel("urnFreq.txt", n, "paramC.txt", "paramExpTruth.txt");
        PrintWriter writer;
        writer = new PrintWriter("ModelProbs.txt");

        for (int i=0; i<n; i++) {
            writer.print(ex[i].className + " ");
            double[] prob = ex[i].ModelProb;
            int len = prob.length;
            writer.print(len);
            writer.print("\n");
            NumberFormat formatter = new DecimalFormat("#0.000");
            for (int j=0; j<len; j++) {
                writer.print(formatter.format(prob[j]) + " ");
            }
            writer.print("\n");
        }
        writer.close();
    }
}
