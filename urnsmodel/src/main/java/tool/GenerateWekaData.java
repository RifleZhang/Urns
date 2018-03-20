package tool;

import synthetic.FeatureExtractor;
import type.Extraction;
import type.SyntheticFeature;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.text.NumberFormat;

public class GenerateWekaData {

    public static void printArffHeaderToFile(PrintWriter writer, int label) {
        String[] attributes = SyntheticFeature.featureNames;
        int n = SyntheticFeature.featureCount;
        writer.println("@RELATION urnsModelData\n");
        for (int i=0; i<n; i++) {
            writer.println("@ATTRIBUTE " + attributes[i] + " NUMERIC");
        }
        writer.println("@ATTRIBUTE " + SyntheticFeature.labelNames[label] + " NUMERIC");
        writer.println("\n@DATA");
    }

    public static void printWekaTrainingDataToFile(
            SyntheticFeature[] features,
            int labelIdx,
            String urnTrainingFile) throws FileNotFoundException {

        PrintWriter writer;
        // clear file
        writer = new PrintWriter(urnTrainingFile);

        printArffHeaderToFile(writer, labelIdx);

        int len = features.length;
        for (int i=0; i<len; i++) {
            double[] feature = features[i].feature;
            NumberFormat formatter = new DecimalFormat("#0.0000");
            for (int j=0; j<feature.length; j++) {
                writer.print(formatter.format(feature[j]) + ",");
            }
            writer.print(features[i].label[labelIdx] + "\n");
        }
        writer.close();
    }

    public static void printWekaTestingDataToFile(
            int n,
            String webData,
            String testingFile,
            String prediction) throws FileNotFoundException {
        PrintWriter writer;

        // clear file
        writer = new PrintWriter(testingFile);
        writer.close();

        writer = new PrintWriter(new FileOutputStream(
                new File(testingFile)),
                true);
        Extraction[] extractions = Parser.readExtractionFromFile(webData, n);
        extractions = FeatureExtractor.extractFeatures(extractions);

        WekaArffHeader.printArffHeaderToFile(writer, prediction);
        for (int i=0; i<n; i++) {
            double[] features = extractions[i].features;
            NumberFormat formatter = new DecimalFormat("#0.0000");
            for (int j=0; j<features.length; j++) {
                writer.print(formatter.format(features[j]) + ",");
            }
            writer.print("?\n");
        }
        writer.close();
    }
}
