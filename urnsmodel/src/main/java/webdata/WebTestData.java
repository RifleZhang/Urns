package webdata;

import type.Extraction;
import synthetic.FeatureExtractor;
import tool.Parser;
import tool.WekaArffHeader;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.text.NumberFormat;

public class WebTestData {
    public static void printWebTestingDataToFile(
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

    public static void main(String args[]) throws FileNotFoundException {
        printWebTestingDataToFile(
                50,
                "urnFreq.txt",
                "webTestDataC.arff",
                "@ATTRIBUTE c  NUMERIC"
        );

        printWebTestingDataToFile(
                50,
                "urnFreq.txt",
                "webTestDataExpTruth.arff",
                "@ATTRIBUTE expTruth  NUMERIC"
        );
    }
}
