package tool;

import synthetic.FeatureExtractor;

import java.io.FileNotFoundException;
import java.io.PrintWriter;

public class WekaArffHeader {
    public static void printArffHeaderToFile(PrintWriter writer, String prediction) {
        String[] attributes = FeatureExtractor.featureNames;
        int n = FeatureExtractor.featureCount;
        writer.println("@RELATION urnsModelData\n");
        for (int i=0; i<n; i++) {
            writer.println("@ATTRIBUTE " + attributes[i] + " " + "NUMERIC");
        }
        writer.println(prediction);
        writer.println("\n@DATA");
    }
    public static void main(String args[]) throws FileNotFoundException {
        PrintWriter writer = new PrintWriter("testHeader.arff");
        printArffHeaderToFile(writer, "");
        writer.close();

    }
}
