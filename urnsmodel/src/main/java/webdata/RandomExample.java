package webdata;

import type.Extraction;
import tool.Parser;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.concurrent.ThreadLocalRandom;

public class RandomExample {
    public static void randomExamplesToFile (int n, String outFile, String dataFile) throws FileNotFoundException {
        Extraction[] ex = Parser.readInstanceFromFile(dataFile, n);
        PrintWriter writer = new PrintWriter(outFile);
        int randn = 2;
        for (int i=0; i<n; i++) {
            int len = ex[i].instances.length;
            String className = ex[i].className;
            for (int j=0; j<randn; j++) {
                int randIdx = ThreadLocalRandom.current().nextInt(0, len);
                writer.print(i + " " + className + " " + ex[i].instances[randIdx] + "\n");
            }

        }
        writer.close();
    }

    public static void main(String args[]) throws FileNotFoundException {
        randomExamplesToFile(50, "labelPairs.txt", "urnName.txt");
    }
}
