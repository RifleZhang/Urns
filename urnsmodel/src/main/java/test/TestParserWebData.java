package test;

import type.Extraction;
import tool.Parser;

public class TestParserWebData {
     static void testReadExtractionFromFile() {
        Extraction[] ex = Parser.readExtractionFromFile("urnFreq.txt", 2);
        System.out.println(ex[1].className);
        int[] extractions = ex[1].extractions;
        int len = extractions.length;
        System.out.println(len);
        for (int i=0; i<len; i++) {
            System.out.print(extractions[i]);
            System.out.print(" ");
        }
    }

    static void testReadInstanceFromFile() {
        Extraction[] ex = Parser.readInstanceFromFile("urnName.txt", 50);
        int idx = 2;
        System.out.println(ex[idx].className);
        String[] instances = ex[idx].instances;
        int len = instances.length;
        System.out.println(len);
        for (int i=0; i<len; i++) {
            System.out.print(instances[i]);
            System.out.print(" ");
        }
    }

    public static void main(String args[]) {
        testReadInstanceFromFile();
    }
}
