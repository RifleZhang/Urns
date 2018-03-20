package tool;

import type.*;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Scanner;

/**
 * Created by riflezhang on 4/10/17.
 */
public class Parser {
    public final static String preloadFile = "big_dist.txt";

    public double[] parse (int len, String Name) {
        Scanner scanner = null;
        try {
            scanner = new Scanner(new File(Name));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        double[] a = new double [len+1];
        a[0] = len;

        for (int i=1; i<=len; i++) {
            scanner.nextDouble();
            scanner.next();
            a[i] = scanner.nextDouble();
            scanner.next();
        }
        return a;
    }

    public static double[] parseWekaOutput (int len, String filename) {
        ClassLoader classLoader = Parser.class.getClassLoader();
        File file = new File(classLoader.getResource(filename).getFile());
        Scanner scanner = null;
        try {
            scanner = new Scanner(file);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        double[] a = new double [len];

        for (int i=0; i<len; i++) {
            scanner.nextDouble();
            scanner.next();
            a[i] = scanner.nextDouble();
            scanner.next();
        }
        return a;
    }

    public static UrnParameters[] readUrnParams (int len, String cFile, String expFile) {
        double[] logLenC = parseWekaOutput(len, cFile);
        double[] exp = parseWekaOutput(len, expFile);

        UrnParameters[] params = new UrnParameters[len];
        for (int i=0; i<len; i++) {
            params[i] = new UrnParameters(logLenC[i], exp[i]);
        }
        return params;
    }

    public int[] realCounts (String Name) {
        Scanner scanner = null;
        try {
            scanner = new Scanner(new File(Name));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        ArrayList<Integer> a = new ArrayList<Integer>();
        while (scanner.hasNextInt()) {
            a.add(scanner.nextInt());
        }
        int[] ret = new int[a.size() + 1];
        ret[0] = a.size();
        for (int i=1; i<=a.size(); i++)
            ret[i] = a.get(i-1);
        return ret;
    }

    public static int isInt(String s) {
        try {
            int op1 = Integer.parseInt(s);
            return op1;
        } catch (NumberFormatException e) {
            return -1;
        }

    }

    public static ArrayList<DataPoint> parseLabeled (String Name) {
        Scanner scanner = null;
        try {
            scanner = new Scanner(new File(Name));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        ArrayList<DataPoint> a = new ArrayList<DataPoint>();
        int tot = 0;
        int prev = 0;
        while (scanner.hasNext()) {
            String s = scanner.next();
            int tmp = isInt(s);
            if (tmp != -1) {
                tot += tmp;
                prev = tmp;
            }
            else {
                tot -= prev;
                DataPoint dp = new DataPoint(tot, prev);

                String ss;
                while (scanner.hasNext()) {
                    ss = scanner.next();
                    tmp = isInt(ss);
                    if (tmp != -1) {
                        tot = tmp;
                        break;
                    }
                    else{
                        s = s + " " + ss;
                    }
                }

                dp.name = s;
                a.add(dp);
            }
        }
        /*
        DataPoint[] array = new DataPoint[a.size() + 1];
        for(int i=0; i<array.length; i++)
            array[i] = a.get(i);
            */
        return a;
    }

    public static Extraction[] readExtractionFromFile (String filename, int n) {
        Extraction[] ex = new Extraction[n];

        ClassLoader classLoader = Parser.class.getClassLoader();
        File file = new File(classLoader.getResource(filename).getFile());
        Scanner scanner = null;
        try {
            scanner = new Scanner(file);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        for (int i=0; i<n; i++) {
            String classname;
            int m;
            classname = scanner.next();
            m = scanner.nextInt();
            int[] extractions = new int[m];
            for (int j=0; j<m; j++) {
                extractions[j] = scanner.nextInt();
            }
            ex[i] = new Extraction(extractions, classname);
        }

        return ex;
    }

    public static Extraction[] readInstanceFromFile (String filename, int n) {
        Extraction[] ex = new Extraction[n];

        ClassLoader classLoader = Parser.class.getClassLoader();
        File file = new File(classLoader.getResource(filename).getFile());
        Scanner scanner = null;
        try {
            scanner = new Scanner(file);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        for (int i=0; i<n; i++) {
            String classname;
            int m;
            classname = scanner.next();
            // System.out.print(i + " ");
            // System.out.println(classname);
            m = scanner.nextInt();
            String[] instances = new String[m];
            for (int j=0; j<m; j++) {
                if (scanner.hasNextInt()) {
                    instances[j] = Integer.toString(scanner.nextInt());
                } else {
                    instances[j] = scanner.next();
                }
            }
            ex[i] = new Extraction(instances, classname);
        }

        return ex;
    }

    public static LabelData[] readLabelFromFile(String filename, int n) {
        ClassLoader classLoader = Parser.class.getClassLoader();
        File file = new File(classLoader.getResource(filename).getFile());
        Scanner scanner = null;
        try {
            scanner = new Scanner(file);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        LabelData[] data = new LabelData[n];
        for (int i=0; i<n; i++) {
            data[i] = new LabelData(
                    scanner.next(),
                    scanner.next(),
                    scanner.nextInt(),
                    scanner.nextInt()
            );
        }
        return data;
    }

    public static Extraction[] readProbFromFile (String filename, int n) {
        Extraction[] ex = new Extraction[n];

        ClassLoader classLoader = Parser.class.getClassLoader();
        File file = new File(classLoader.getResource(filename).getFile());
        Scanner scanner = null;
        try {
            scanner = new Scanner(file);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        for (int i=0; i<n; i++) {
            String classname;
            int m;
            classname = scanner.next();
            m = scanner.nextInt();
            //System.out.println(i + " " + classname + " " + m);
            double[] probs = new double[m];
            for (int j=0; j<m; j++) {
                probs[j] = scanner.nextDouble();
            }
            ex[i] = new Extraction();
            ex[i].probs = probs;
            ex[i].className = classname;
        }

        return ex;
    }

    public static void preload(int[] sizeN, int[] lenW) {
        ClassLoader classLoader = Parser.class.getClassLoader();
        File file = new File(classLoader.getResource(preloadFile).getFile());
        Scanner scanner = null;
        try {
            scanner = new Scanner(file);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        int n = SyntheticParam.preloadSize;
        for (int i=0; i < n; i++) {
            sizeN[i] = scanner.nextInt();
        }
        for (int i=0; i < n; i++) {
            lenW[i] = scanner.nextInt();
        }
    }


    public static void main(String args[]) {
        readProbFromFile("EMProbs.txt", 50);

    }
}
