package webdata;

import tool.Parser;
import type.Extraction;
import type.LabelData;

import java.io.FileNotFoundException;

public class EvaluateModels {
    static double eps = 1e-5;
    static int bsearch (int[] a, int x) {
        int l = 0;
        int r = a.length;
        while (l + 1 < r) {
            int mid = (l+r)/2;
            if (a[mid] > x)
                r = mid;
            else
                l = mid;
        }
        return l;
    }

    public static void eval(
            String extractionFile,
            String modelProbFile,
            String EMProbFile,
            String labelFile,
            int n) {
        int m = 200;
        LabelData[] label =Parser.readLabelFromFile(labelFile, m);
        Extraction[] ex = Parser.readExtractionFromFile(extractionFile, n);
        Extraction[] modelEx = Parser.readProbFromFile(modelProbFile, n);
        Extraction[] EMEx = Parser.readProbFromFile(EMProbFile, n);

        double modelL = 0;
        double EML = 0;
        for (int i=0; i<m; i++) {
            LabelData d = label[i];
            for (int j=0; j<n; j++) {
                if (d.classname.equals(ex[j].className)) {
                    int idx = bsearch(ex[j].extractions, d.freq);

                    System.out.println(d.instancename + " " + d.classname + " " + d.label + " " + d.freq);
                    System.out.println(ex[j].extractions[idx]);
                    System.out.println("Model Prob: " + modelEx[j].probs[idx]);
                    System.out.println("EM Prob: " + EMEx[j].probs[idx]);

                    if (d.label == 1) {
                        modelL += Math.log(modelEx[j].probs[idx] + eps);
                        EML += Math.log(EMEx[j].probs[idx] + eps);
                    }
                    else {
                        modelL += Math.log(1 - modelEx[j].probs[idx] + eps);
                        EML += Math.log(1 - EMEx[j].probs[idx] + eps);

                    }
                }
            }
        }
        System.out.println("Model log likelihood: " + modelL);
        System.out.println("EM log likelihood: " + EML);
    }

    public static void main(String args[]) throws FileNotFoundException {
        eval("urnFreq.txt",
                "ModelProbs.txt",
                "EMProbs.txt",
                "label.txt",
                49
                );
    }

}
