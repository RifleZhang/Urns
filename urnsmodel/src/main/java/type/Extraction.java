package type;

public class Extraction {
    public String className;
    public int[] extractions;
    public double[] probs;
    public double[] EMProb;
    public double[] ModelProb;
    public double[] features;
    public int[] labels;
    public String[] instances;

    public Extraction() {}
    public Extraction(int[] extractions, String className) {
        this.extractions = extractions;
        this.className = className;
    }

    public Extraction(String[] instances, String className) {
        this.instances = instances;
        this.className = className;
    }
}
