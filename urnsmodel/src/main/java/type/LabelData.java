package type;

public class LabelData {
    public String classname;
    public String instancename;
    public int freq;
    public int label;

    public LabelData(String instancename, String classname, int freq, int label) {
        this.instancename = instancename;
        this.classname = classname;
        this.freq = freq;
        this.label = label;
    }
}
