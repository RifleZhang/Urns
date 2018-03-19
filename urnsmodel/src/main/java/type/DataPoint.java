package type;

/**
 * Created by riflezhang on 4/17/17.
 */
public class DataPoint {
    public int num, label;
    public String name;
    public double prob;

    public DataPoint(int num, int label) {
        this.num = num;
        this.label = label;
    }

    public DataPoint(int num, int label, String name) {
        this.num = num;
        this.label = label;
        this.name = name;
    }

    public String toString() {
        return num + " " + label + " " + name;
    }
}
