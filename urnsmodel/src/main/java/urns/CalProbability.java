package urns;

import java.util.*;

import cern.jet.stat.Gamma;
import cern.colt.matrix.impl.*;
import cern.colt.matrix.*;
import type.BigDouble;
import type.ExtractionUrnCounts;
import type.UrnParameters;
import weka.core.Matrix;
import weka.core.*;
import weka.classifiers.functions.LinearRegression;
/**
 * Performs urns-based assessment of extractions, which can come from
 * the database or (in the probabilitiesForCounts API) from an array
 * of extraction counts.
 *
 * reuse AssignUrnProbability
 */


public class CalProbability
{
    private static int MAXITERS = 10; //number of EM iterations to perform
    //for left-hand rules.
    public static UrnParameters _upLast; //set to most recently learned urnparameters
    public static long tot_time = 0;


    //see probabilitiesForCounts (below)
    //"total objects," if positive, is taken as the value of |C| + |E|
    //priorC, if positive, is taken as the prior on |C|
    //also sets the CSIZE field of the class for later lookup
    //(note, this last piece is obviously not thread safe).
    public static double[] probabilitiesForCountsConstrained(int[][] counts,
                                                             double[] precisions, int totalObjects, double priorC) throws
            Exception
    {
        int numUrns = precisions.length;
        double[] probs = new double[counts.length];
        ArrayList urnCounts = new ArrayList();

        int nTot = 0;
        for (int i = 0; i < counts.length; i++) {
            ExtractionUrnCounts euc = new ExtractionUrnCounts("dummyargs", new int[1],
                    new int[1]);
            if (numUrns == 1)
                euc.setLeftCounts(new int[] {counts[i][0]});
            else if (numUrns == 2) {
                euc.setLeftCounts(new int[] {counts[i][0]});
                euc.setRightCounts(new int[] {counts[i][1]});
            } else if (numUrns == 4) {
                euc.setLeftCounts(new int[] {counts[i][0], counts[i][1]});
                euc.setRightCounts(new int[] {counts[i][2], counts[i][3]});
            }
            nTot += euc.sumCounts();
            urnCounts.add(euc);
        }
        UrnParameters prior;
        double priorStrength = 1.0;
        if(nTot < 5000)
        {
            if(totalObjects > 0 && priorC > 0)
                prior = new UrnParameters(priorC, 1.5);
            else if (totalObjects > 0) { //no prior:
                prior = null;//new UrnParameters(priorC / (double) totalObjects, 1.5);
                priorStrength = 0.00;
            }
            else
                prior = new UrnParameters(1000.0, 1.5);
        }
        else
            prior = null;
        UrnParameters up;

        final long startTime = System.currentTimeMillis();
        up = computeUrnParameters(urnCounts, prior, precisions);
        final long endTime = System.currentTimeMillis();
        tot_time += endTime - startTime;

        Hashtable countsStrToProb = new Hashtable();
        for (int i = 0; i < counts.length; i++) {

            if (countsStrToProb.contains(urnCounts.get(i).toString())) {
                probs[i] = ( (Double) countsStrToProb.get(urnCounts.get(i).toString())).
                        doubleValue();
            }

            if (numUrns == 1)
                probs[i] = oneUrnProbability(counts[i][0], up.exptruth,
                        Math.min(up.exptruth - 0.0001, 1.0001),
                        up.c, up.e, precisions[0],
                        up.urnTotals.getLeftCounts()[0]);

            countsStrToProb.put(urnCounts.get(i).toString(), new Double(probs[i]));
        }
        _upLast = up;
        return probs;
    }

    /**
     * returns an array of probabilities for a given array of counts.
     * 'counts' is an int array of size [NUM_EXTRACTIONS][NUM_URNS]
     * storing the number of hits for each extraction, for each urn.
     * The number of urns must be either 1, 2, or 4.
     * 'precisions' is a double array of size [NUM_URNS], giving the
     * anticipated precision for each urn.
     * @param counts int[][]
     * @param precisions double[]
     * @return double[]
     */
    public static double[] probabilitiesForCounts(int[][] counts, double[] precisions) throws
            Exception
    {
        return probabilitiesForCountsConstrained(counts, precisions, -1, -1);
    }

    /**
     * Returns the UrnParameters for the given set of counts, using the
     * given prior, and assume the passed-in rule precisions.
     * UrnCounts is a list of ExtractionUrnCounts objects, and those
     * objects must have either:
     *     2 left counts and 2 right counts (in order lab1left, lab2left, lab1right, lab2right)
     *     1 left count  and 1 right count     or
     *     1 left count
     * If the rule precision array is specified, it must have the same
     * dimensionality as each element of urnCounts.
     * If either prior or precs is null, defaults are used.
     * @param urnCounts List
     * @param prior UrnParameters
     * @param precs double[]
     * @return UrnParameters
     * @throws Exception
     */
    private static UrnParameters computeUrnParameters(List urnCounts, UrnParameters prior, double[] precs) throws Exception
    {
        //get sample size
        //set up default values for params.
        //Loop for awhile:
        //  Assign probabilities to all facts.
        //  Learn new params from assigned probabilities
        int leftns[] = new int[1];
        int rightns[] = new int[1];
        if(precs.length==4)
        {
            leftns = new int[2];
            rightns = new int[2];
        }
//    File outfile = new File(Environment.getWorkingDirectory(),
//                        "urnresults_" + Environment.getCycleNumber() +
//                        ".txt");
//     BufferedWriter bw = new BufferedWriter(new FileWriter(outfile, true));

        int n = 0;


        for (Iterator it = urnCounts.iterator(); it.hasNext(); ) {
            ExtractionUrnCounts euc = (ExtractionUrnCounts) it.next();
            for (int i = 0; i < euc.getLeftCounts().length; i++) {
                leftns[i] += euc.getLeftCounts()[i];
                rightns[i] += euc.getRightCounts()[i];
                n += euc.getLeftCounts()[i] + euc.getRightCounts()[i];
            }
            //System.out.println(euc.getRightCounts()[1]);
        }

        int numUrns = precs.length;
        if(numUrns == 2)
            System.out.println("urncounts = " + leftns[0] + " " + rightns[0]);
        if (numUrns == 4)
            System.out.println("urncounts = " + leftns[0] + " " + rightns[0] + " " +
                    leftns[1] + " " + rightns[1]);
        //System.out.println("n = " + n);
        double experror = 1.0001;
        double exptruth = 1.9;
        double c = 300;
        double e = 1000000;
        double lastexptruth = 1.9;
        double lastc = 300;
        for (int iter = 0; iter < MAXITERS; iter++) {
            DoubleMatrix2D hitsAndProbs = new DenseDoubleMatrix2D(urnCounts.size(),
                    2);
            double ones = 0;
            double totalProb = 0.0;
            double totalHits = 0.0;
            Hashtable countsStrToProb = new Hashtable();
            for (int ex = 0; ex < urnCounts.size(); ex++) {
                double theProb = 0.0;
                ExtractionUrnCounts euc = (ExtractionUrnCounts) urnCounts.get(ex);

                int sumhits = euc.sumCounts();
                if (sumhits == 1) ones++;

                if (numUrns == 1) {
                    if (countsStrToProb.contains(euc.toString())) {
                        theProb = ( (Double) countsStrToProb.get(euc.toString())).
                                doubleValue();
                    } else {
                        theProb = oneUrnProbability(euc.getLeftCounts()[0],
                                exptruth, experror, c, e, precs[0], leftns[0]);
                        countsStrToProb.put(euc.toString(), new Double(theProb));
                    }
                }
                hitsAndProbs.setQuick(ex, 0, sumhits);
                hitsAndProbs.setQuick(ex, 1, theProb);
//        if (iter == MAXITERS - 1)
//          bw.write(euc.getArgsXml() + "\t" + euc.toString() + "\t" + theProb + "\r\n");
                totalHits += sumhits*theProb;
                totalProb += theProb;
            }
            //learn new params:
            hitsAndProbs = (DoubleMatrix2D) cern.
                    colt.matrix.doublealgo.Sorting.
                    quickSort.sort(hitsAndProbs, 0);
            hitsAndProbs = (DoubleMatrix2D) hitsAndProbs.viewRowFlip();
            Matrix m = combine(hitsAndProbs);
            FastVector attr = new FastVector();
            attr.addElement(new Attribute("rank"));
            attr.addElement(new Attribute("frequency"));
            Instances is = new Instances("hitToProb", attr, m.numRows());
            is.setClassIndex(is.numAttributes() - 1);
            for (int i = 0; i < m.numRows(); i++)
            {
                if(m.getElement(i, 1) > 0)
                    is.add(new DenseInstance(1.0, new double[] {Math.log(i + 1),
                            Math.log(m.getElement(i, 1) + 0.001)}));
            }
            if(is.numInstances()!=0)
            {
                LinearRegression lr = new LinearRegression();
                lr.buildClassifier(is);
                double slope = lr.coefficients()[0];
                double intercept = lr.coefficients()[2];
                //do Good-Turing estimation for unseen elements:
                double tarea = 0;
//      tarea = totalHits;
                for (int i = 1; i < totalProb; i++) {
                    tarea +=
                            (Math.exp(intercept) * Math.pow(i, slope) +
                                    Math.exp(intercept) * Math.pow(i + 1, slope)) / 2.0;
                }
                //System.out.println("tarea " + tarea);
                ones = ones *
                        oneUrnProbability(1, exptruth, experror, c, e, precs[0], n);
                //System.out.println("ones " + ones);
                double toadd = tarea * (ones / (precs[0] * (double) n)) /
                        (1.0 - ones / (precs[0] * (double) n));
                //System.out.println("toadd " + toadd);
                lastc = c;
                lastexptruth = exptruth;
                c = totalProb;
//      System.out.println("c before " + c);
                exptruth = -slope;
                //    System.out.println("exptruth " + exptruth);
//Add in number of 'unseen elements'.
//Approximate probability of unseen elements with probability
//of last seen element:
//(implementation note: this simple method gave a good match between expected
// and actual number of distinct elements seen in one experiment.  A better
// method for future work would involve fitting the unseen
// element prob. so the expected number of uniques is equal to actual).
                for (double add = 0; add < toadd && c < e;
                     add += (Math.exp(intercept) * Math.pow(totalProb, slope))) c++;
            }
            else {
                lastc = c;
                lastexptruth = exptruth;
                c = 2.0;
                exptruth = 0.999;
            }
//           add += (Math.exp(intercept) * Math.pow(c, slope) +
//                   Math.exp(intercept) * Math.pow(c + 1, slope)) / 2.0) c++;
            if (prior != null) {
                double priorStrength = 200.0/(double)n;
                //priorStrength = 1.0;
                c = (c + lastc + 2*priorStrength * prior.c) / (2 * priorStrength + 2.0);
                exptruth = (exptruth + lastexptruth +
                        2 * priorStrength * prior.exptruth) / (2 * priorStrength + 2.0);
            }
            //avoid zero exptruth on particularly bad input (for TextRunner):
            if(exptruth < 0.1) exptruth = 0.1;
            //Ensure that function is increasing with repetition:
            if (exptruth < 1) experror = exptruth - 0.0001;
            else experror = 1.0001;
            //  Comment out
            //System.out.println("c = " + c + " exptruth = " + exptruth);
//      System.out.println("1: " +
//                         oneUrnProbability(1, exptruth, experror, c, e, p, n));
//      System.out.println("10: " +
//                         oneUrnProbability(10, exptruth, experror, c, e, p, n));
        }
//    bw.close();
        UrnParameters up = new UrnParameters();
        // Comment out
        //System.out.println("c = " + c + " exptruth = " + exptruth);
        up.c = c;
        up.exptruth = exptruth;
        up.urnTotals = new ExtractionUrnCounts("totals", leftns, rightns);
        up.e = 1000000;
        return up;
    }




    /**
     * Returns the expected number of correct extractions at precision 0.9.
     * @param up UrnParameters
     * @return double
     */
    private static double expectedRecall(UrnParameters up, double n)
    {
        double e = 1000000;
        double p = 0.90;
        double experror = Math.min(up.exptruth, 1.0001);
        double prec = 0.0;
        double precTarget = 0.9;
        final int STARTMAX = 100000;
        int maxThresh = STARTMAX;
        int minThresh = 1;
        int thresh = 1;
        boolean done = false;
        while(!done)
        {
            System.out.println("trying " + thresh);
            prec = precisionAtThresh(up.exptruth,
                    experror, up.c, e, p, n, thresh);
            System.out.println("Got " + prec);
            if(prec < precTarget)
            {
                minThresh = thresh;
                if (maxThresh == STARTMAX)
                    thresh += (int) Math.min(100, 1 + n/10);
                else
                    thresh = thresh + (maxThresh - thresh) / 2;
            }
            else
            {
                maxThresh = thresh;
                thresh = minThresh + (thresh - minThresh) / 2;
            }
            done = (maxThresh - minThresh <= 1);
            if(thresh > n) return -1;
        }
        thresh = maxThresh;
        if(thresh == 1) return recallAtThresh(up.exptruth, up.c, p, n, thresh);
        double precbefore = precisionAtThresh(up.exptruth,
                experror, up.c, e, p, n, thresh - 1);
        double recbefore = recallAtThresh(up.exptruth, up.c, p, n, thresh - 1);
        double rec = recallAtThresh(up.exptruth, up.c, p, n, thresh);
        System.out.println("0.5 thresh: " + thresh + " prec = " + prec + " rec = " +
                rec + " precbefore = " + precbefore + " recbefore = " +
                recbefore);
        return (recbefore - rec)*(0.9 - precbefore)/(prec - precbefore) + rec;
    }

    private static double recallAtThresh(double exp, double size, double p, double n, double thresh)
    {
        double sum = 0.0;
        double s = 0.0;
        for(int i=1; i<=size; i++)
        {
            s += (1/Math.pow(i, exp));
        }
        if(n > 200)
        {
            for (int i = 1; i <= size; i++) {
                sum +=
                        Gamma.incompleteGammaComplement(thresh,
                                (1 / Math.pow(i, exp) * n * p / s));
//      if(i % 10000 == 0)
//        System.out.println("Partial sum " + i + " " + sum);
            }
        }
        else
            for (int i = 1; i <= size; i++)
                for (int j = 0; j < thresh; j++)
                    sum +=
                            Math.pow(p * (1/Math.pow(i, exp)) / s, j) *
                                    Math.pow(1 - p * (1/Math.pow(i, exp)) / s, n - j);

        return size - sum;
    }


    private static double precisionAtThresh(double exptruth, double experror,
                                            double c, double e, double p,
                                            double n, double thresh)
    {
        return 1.0/(1 +
                (recallAtThresh(experror, e, 1 - p, n, thresh) /
                        recallAtThresh(exptruth, c, p, n, thresh)));
    }

    /**
     * Given a two-d matrix, where each row is a (count, probability)
     * pair, round these counts into an expected matrix where
     * each row is a (rank, count).  In the output matrix, the
     * 'rank' elements are increasing from 1 to the number of rows.
     * @param d DoubleMatrix2D
     * @return Matrix
     * @throws Exception
     */
    private static Matrix combine(DoubleMatrix2D d) throws Exception
    {
        double pTotal = 0.00001;
        for (int i = 0; i < d.rows(); i++) {
            pTotal += d.get(i, 1);
        }
        int numcol = Math.max(1, (int) Math.ceil(pTotal));
        double combinedD[][] = new double[numcol][2];
        double wtot = 0.0;
        double itemavg = 0.0;
        int outrow = 0;
        for (int i = 0; i < d.rows(); i++) {
            if (wtot + d.get(i, 1) >= 1) {
                itemavg += (1 - wtot) * d.get(i, 0);
                combinedD[outrow][0] = outrow+1;
                combinedD[outrow++][1] = itemavg;
                wtot = wtot + d.get(i, 1) - 1;
                itemavg = d.get(i, 0) * wtot;
            } else {
                wtot += d.get(i, 1);
                itemavg += d.get(i, 0) * d.get(i, 1);
            }
        }
        //make sure something is in combinedD:
        if(outrow == 0)
        {
            combinedD[outrow][0] = outrow+1;
            combinedD[outrow++][1] = itemavg;
        }
        Matrix m = new Matrix(combinedD);
        return m;
    }

    /**
     * Return the probability for a single urn given the parameters.
     * @param k1 int   extraction
     * @param exptruth double   zC
     * @param experror double   zE
     * @param c double lenC
     * @param e double lenE
     * @param p1 double precision
     * @param n1 double total counts
     * @return double
     */
    public static double oneUrnProbability(int k1, double exptruth, double experror,
                                           double c, double e, double p1, double n1)
    {
        BigDouble tProb = setProbRule(k1, exptruth, c, p1, n1);
        BigDouble fProb = setProbRule(k1, experror, e, 1-p1, n1);
        return toProb(tProb, fProb);
    }




    /**
     * Return a probability for a/b, with error checking for numeric issues.
     * @param a BigDouble
     * @param b BigDouble
     * @return double
     */
    public static double toProb(BigDouble a, BigDouble b)
    {
        if(b.isZero())
            return 1.0;
        BigDouble ans = a.divide(a.plus(b));
        return Math.min(1, Math.max(0, ans.toDouble()));
//    if(Double.isNaN(a)||Double.isInfinite(a)) return 1.0;
//    if(Double.isNaN(b)||Double.isInfinite(b)) return 1.0;
//    if(b == 0.0) return 1.0;
//    return a/(a+b);
    }


    //Helper methods for probability calculations:
    private static BigDouble setProbRule(int k1, double exp, double size, double p1,
                                         double n1)
    {
        double sizeexp = Math.pow(size, exp);
        double temp = (sizeexp - size);
        BigDouble r = new BigDouble(1.0);
        r.timesEquals(BigDouble.pow( (1 - Math.pow(size, 1.0 - exp)) /
                (exp * n1 * p1 - n1 * p1), k1 - 1.0 / exp));
        r.timesEquals(BigDouble.pow( (exp - 1) * n1 * p1 * sizeexp / temp, k1));
        r.timesEquals(BigDouble.upperGamma(k1 - 1.0 / exp, (exp - 1) * n1 * p1 / temp).minus(
                BigDouble.upperGamma(k1 - 1.0 / exp, (exp - 1) * n1 * p1 * sizeexp / temp)));
        r.divideEquals(BigDouble.factorial(k1).times(exp));
        return r;
    }

    public static void main(String args[]) throws Exception
    {
        //The following example shows how to use the main method for Urns,
        //probabilitiesForCounts(), in the single-urn case.
        int[][] extractionCounts = new int[1000][1];
        double [] precisions = new double[] {0.9};
        for(int i=0;i<extractionCounts.length; i++) {
            extractionCounts[i][0] = Math.max(1, (int)Math.round(1000.0/(double)i + Math.random()*20 - 15));
        }

        double [] probEstimates = probabilitiesForCounts(extractionCounts, precisions);
        for(int i=0; i<probEstimates.length; i++)
            System.out.println(extractionCounts[i][0] + "\t" + probEstimates[i]);
        if(true)
            return;
    }
}
