package urns;

import java.io.*;
import java.util.*;
import java.util.logging.*;
import java.util.regex.*;

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
 *
 * @author ddowney
 * @version 1.0
 */


public class AssignUrnProbabilities
{


  private Logger _logger;
  private List _input_predicates = null;
  private static int MAXITERS = 10; //number of EM iterations to perform
  private static double RULEPREC = 0.95; //default precision of rules
  private static boolean DODBUPDATE = false; //do we update probabilities in the DB?
  private static double SYSFACTOR = 0.9; //the fraction of errors appearing in all urns.
  private static boolean COMPUTEPRIORFORBINARIES = false; //whether to compute a
  //average of the parameters for all binary predicate bindings to serve as a prior
  //for the predicate (helps with sparse binary predicates).
  private static double LABELVSLEFTRIGHT = 0.5; //the fraction of errors appearing
  //for left-hand rules.
  private static final boolean READFROMFILE = true;
  public static UrnParameters _upLast; //set to most recently learned urnparameters




  private ArrayList getCountsFromFile(File infile) throws Exception
  {
    BufferedReader brIn = new BufferedReader(new FileReader(infile));
    String sLine = brIn.readLine();
    int numurns = Integer.parseInt(sLine.split(" ")[1]);
    ArrayList alOut = new ArrayList();
    while((sLine = brIn.readLine())!= null)
    {
      int[] leftcounts = new int[ (numurns + 1) / 2];
      int[] rightcounts = new int[ (numurns + 1) / 2];
      String [] fields = sLine.split("\t");
      String [] argsfields = fields[fields.length -1].split("\\|");
      String argsxml = "<v>" + argsfields[0].toLowerCase() + "</v><v>" + argsfields[1].toLowerCase() + "</v>";
      if(numurns==1) {leftcounts[0] = Integer.parseInt(fields[0]);
      }
      else
      {
        for(int i=0;i<numurns/2;i++){
          leftcounts[i] = Integer.parseInt(fields[i*2]);
          rightcounts[i] = Integer.parseInt(fields[i*2 + 1]);
        }
      }
      ExtractionUrnCounts euc = new ExtractionUrnCounts(argsxml, leftcounts, rightcounts);
      alOut.add(euc);
    }
    brIn.close();
    return alOut;
  }

  /***
   * main run methods
   */



  /**
   * Returns the 'binding' in the given Extraction args_xml string,
   * assuming the first argument is bound.
   * @param argsXml String
   * @return String
   */
  public String getBinding(String argsXml)
  {
    Pattern p = Pattern.compile("(<v>.*</v>)<v>.*</v>");
    Matcher m = p.matcher(argsXml);
    if(m.find())
      return m.group(1);
    return "";
  }

  public static double computeLogLikelihood(int[] counts,
                                            double[] probabilities,
                                            double [] weights,
                                            double expe, double expc, double c,
                                            double e, double p, double n)
  {
    Hashtable<Integer, Double> countToPosProb = new Hashtable<Integer, Double> ();
    Hashtable<Integer, Double> countToNegProb = new Hashtable<Integer, Double> ();

    double ll = 0;
    for (int i = 0; i < counts.length; i++) {
      int count = counts[i];
      double posprob = 0;
      double negprob = 0;
      if (countToPosProb.containsKey(count)) {
        posprob = countToPosProb.get(count);
//        negprob = countToNegProb.get(count);
      } else {
        posprob = oneUrnProbability(count, expc, expe, c, e, p, n);
        countToPosProb.put(count, posprob);
//        posprob = BigDouble.lnSmall(setProbRule(count, expc, c, p, n));
//        if(Double.isNaN(posprob) || posprob == 0.0) {
//          return Double.NEGATIVE_INFINITY;
//        }
//        negprob = BigDouble.lnSmall(setProbRule(count, expe, e, (1-p), n));
//        if(Double.isNaN(negprob) || negprob == 0.0) {
//          return Double.NEGATIVE_INFINITY;
//        }
//        countToPosProb.put(count, posprob);
//        countToNegProb.put(count, negprob);
      }
      posprob = (posprob + 0.000001)/(1.000002);
      ll += weights[i]*(Math.log(posprob) * (probabilities[i]) +
                   Math.log(1.0 - posprob) * (1.0 - probabilities[i]));
    }
    return ll;
  }

  public static double [] probabilitiesForCountsAndParams(int [] counts,
      double expc, double expe, double c, double e,
      double p, double n) {
    Hashtable<Integer, Double> countToProb = new Hashtable<Integer, Double> ();
    double[] out = new double[counts.length];
    for (int i = 0; i < counts.length; i++) {
      int count = counts[i];
      double prob = 0;
      if (countToProb.containsKey(count)) {
        prob = countToProb.get(count);
      } else {
        prob = oneUrnProbability(count, expc, expe, c, e, p, n);
        countToProb.put(count, prob);
      }
      out[i] = prob;
    }
    return out;
  }

  //learns parameters for a given set of (single-urn) counts, and returns
  //probabilities for all the counts.
  //uses a lambda of 0.5 (even weight on tagged, untagged egs.
  public static double[] learnWithEM(int [] counts, int totalObjects,
                                     double [] probs, int taggedDataBegins, double p) {
    double startC = Math.min(totalObjects/2, 300);
    double n = 0;
    for (int i = 0; i < counts.length; i++)
      n += counts[i];
    double[] newprobs = probabilitiesForCountsAndParams(counts, 1.9, 1.01, startC,
        totalObjects - startC,
        0.9, n);
    //set weights for lambda = 0.5:
    double[] weights = new double[newprobs.length];
    for (int i = 0; i < weights.length; i++)
      if (i >= taggedDataBegins)
        weights[i] = 1.0;
      else
        weights[i] = 0.0; //no untagged data first time around.
    for(int i=0; i<3; i++) {
      for(int j=0;j<taggedDataBegins;j++) { //update the untagged example probs:
        probs[j] = newprobs[j];
      }
      newprobs = hillClimb(counts, probs, weights, totalObjects, p);
      for (int j = 0; j < weights.length; j++)
        if (j < taggedDataBegins)
          weights[j] = ( (double) (weights.length - taggedDataBegins)) /
              (double) taggedDataBegins;
    }
    return probs;
  }

  //finds parameters for urn via hillclimbing.  Only works for single urn.
  //counts holds the observation count, probabilities holds the estimated
  //probability of truth.
  //returns probabilities for all the counts.
  public static double[] hillClimb(int [] counts, double[] probabilities,
                                   double [] weights, int totalObjects, double p) {
    //right now just does grid search.
    double bestLL = Double.NEGATIVE_INFINITY;
    double bestExpe = -1;
    double bestExpc = -1;
    double bestP = -1;
    double bestC = -1;
    double GRADSTEP = 0.0001;
    double MAXSTEP = 0.1;
    double n=0;
    for(int i=0;i<counts.length;i++)
      n += counts[i];

    for(double expe=0.8;expe<=0.81;expe+=0.1) {
      for(double expc=1.1;expc<=1.1;expc+=0.1) {
        for(double c=10;c<(double)totalObjects/2.0;c+=((double)totalObjects)/10.0) {
            double ll = computeLogLikelihood(counts, probabilities, weights, expe,
                                             expc, c, (double)totalObjects - c, p, n);
            if(ll > bestLL) {
              System.out.println("New best: " + expe + "\t" + expc + "\t" + c + "\t" + p + "\t" + ll);
              bestLL = ll;
              bestExpe = expe;
              bestExpc = expc;
              bestP = p;
              bestC = c;
            }
            //System.out.println(expe + "\t" + expc + "\t" + c + "\t" + p + "\t" + ll);
        }
      }
    }
    for(int i=0;i<10;i++) {
      double ll = computeLogLikelihood(counts, probabilities, weights, bestExpe,
                                             bestExpc, bestC, (double)totalObjects - bestC, bestP, n);
     double deltaexpe = computeLogLikelihood(counts, probabilities, weights, bestExpe + bestExpe*GRADSTEP,
                                             bestExpc, bestC, (double)totalObjects - bestC, bestP, n) - ll;
     double deltaexpc = computeLogLikelihood(counts, probabilities, weights, bestExpe,
                                             bestExpc + bestExpc*GRADSTEP, bestC, (double)totalObjects - bestC, bestP, n) - ll;
     double deltac = computeLogLikelihood(counts, probabilities, weights, bestExpe,
                                             bestExpc, bestC + bestC*GRADSTEP, (double)totalObjects - (bestC + bestC*GRADSTEP), bestP, n) - ll;
     double deltap = computeLogLikelihood(counts, probabilities, weights, bestExpe,
                                             bestExpc, bestC, (double)totalObjects - bestC, bestP + bestP*GRADSTEP, n) - ll;
        deltap = 0.0; //HACK:don't adjust p for now
     //binary line search:
     double topLL = Double.NEGATIVE_INFINITY;
     double bottomLL = ll;
     double s = MAXSTEP;
     double gradLength = Math.sqrt(deltaexpe*deltaexpe + deltaexpc*deltaexpc + deltac*deltac + deltap*deltap);
     s = s/gradLength;
     double midLL = 0.0;
     double newBestLL = bestLL;
     double newBestExpe = bestExpe;
     double newBestExpc = bestExpc;
     double newBestC = bestC;
     double newBestP = bestP;
     for (int j = 0; j < 5; j++) {
       midLL = computeLogLikelihood(counts, probabilities, weights,
                                    bestExpe + bestExpe*deltaexpe * s,
                                    bestExpc + bestExpc*deltaexpc * s,
                                    bestC + bestC*deltac * s,
                                    (double) totalObjects -
                                    (bestC + bestC*deltac * s),
                                    bestP + bestP*deltap * s, n);
       if (midLL > newBestLL) {
         newBestLL = midLL;
         newBestExpe = bestExpe + deltaexpe * s;
         newBestExpc = bestExpc + deltaexpc * s;
         newBestC = bestC + deltac * s;
         newBestP = bestP +deltap * s;
         System.out.println("New best from grad: " + newBestExpe + "\t" + newBestExpc + "\t" + newBestC + "\t" + newBestP + "\t" + newBestLL);
       }
       if (topLL > bottomLL) {
         bottomLL = midLL;
         s += MAXSTEP / (Math.pow(2.0, (double) (j + 1))*gradLength);
       } else {
         topLL = midLL;
         s -= MAXSTEP / (Math.pow(2.0, (double) (j + 1))*gradLength);
       }
     }
     bestLL = newBestLL;
     bestExpe = newBestExpe;
     bestExpc = newBestExpc;
     bestC = newBestC;
     bestP = newBestP;
   }
   System.out.println("bestLL: " + bestLL);
   System.out.println("bestExpe: " + bestExpe);
    System.out.println("bestExpc: " + bestExpc);
    System.out.println("bestP: " + bestP);
    System.out.println("bestC: " + bestC);
    System.out.println("bestE: " + ((double)totalObjects - bestC));
    return probabilitiesForCountsAndParams(counts, bestExpc, bestExpe, bestC,
                                           (double) totalObjects - bestC, bestP,
                                           n);
  }

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
    if (totalObjects > 0)
      up = computeUrnParametersConstrained(urnCounts, prior, precisions, totalObjects, priorStrength);
    else
      up = computeUrnParameters(urnCounts, prior, precisions);
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
      else if (numUrns == 2)
        probs[i] = twoUrnProbability(counts[i][0], counts[i][1], up.exptruth,
                                     Math.min(up.exptruth - 0.0001, 1.0001),
                                     up.c, up.e, precisions[0], precisions[1],
                                     up.urnTotals.getLeftCounts()[0],
                                     up.urnTotals.getRightCounts()[0], SYSFACTOR);
      else if (numUrns == 4)
        probs[i] = fourUrnProbability(counts[i][0], counts[i][1], counts[i][2],
                                      counts[i][3], up.exptruth,
                                      Math.min(up.exptruth - 0.0001, 1.0001),
                                      up.c, up.e, precisions[0],
                                      precisions[1], precisions[2],
                                      precisions[3],
                                      up.urnTotals.getLeftCounts()[0],
                                      up.urnTotals.getLeftCounts()[1],
                                      up.urnTotals.getRightCounts()[0],
                                      up.urnTotals.getRightCounts()[1], LABELVSLEFTRIGHT,
                                      SYSFACTOR);
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
        if (numUrns == 4) {
          if(countsStrToProb.contains(euc.toString()))
          {
            theProb = ((Double)countsStrToProb.get(euc.toString())).doubleValue();
          }
          else{
            theProb = fourUrnProbability(euc.getLeftCounts()[0],
                                              euc.getRightCounts()[0],
                                              euc.getLeftCounts()[1],
                                              euc.getRightCounts()[1],
                                              exptruth, experror, c, e, precs[0], precs[1], precs[2],
                                              precs[3], leftns[0], rightns[0],
                                              leftns[1],
                                              rightns[1], 0.5, SYSFACTOR);
            countsStrToProb.put(euc.toString(), new Double(theProb));
          }
        }
        else if (numUrns == 2) {
          if (countsStrToProb.contains(euc.toString())) {
            theProb = ( (Double) countsStrToProb.get(euc.toString())).
                doubleValue();
          } else {
            theProb = twoUrnProbability(euc.getLeftCounts()[0],
                                              euc.getRightCounts()[0],
                                              exptruth, experror, c, e, precs[0], precs[1], leftns[0], rightns[0], 0.5);
            countsStrToProb.put(euc.toString(), new Double(theProb));
          }
        }
        else {
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


  //same as computeUrnParameters, but requires that |C| + |E| = totalObjects
  private static UrnParameters computeUrnParametersConstrained(List urnCounts,
      UrnParameters prior, double[] precs,
      int totalObjects, double priorStrength) throws Exception
  {
	  int leftns[] = new int[1];
	  int rightns[] = new int[1];
	  if(precs.length==4)
	  {
	    leftns = new int[2];
	    rightns = new int[2];
	  }
    if (urnCounts.size() <= 1) {
      UrnParameters up = new UrnParameters();
      up.c = 100;
      up.exptruth = 1.5;
      if (prior != null)
        return prior;
      else return up;
    }
    int n = 0;

    for (Iterator it = urnCounts.iterator(); it.hasNext(); ) {
      ExtractionUrnCounts euc = (ExtractionUrnCounts) it.next();
      for (int i = 0; i < euc.getLeftCounts().length; i++) {
        leftns[i] += euc.getLeftCounts()[i];
        rightns[i] += euc.getRightCounts()[i];
        n += euc.getLeftCounts()[i] + euc.getRightCounts()[i];
      }
    }
    if (precs == null) {
      precs = new double[leftns.length + rightns.length];
      for (int i = 0; i < precs.length; i++) {
        precs[i] = RULEPREC;
      }
    }
    int numUrns = precs.length;
    double experror = 1.0001;
    double exptruth = 1.9;
    double c = 300;
    double e = totalObjects - c;
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
        if (numUrns == 4) {
          if (countsStrToProb.contains(euc.toString())) {
            theProb = ( (Double) countsStrToProb.get(euc.toString())).
                doubleValue();
          } else {
            theProb = fourUrnProbability(euc.getLeftCounts()[0],
                                         euc.getRightCounts()[0],
                                         euc.getLeftCounts()[1],
                                         euc.getRightCounts()[1],
                                         exptruth, experror, c, e, precs[0],
                                         precs[1], precs[2],
                                         precs[3], leftns[0], rightns[0],
                                         leftns[1],
                                         rightns[1], 0.5, SYSFACTOR);
            countsStrToProb.put(euc.toString(), new Double(theProb));
          }
        } else if (numUrns == 2) {
          if (countsStrToProb.contains(euc.toString())) {
            theProb = ( (Double) countsStrToProb.get(euc.toString())).
                doubleValue();
          } else {
            theProb = twoUrnProbability(euc.getLeftCounts()[0],
                                        euc.getRightCounts()[0],
                                        exptruth, experror, c, e, precs[0],
                                        precs[1], leftns[0], rightns[0], 0.5);
            countsStrToProb.put(euc.toString(), new Double(theProb));
          }
        } else {
          if (countsStrToProb.contains(euc.toString())) {
            theProb = ( (Double) countsStrToProb.get(euc.toString())).
                doubleValue();
          } else {
            theProb = oneUrnProbability(euc.getLeftCounts()[0],
                                        exptruth, experror, c, e, precs[0],
                                        leftns[0]);
            countsStrToProb.put(euc.toString(), new Double(theProb));
          }
        }
        hitsAndProbs.setQuick(ex, 0, sumhits);
        hitsAndProbs.setQuick(ex, 1, theProb);
        //        if (iter == MAXITERS - 1)
        //          bw.write(euc.getArgsXml() + "\t" + euc.toString() + "\t" + theProb + "\r\n");
        totalHits += sumhits * theProb;
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
      for (int i = 0; i < m.numRows(); i++) {
        if (m.getElement(i, 1) > 0)
          is.add(new DenseInstance(1.0, new double[] {Math.log(i + 1),
                              Math.log(m.getElement(i, 1) + 0.001)}));
      }
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
      ones = ones * oneUrnProbability(1, exptruth, experror, c, e, precs[0], n);
      double toadd = tarea * (ones / (precs[0] * (double) n)) /
          (1.0 - ones / (precs[0] * (double) n));
      c = totalProb;
      exptruth = -slope;
      for (double add = 0; add < toadd && c < e;
           add += (Math.exp(intercept) * Math.pow(totalProb, slope))) c++;
      if (prior != null) {
        c = (c + lastc + 2 * priorStrength * prior.c) / (2 * priorStrength + 2.0);
        exptruth = (exptruth + lastexptruth +
                    2 * priorStrength * prior.exptruth) / (2 * priorStrength + 2.0);
      }
      else {
        c = (c + lastc)/2.0;
        exptruth = (exptruth + lastexptruth)/2.0;
      }
      //avoid zero exptruth on particularly bad input (for TextRunner):
      if (exptruth < 0.1) exptruth = 0.1;
      //Ensure that function is increasing with repetition:
      if (exptruth < 1) experror = exptruth - 0.0001;
      else experror = 1.0001;
      //constrain:
      e = totalObjects - c;
      lastc = c;
      lastexptruth = exptruth;
      System.out.println("c = " + c + " exptruth = " + exptruth + " e = " + e);
    }
    UrnParameters up = new UrnParameters();
    System.out.println("c = " + c + " exptruth = " + exptruth + " e = " + e);
    up.c = c;
    up.exptruth = exptruth;
    up.e = e;
    up.urnTotals = new ExtractionUrnCounts("totals", leftns, rightns);
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

  public static double oneUrnProbability(int k1, UrnParameters up, double p1)
  {
    double experror = 1.01;
    if(up.exptruth < experror)
      experror = up.exptruth - 0.01;
    return oneUrnProbability(k1, up.exptruth, experror,
                             up.c, up.e, p1, up.urnTotals.sumCounts());

  }

  /**
   * Return the probability for a single urn given the parameters.
   * @param k1 int
   * @param exptruth double
   * @param experror double
   * @param c double
   * @param e double
   * @param p1 double
   * @param n1 double
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
   * Return the probability for two urns given the parameters.
   * @param k1 int
   * @param k2 int
   * @param exptruth double
   * @param experror double
   * @param c double
   * @param e double
   * @param p1 double
   * @param p2 double
   * @param n1 int
   * @param n2 int
   * @param sysfactor double
   * @return double
   */
  public static double twoUrnProbability(int k1, int k2, double exptruth,
                                         double experror,
                                         double c, double e, double p1,
                                         double p2, int n1, int n2,
                                         double sysfactor)
  {
    BigDouble tProbAll = setProb2Rules(k1, k2, exptruth, c, p1, p2, n1, n2);
    BigDouble fProbAll = setProb2Rules(k1, k2, experror, e, 1 - p1, 1 - p2, n1,
                                       n2);
    BigDouble nonsystematic = new BigDouble(0);
    if (k1 == 0) nonsystematic.plusEquals(
        setProbRule(k2, experror, e, 1 - p2, n2));
    if (k2 == 0) nonsystematic.plusEquals(
        setProbRule(k1, experror, e, 1 - p1, n1));
    return toProb(tProbAll,
                  fProbAll.times(sysfactor).plus(nonsystematic.times(1 -
        sysfactor)));
  }

  /**
   * Return the probability for four urns given the parameters.
   * @param k1 int
   * @param k2 int
   * @param k3 int
   * @param k4 int
   * @param exptruth double
   * @param experror double
   * @param c double
   * @param e double
   * @param p1 double
   * @param p2 double
   * @param p3 double
   * @param p4 double
   * @param n1 int
   * @param n2 int
   * @param n3 int
   * @param n4 int
   * @param labvslr double
   * @param sysfactor double
   * @return double
   */
  public static double fourUrnProbability(int k1, int k2, int k3, int k4,
                                          double exptruth,
                                          double experror, double c, double e,
                                          double p1,
                                          double p2, double p3, double p4,
                                          int n1, int n2, int n3, int n4,
                                          double labvslr,
                                          double sysfactor)
  {
    BigDouble tProbAll = setProb4Rules(k1, k2, k3, k4, exptruth, c, p1, p2, p3, p4,
                                    n1, n2, n3, n4);
    BigDouble fProbAll = setProb4Rules(k1, k2, k3, k4, experror, e, 1 - p1, 1 - p2,
                                    1 - p3, 1 - p4,
                                    n1, n2, n3, n4);
    BigDouble nonsystematic = new BigDouble(0);
    if (k1 + k2 == 0)
      nonsystematic.plusEquals(
          setProb2Rules(k3, k4, experror, e, 1 - p3, 1 - p4, n3, n4).times(labvslr));
    if (k3 + k4 == 0)
      nonsystematic.plusEquals(
          setProb2Rules(k1, k2, experror, e, 1 - p1, 1 - p2, n1, n2).times(labvslr));
    if (k1 + k3 == 0)
      nonsystematic.plusEquals(
          setProb2Rules(k2, k4, experror, e, 1 - p2, 1 - p4, n2, n4).times(1 - labvslr));
    if (k2 + k4 == 0)
      nonsystematic.plusEquals(
          setProb2Rules(k1, k3, experror, e, 1 - p1, 1 - p3, n1, n3).times(1 - labvslr));
    return toProb(tProbAll, fProbAll.times(sysfactor).plus(nonsystematic.times(1 - sysfactor)));
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

  private static BigDouble setProb2Rules(int k1, int k2, double exp, double size, double p1,
                                       double p2, int n1, int n2)
    {
      double sizeexp = Math.pow(size, exp);
      double temp = (sizeexp - size);
      BigDouble r = new BigDouble(1.0);
      r.timesEquals(BigDouble.pow( (exp - 1) * n1 * p1 *sizeexp/ temp, k1));
      r.timesEquals(BigDouble.pow( (exp - 1) * n2 * p2 * sizeexp/temp, k2));
      r.timesEquals(BigDouble.pow(temp/((exp - 1)*(n1*p1+n2*p2)*sizeexp), k1+k2-1.0/exp));
      r.timesEquals(BigDouble.upperGamma(k1+k2-1.0/exp, (exp - 1)*(n1*p1+n2*p2)/temp).minus(
            BigDouble.upperGamma(k1+k2-1.0/exp, (exp - 1)*(n1*p1+n2*p2)*sizeexp/temp)));
      r.divideEquals(BigDouble.factorial(k1).times(BigDouble.factorial(k2)).times(exp));
      return r;
    }

  private static double sizeKOverFactK(double size, double k)
  {
    double r = 1;
    for(int i=0; i<k; i++)
    {
      r *= size/(i+1);
    }
    return r;
  }

  private static BigDouble setProb4Rules(int k1, int k2, int k3, int k4,
                                        double exp,
                                        double size, double p1, double p2,
                                        double p3, double p4, int n1, int n2,
                                        int n3,
                                        int n4)
  {
    double temp = (Math.pow(size, exp) - size);
    BigDouble r; //= BigDouble.pow(size, k1 * exp).divide(BigDouble.factorial(k1));
    r = BigDouble.pow(size, exp * (k1 + k2 + k3 + k4));
    r.timesEquals(BigDouble.pow( ( ( (exp - 1) * n1 * p1) / temp), k1));
    r.timesEquals(BigDouble.pow( ( ( (exp - 1) * n2 * p2) / temp), k2));
    r.timesEquals(BigDouble.pow( ( ( (exp - 1) * n3 * p3) / temp), k3));
    r.timesEquals(BigDouble.pow( ( ( (exp - 1) * n4 * p4) / temp), k4));
    r.timesEquals(BigDouble.pow(temp /
                                ( (exp - 1) *
                                 (n1 * p1 + n2 * p2 + n3 * p3 + n4 * p4) *
                                 Math.pow(size, exp)), k1+k2+k3+k4-1.0/exp));

    BigDouble factor = BigDouble.upperGamma(k1 + k2 + k3 + k4 - 1.0 / exp,
                                      (exp - 1) *
                                      (n1 * p1 + n2 * p2 + n3 * p3 + n4 * p4) /
                                      temp);
  if(!(k1 + k2 +
     k3 + k4 - 1.0 / exp < 2000 && (exp - 1) *
     (n1 * p1 + n2 * p2 + n3 * p3 + n4 * p4) *
     Math.pow(size, exp) / temp > 30000 && factor.expIsLargerThan(-4000)))
  factor.minusEquals(BigDouble.upperGamma(k1 + k2 +
       k3 + k4 - 1.0 / exp,
       (exp - 1) *
       (n1 * p1 + n2 * p2 + n3 * p3 + n4 * p4) *
       Math.pow(size, exp) / temp));
   r.timesEquals(factor);
   r.divideEquals(exp);
   r.divideEquals(BigDouble.factorial(k1));
   r.divideEquals(BigDouble.factorial(k2));
   r.divideEquals(BigDouble.factorial(k3));
   r.divideEquals(BigDouble.factorial(k4));
  return r;
  }

  public static double upperGamma(double a, double x)
  {
    if(a < 0)
      return (upperGamma(a + 1, x) - Math.exp( -x) * Math.pow(x, a)) / a;
    return Gamma.incompleteGammaComplement(a, x) * Gamma.gamma(a);
  }

  //gets single and multi-urn probabilities for an input file of extraction counts
  //outputs the probabilities for the extrs in the input file order
  //the hitsFile includes just the hits from which the urns params are learned.
  //first line in extraction counts file gives ns
  private static void urnRunner(String hitsFile, String inFile, String outFile,
      int numUrns, double labvslr, double sysfactor, boolean combineToOneUrn) throws Exception
  {
    SYSFACTOR = sysfactor;
    LABELVSLEFTRIGHT = labvslr;
    BufferedReader brIn = new BufferedReader(new FileReader(hitsFile));

    String sLine = null;
    ArrayList<Integer> cnts = new ArrayList<Integer> ();
    while ( (sLine = brIn.readLine()) != null) {
      cnts.add(Integer.parseInt(sLine));
    }
    int[][] learningCounts = new int[cnts.size()][1];
    for (int i = 0; i < learningCounts.length; i++)
      learningCounts[i][0] = cnts.get(i);
    probabilitiesForCounts(learningCounts, new double[] {0.9});
    brIn.close();

    brIn = new BufferedReader(new FileReader(inFile));
    //get ns:
    sLine = brIn.readLine();
    String [] sNs = sLine.split(" ");
    int[] ns = new int[sNs.length];
    for(int i=0; i<ns.length; i++)
      ns[i] = Integer.parseInt(sNs[i]);
    ExtractionUrnCounts euc;
    if(numUrns==4) {
      euc = new ExtractionUrnCounts("blank", new int[] {ns[0], ns[1]}, new int[] {ns[2], ns[3]});
    }
    else
      euc = new ExtractionUrnCounts("blank", new int[] {ns[0]}, new int[] {ns[1]});
    System.out.println("n = " + euc.sumCounts());
    _upLast.urnTotals = euc;
    ArrayList<int []> alUrnCounts = new ArrayList<int[]>();
    while((sLine = brIn.readLine())!=null) {
      sLine = sLine.replace("{", "").replace("}", "").replace("., ", ",");
      int [] urnCounts = new int[numUrns];
      String [] fields = sLine.split(",");
      for(int i=0; i<numUrns; i++) {
        urnCounts[i] = Integer.parseInt(fields[i]);
      }
      alUrnCounts.add(urnCounts);
    }
    brIn.close();
    int [][] aCounts = new int[alUrnCounts.size()][numUrns];
    for (int i = 0; i < aCounts.length; i++) {
      aCounts[i] = alUrnCounts.get(i);
    }
    double [] probs = new double[aCounts.length];
    if(combineToOneUrn) {
      int [][] combCounts = new int[aCounts.length][1];
      for(int i=0; i<combCounts.length; i++) {
        for (int j = 0; j < aCounts[i].length; j++)
          combCounts[i][0] += aCounts[i][j];
        probs[i] = oneUrnProbability(combCounts[i][0], _upLast, 0.9);
      }
    }
    else {
      for(int i=0; i<aCounts.length; i++) {
        if(numUrns==4)
        probs[i] = fourUrnProbability(aCounts[i][0], aCounts[i][1], aCounts[i][2],
                                      aCounts[i][3], _upLast.exptruth,
                                      Math.min(_upLast.exptruth - 0.0001, 1.0001),
                                      _upLast.c, _upLast.e, 0.9, 0.9, 0.9, 0.9,
                                      _upLast.urnTotals.getLeftCounts()[0],
                                      _upLast.urnTotals.getLeftCounts()[1],
                                      _upLast.urnTotals.getRightCounts()[0],
                                      _upLast.urnTotals.getRightCounts()[1], LABELVSLEFTRIGHT,
                                      SYSFACTOR);
      else if(numUrns==2)
        probs[i] = twoUrnProbability(aCounts[i][0], aCounts[i][1], _upLast.exptruth,
                             Math.min(_upLast.exptruth - 0.0001, 1.0001),
                             _upLast.c, _upLast.e, 0.9, 0.9,
                             _upLast.urnTotals.getLeftCounts()[0],
                             _upLast.urnTotals.getRightCounts()[0], SYSFACTOR);

      }
    }
    BufferedWriter bwOut = new BufferedWriter(new FileWriter(outFile));
    for(int i=0; i<probs.length; i++)
      bwOut.write(probs[i] + "\r\n");
    bwOut.close();


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
