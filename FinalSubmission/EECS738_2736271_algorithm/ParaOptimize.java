import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.meta.MultiSearch;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.FastVector;
 
public class ParaOptimize {
	
	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;
 
		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}
 
		return inputReader;
	}

	public static Evaluation classify(Classifier model,
			Instances trainingSet, Instances testingSet) throws Exception {
		
		Evaluation evaluation = new Evaluation(trainingSet);
 
		model.buildClassifier(trainingSet);
		evaluation.evaluateModel(model, testingSet);
 
		return evaluation;
	}

	public static double calculateAccuracy(FastVector predictions) {
		double correct = 0;
 
		for (int i = 0; i < predictions.size(); i++) {
			NominalPrediction np = (NominalPrediction) predictions.elementAt(i);
			if (np.predicted() == np.actual()) {
				correct++;
			}
		}

		return 100 * correct / predictions.size();
	}
 
	public static Instances[][] crossValidationSplit(Instances data, int numberOfFolds) {
		Instances[][] split = new Instances[2][numberOfFolds];
 
		for (int i = 0; i < numberOfFolds; i++) {
			split[0][i] = data.trainCV(numberOfFolds, i);
			split[1][i] = data.testCV(numberOfFolds, i);
		}
 
		return split;
	}
 
	public static void main(String[] args) throws Exception {

		CSVLoader loader_train = new CSVLoader();
		loader_train.setSource(new File(
				"/Users/syan/Dropbox/Study/2015Fall/EECS 738/Final Project/EECS738_Train_Yes_No.csv"));
		
		CSVLoader loader_test = new CSVLoader();
		 loader_test.setSource(new File(
				 "/Users/syan/Dropbox/Study/2015Fall/EECS 738/Final Project/EECS738_Test.csv"));
		
		// Have weka process each buffer as a "data instance"
		Instances data_train = loader_train.getDataSet();
		Instances data_test = loader_test.getDataSet();

		// Tell weka that the second column contains the class labels
		data_train.setClassIndex(1);
		data_test.setClassIndex(1);
 
		// Do 10-split cross validation
		Instances[][] split = crossValidationSplit(data_train, 10);
 
		// Separate split into training and testing arrays
		Instances[] trainingSplits = split[0];
		Instances[] testingSplits = split[1];
 
		MultiSearch multisearch = new MultiSearch();
		
		// Use a set of classifiers
		Classifier[] models = {
			multisearch
		};

		String[][] parameters = {
			
			{	// Logistic
				"-E", "ACC", 
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.maxIts -min -1 -max 12.0 -step 1.0 -base 10.0 -expression I",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.ridge -min -10.0 -max 10.0 -step 1.0 -base 10.0 -expression pow(BASE,I)",
				"-sample-size", "100.0", 
				"-log-file", "/Users/syan/Documents/Logistic.txt", 
				"-initial-folds", "10", 
				"-subsequent-folds", "10", 
				"-num-slots", "8", 
				"-S", "1", 
				"-W", "weka.classifiers.functions.Logistic", "--", "-R", "1.0E-6", "-M", "-1"
			},
			
			{	// MultilayerPerceptron
				"-E", "ACC", 
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.learningRate -min 0.1 -max 1.0 -step 0.1 -base 10.0 -expression I", 
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.momentum -min 0.1 -max 1.0 -step 0.1 -base 10.0 -expression I", 
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.validationThreshold -min 10 -max 30 -step 2 -base 10.0 -expression I", 
				"-search", "weka.core.setupgenerator.ListParameter -property classifier.normalizeAttributes -list \"True False\"", 
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.trainingTime -min 200.0 -max 900.0 -step 100.0 -base 10.0 -expression I", 
				"-sample-size", "100.0", 
				"-log-file", "/Users/syan/Documents/MultilayerPerceptron.txt", 
				"-initial-folds", "10", 
				"-subsequent-folds", "10", 
				"-num-slots", "8", 
				"-S", "1", 
				"-W", "weka.classifiers.functions.MultilayerPerceptron", "--", "-L", "0.3", "-M", "0.2", "-N", "500", "-V", "10", "-S", "1", "-E", "20", "-H", "a", "-B", "-C", "-do-not-check-capabilities"
			},

			{	// RandomForest
				"-E", "ACC",
				"-search", "weka.core.setupgenerator.ListParameter -property classifier.breakTiesRandomly -list \"True False\"", 
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.numTrees -min 1.0 -max 301.0 -step 50.0 -base 10.0 -expression I",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.maxDepth -min 0.0 -max 100.0 -step 20.0 -base 10.0 -expression I", 
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.numFeatures -min 0.0 -max 50.0 -step 5.0 -base 10.0 -expression I",
				"-sample-size", "100.0", 
				"-log-file", "/Users/syan/Documents/RandomForest.txt", 
				"-initial-folds", "10", 
				"-subsequent-folds", "10", 
				"-num-slots", "8", 
				"-S", "1",
				"-W", "weka.classifiers.trees.RandomForest", "--", "-I", "100", "-K", "0", "-S", "1", "-num-slots", "1", "-do-not-check-capabilities"
			},

			{	// LMT
				"-E", "ACC",
				"-search", "weka.core.setupgenerator.ListParameter -property classifier.doNotMakeSplitPointActualValue -list \"True False\"",
				"-search", "weka.core.setupgenerator.ListParameter -property classifier.errorOnProbabilities -list \"True False\"",
				"-search", "weka.core.setupgenerator.ListParameter -property classifier.useAIC -list \"True False\"",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.minNumInstances -min 6 -max 30 -step 3 -base 10.0 -expression I",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.numBoostingIterations -min -10 -max 10 -step 2 -base 10.0 -expression I", 
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.numFeatures -min 0.0 -max 50.0 -step 5.0 -base 10.0 -expression I",
				"-sample-size", "100.0", 
				"-log-file", "/Users/syan/Documents/LMT.txt", 
				"-initial-folds", "10", 
				"-subsequent-folds", "10", 
				"-num-slots", "8", 
				"-S", "1",
				"-W", "weka.classifiers.trees.LMT", "--", "-C", "-I", "-1", "-M", "15", "-W", "0.0", "-do-not-check-capabilities"
			},

			{	// RBFNetwork
				"-E", "ACC", 
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.maxIts -min -1 -max 9.0 -step 1.0 -base 10.0 -expression I",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.ridge -min -8.0 -max 4 -step 2.0 -base 10.0 -expression pow(BASE,I)",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.minStdDev -min 0.1 -max 0.5 -step 0.1 -base 10.0 -expression I", 
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.numClusters -min 2 -max 10.0 -step 2 -base 10.0 -expression I",
				"-sample-size", "100.0", 
				"-log-file", "/Users/syan/Documents/RBFNetwork.txt", 
				"-initial-folds", "10", 
				"-subsequent-folds", "10", 
				"-num-slots", "8", 
				"-S", "1",
				"-W", "weka.classifiers.functions.RBFNetwork", "--", "-B", "2", "-S", "1", "-R", "1.0E-8", "-M", "-1", "-W", "0.1", "-do-not-check-capabilities"
			},
		};
		
		System.out.println("Testing...");
		multisearch.setOptions(parameters[5]);
		multisearch.buildClassifier(data_train);
		System.out.println("Done");
	}
}