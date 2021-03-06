
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.BufferedWriter;
import java.util.Random;
import java.util.Arrays;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.evaluation.NumericPrediction;

import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.NaiveBayesMultinomial;

import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SGD;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.VotedPerceptron;

import weka.classifiers.meta.AttributeSelectedClassifier;

import weka.classifiers.meta.AttributeSelectedClassifier;
import weka.classifiers.meta.CVParameterSelection;
import weka.classifiers.meta.MultiSearch;
import weka.classifiers.functions.RBFNetwork;
import weka.classifiers.misc.InputMappedClassifier;
import weka.classifiers.misc.SerializedClassifier;

import weka.classifiers.rules.DecisionTable;
import weka.classifiers.rules.JRip;
import weka.classifiers.rules.OneR;
import weka.classifiers.rules.PART;
import weka.classifiers.rules.ZeroR;

import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.HoeffdingTree;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.LMT;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.RandomTree;
import weka.classifiers.trees.REPTree;

import weka.attributeSelection.ASEvaluation.*;

import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.NumericToBinary;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.core.SerializationHelper;
import weka.classifiers.evaluation.output.prediction.CSV;
import weka.classifiers.evaluation.output.prediction.PlainText;
import weka.core.FastVector;
 
public class RBFWrapBF {

	public static void main(String[] args) throws Exception {
		
		CSVLoader loader_train = new CSVLoader();
		loader_train.setSource(new File("EECS738_Train.csv"));

		Instances data_train = loader_train.getDataSet();

		// Tell weka that the second column contains the class labels
		data_train.setClassIndex(0);
		
		// Normalize features
		Normalize m_Normalize = new Normalize();
		m_Normalize.setInputFormat(data_train);
		m_Normalize.setScale(100.0);
		data_train = Filter.useFilter(data_train, m_Normalize);
		
		MultiSearch multisearch = new MultiSearch();

		String[][] parameters = {

			{	// 1. Logistic + WrapperSubsetEval + BestFirst
				"-E", "ACC", 
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.maxIts -min -1 -max 12.0 -step 1.0 -base 10.0 -expression I",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.ridge -min -10.0 -max 10.0 -step 1.0 -base 10.0 -expression pow(BASE,I)",
				"-sample-size", "100.0", 
				"-log-file", "/projects/huanlab/wei/results/Logistic_WrapperSubsetEval_BestFirst.txt", 
				"-initial-folds", "10", 
				"-subsequent-folds", "10", 
				"-num-slots", "8", 
				"-S", "1", 
				"-W", "weka.classifiers.meta.AttributeSelectedClassifier", "-do-not-check-capabilities", "--", 
				"-E", "weka.attributeSelection.WrapperSubsetEval -B weka.classifiers.functions.Logistic -F 10 -T 0.01 -R 1 -E ACC -- -R 0.001 -M -1 -do-not-check-capabilities",  
				"-S", "weka.attributeSelection.BestFirst -D 2 -N 6",
				"-W", "weka.classifiers.functions.Logistic", "-do-not-check-capabilities", "--", "-R", "1.0E-3", "-M", "-1"
			},
			
			{	// 2. Logistic + WrapperSubsetEval + GreedyStepwise
				"-E", "ACC", 
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.maxIts -min -1 -max 12.0 -step 1.0 -base 10.0 -expression I",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.ridge -min -10.0 -max 10.0 -step 1.0 -base 10.0 -expression pow(BASE,I)",
				"-sample-size", "100.0", 
				"-log-file", "/projects/huanlab/wei/results/Logistic_WrapperSubsetEval_GreedyStepwise.txt", 
				"-initial-folds", "10", 
				"-subsequent-folds", "10", 
				"-num-slots", "8", 
				"-S", "1", 
				"-W", "weka.classifiers.meta.AttributeSelectedClassifier", "-do-not-check-capabilities", "--", 
				"-E", "weka.attributeSelection.WrapperSubsetEval -B weka.classifiers.functions.Logistic -F 10 -T 0.01 -R 1 -E ACC -- -R 0.001 -M -1 -do-not-check-capabilities",  
				"-S", "weka.attributeSelection.GreedyStepwise -T -1.7976931348623157E308 -N -1 -num-slots 1",
				"-W", "weka.classifiers.functions.Logistic", "-do-not-check-capabilities", "--", "-R", "1.0E-3", "-M", "-1"
			},
			
			{	// 3. RandomForest + WrapperSubsetEval + BestFirst
				"-E", "ACC", 
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.numTrees -min 100.0 -max 300.0 -step 50.0 -base 10.0 -expression I",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.maxDepth -min 0.0 -max 100.0 -step 20.0 -base 10.0 -expression I", 
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.numFeatures -min 10.0 -max 40.0 -step 5.0 -base 10.0 -expression I",
				"-sample-size", "100.0", 
				"-log-file", "/projects/huanlab/wei/results/RandomForest_WrapperSubsetEval_BestFirst.txt", 
				"-initial-folds", "10", 
				"-subsequent-folds", "10", 
				"-num-slots", "8", 
				"-S", "1", 
				"-W", "weka.classifiers.meta.AttributeSelectedClassifier", "-do-not-check-capabilities", "--", 
				"-E", "weka.attributeSelection.WrapperSubsetEval -B weka.classifiers.trees.RandomForest -F 10 -T 0.01 -R 1 -E ACC -- -I 200 -K 14 -S 1 -num-slots 1 -do-not-check-capabilities",  
				"-S", "weka.attributeSelection.BestFirst -D 2 -N 6",
				"-W", "weka.classifiers.trees.RandomForest", "--", "-I", "200", "-K", "14", "-S", "1", "-num-slots", "1", "-do-not-check-capabilities"
			},
			
			{	// 4. RandomForest + WrapperSubsetEval + GreedyStepwise
				"-E", "ACC", 
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.numTrees -min 100.0 -max 300.0 -step 50.0 -base 10.0 -expression I",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.maxDepth -min 0.0 -max 100.0 -step 20.0 -base 10.0 -expression I", 
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.numFeatures -min 10.0 -max 40.0 -step 5.0 -base 10.0 -expression I",
				"-sample-size", "100.0", 
				"-log-file", "/projects/huanlab/wei/results/RandomForest_WrapperSubsetEval_GreedyStepwise.txt", 
				"-initial-folds", "10", 
				"-subsequent-folds", "10", 
				"-num-slots", "8", 
				"-S", "1", 
				"-W", "weka.classifiers.meta.AttributeSelectedClassifier", "-do-not-check-capabilities", "--", 
				"-E", "weka.attributeSelection.WrapperSubsetEval -B weka.classifiers.trees.RandomForest -F 10 -T 0.01 -R 1 -E ACC -- -I 200 -K 14 -S 1 -num-slots 1 -do-not-check-capabilities",  
				"-S", "weka.attributeSelection.GreedyStepwise -T -1.7976931348623157E308 -N -1 -num-slots 1",
				"-W", "weka.classifiers.trees.RandomForest", "--", "-I", "200", "-K", "14", "-S", "1", "-num-slots", "1", "-do-not-check-capabilities"
			},

			{	// 5. RBFNetwork + WrapperSubsetEval + BestFirst
				"-E", "ACC", 
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.ridge -min -4.0 -max 4.0 -step 2.0 -base 10.0 -expression pow(BASE,I)",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.minStdDev -min 0.4 -max 0.9 -step 0.1 -base 10.0 -expression I", 
				"-search", "weka.core.setupgenerator.ListParameter -property classifier.classifier.numClusters -list \"50 100 200 220 250\"",
				"-sample-size", "50.0",
				"-log-file", "/projects/huanlab/wei/results16/RBFNetwork_WrapperSubsetEval_BestFirst.txt", 
				"-initial-folds", "10", 
				"-subsequent-folds", "10", 
				"-num-slots", "16", 
				"-S", "1", 
				"-W", "weka.classifiers.meta.AttributeSelectedClassifier", "-do-not-check-capabilities", "--", 
				"-E", "weka.attributeSelection.WrapperSubsetEval -B weka.classifiers.functions.RBFNetwork -F 10 -T 0.01 -R 1 -E ACC -- -B 200 -S 1 -R 1.0E2 -M -1 -W 0.7 -do-not-check-capabilities", 
				"-S", "weka.attributeSelection.BestFirst -D 2 -N 6",
				"-W", "weka.classifiers.functions.RBFNetwork", "-do-not-check-capabilities", "--", "-B", "200", "-S", "1", "-R", "1.0E-2", "-M", "-1", "-W", "0.7", "-do-not-check-capabilities"
			},
			
			{	// 6. RBFNetwork + WrapperSubsetEval + GreedyStepwise
				"-E", "ACC", 
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.ridge -min -4.0 -max 4.0 -step 2.0 -base 10.0 -expression pow(BASE,I)",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.minStdDev -min 0.1 -max 0.9 -step 0.1 -base 10.0 -expression I", 
				"-search", "weka.core.setupgenerator.ListParameter -property classifier.classifier.numClusters -list \"50 100 200 220 250\"",
				"-sample-size", "50.0", 
				"-log-file", "/projects/huanlab/wei/results/RBFNetwork_WrapperSubsetEval_GreedyStepwise.txt", 
				"-initial-folds", "10", 
				"-subsequent-folds", "10", 
				"-num-slots", "8", 
				"-S", "1", 
				"-W", "weka.classifiers.meta.AttributeSelectedClassifier", "-do-not-check-capabilities", "--", 
				"-E", "weka.attributeSelection.WrapperSubsetEval -B weka.classifiers.functions.RBFNetwork -F 10 -T 0.01 -R 1 -E ACC -- -B 200 -S 1 -R 1.0E2 -M -1 -W 0.7 -do-not-check-capabilities", 
				"-S", "weka.attributeSelection.GreedyStepwise -T -1.7976931348623157E308 -N -1 -num-slots 1",
				"-W", "weka.classifiers.functions.RBFNetwork", "-do-not-check-capabilities", "--", "-B", "200", "-S", "1", "-R", "1.0E-2", "-M", "-1", "-W", "0.7", "-do-not-check-capabilities"
			}
		};
		
		System.out.println("Testing...");
		multisearch.setOptions(parameters[4]);
		multisearch.buildClassifier(data_train);
		System.out.println("Done");
	}
}
