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
import weka.filters.unsupervised.attribute.NumericToBinary;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.core.SerializationHelper;
import weka.classifiers.evaluation.output.prediction.CSV;
import weka.classifiers.evaluation.output.prediction.PlainText;
import weka.core.FastVector;
 
public class iLogSymmRank {

	public static void main(String[] args) throws Exception {
		
		CSVLoader loader_train = new CSVLoader();
		loader_train.setSource(new File(
				"EECS738_Train.csv"));

		Instances data_train = loader_train.getDataSet();

		// Tell weka that the second column contains the class labels
		data_train.setClassIndex(0);
		
		MultiSearch multisearch = new MultiSearch();

		String[][] parameters = {

			{	// 1. Logistic + CfsSubsetEval + BestFirst
				"-E", "ACC", 
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.maxIts -min -1 -max 12.0 -step 1.0 -base 10.0 -expression I",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.ridge -min -10.0 -max 10.0 -step 1.0 -base 10.0 -expression pow(BASE,I)",
				"-sample-size", "10.0", 
				"-log-file", "/projects/huanlab/wei/results/Logistic_CfsSubsetEval_BestFirst.txt", 
				"-initial-folds", "10", 
				"-subsequent-folds", "10", 
				"-num-slots", "8", 
				"-S", "1", 
				"-W", "weka.classifiers.meta.AttributeSelectedClassifier", "-do-not-check-capabilities", "--", 
				"-E", "weka.attributeSelection.CfsSubsetEval -P 1 -E 1", 
				"-S", "weka.attributeSelection.BestFirst -D 2 -N 6",
				"-W", "weka.classifiers.functions.Logistic", "-do-not-check-capabilities", "--", "-R", "1.0E-6", "-M", "-1"
			},
			
			{	// 2. Logistic + CfsSubsetEval + GreedyStepwise
				"-E", "ACC", 
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.maxIts -min -1 -max 12.0 -step 1.0 -base 10.0 -expression I",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.ridge -min -10.0 -max 10.0 -step 1.0 -base 10.0 -expression pow(BASE,I)",
				"-sample-size", "100.0", 
				"-log-file", "/projects/huanlab/wei/results/Logistic_CfsSubsetEval_GreedyStepwise.txt", 
				"-initial-folds", "10", 
				"-subsequent-folds", "10", 
				"-num-slots", "8", 
				"-S", "1", 
				"-W", "weka.classifiers.meta.AttributeSelectedClassifier", "-do-not-check-capabilities", "--", 
				"-E", "weka.attributeSelection.CfsSubsetEval -P 1 -E 1", 
				"-S", "weka.attributeSelection.GreedyStepwise -T -1.7976931348623157E308 -N -1 -num-slots 1",
				"-W", "weka.classifiers.functions.Logistic", "-do-not-check-capabilities", "--", "-R", "1.0E-6", "-M", "-1"
			},
			
			{	// 3. Logistic + CorrelationAttributeEval + Ranker
				"-E", "ACC", 
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.maxIts -min -1 -max 12.0 -step 1.0 -base 10.0 -expression I",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.ridge -min -10.0 -max 10.0 -step 1.0 -base 10.0 -expression pow(BASE,I)",
				"-sample-size", "100.0", 
				"-log-file", "/projects/huanlab/wei/results/Logistic_CorrelationAttributeEval_Ranker.txt", 
				"-initial-folds", "10", 
				"-subsequent-folds", "10", 
				"-num-slots", "8", 
				"-S", "1", 
				"-W", "weka.classifiers.meta.AttributeSelectedClassifier", "-do-not-check-capabilities", "--", 
				"-E", "weka.attributeSelection.CorrelationAttributeEval ", 
				"-S", "weka.attributeSelection.Ranker -T -1.7976931348623157E308 -N -1",
				"-W", "weka.classifiers.functions.Logistic", "-do-not-check-capabilities", "--", "-R", "1.0E-6", "-M", "-1"
			},
			
			{	// 4. Logistic + GainRatioAttributeEval + Ranker
				"-E", "ACC", 
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.maxIts -min -1 -max 12.0 -step 1.0 -base 10.0 -expression I",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.ridge -min -10.0 -max 10.0 -step 1.0 -base 10.0 -expression pow(BASE,I)",
				"-sample-size", "100.0", 
				"-log-file", "/projects/huanlab/wei/results/Logistic_GainRatioAttributeEval_Ranker.txt", 
				"-initial-folds", "10", 
				"-subsequent-folds", "10", 
				"-num-slots", "8", 
				"-S", "1", 
				"-W", "weka.classifiers.meta.AttributeSelectedClassifier", "-do-not-check-capabilities", "--", 
				"-E", "weka.attributeSelection.GainRatioAttributeEval ", 
				"-S", "weka.attributeSelection.Ranker -T -1.7976931348623157E308 -N -1",
				"-W", "weka.classifiers.functions.Logistic", "-do-not-check-capabilities", "--", "-R", "1.0E-6", "-M", "-1"
			},
			
			{	// 5. Logistic + InfoGainAttributeEval + Ranker
				"-E", "ACC", 
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.maxIts -min -1 -max 12.0 -step 1.0 -base 10.0 -expression I",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.ridge -min -10.0 -max 10.0 -step 1.0 -base 10.0 -expression pow(BASE,I)",
				"-sample-size", "100.0", 
				"-log-file", "/projects/huanlab/wei/results/Logistic_InfoGainAttributeEval_Ranker.txt", 
				"-initial-folds", "10", 
				"-subsequent-folds", "10", 
				"-num-slots", "8", 
				"-S", "1", 
				"-W", "weka.classifiers.meta.AttributeSelectedClassifier", "-do-not-check-capabilities", "--", 
				"-E", "weka.attributeSelection.InfoGainAttributeEval ", 
				"-S", "weka.attributeSelection.Ranker -T -1.7976931348623157E308 -N -1",
				"-W", "weka.classifiers.functions.Logistic", "-do-not-check-capabilities", "--", "-R", "1.0E-6", "-M", "-1"
			},
			
			{	// 6. Logistic + OneRAttributeEval + Ranker
				"-E", "ACC", 
				"-search", "weka.core.setupgenerator.ListParameter -property classifier.evaluator.minimumBucketSize -list \"6 10 15 20 30 50 100\"",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.maxIts -min -1 -max 12.0 -step 1.0 -base 10.0 -expression I",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.ridge -min -10.0 -max 10.0 -step 1.0 -base 10.0 -expression pow(BASE,I)",
				"-sample-size", "100.0", 
				"-log-file", "/projects/huanlab/wei/results/Logistic_OneRAttributeEval_Ranker.txt", 
				"-initial-folds", "10", 
				"-subsequent-folds", "10", 
				"-num-slots", "8", 
				"-S", "1", 
				"-W", "weka.classifiers.meta.AttributeSelectedClassifier", "-do-not-check-capabilities", "--", 
				"-E", "weka.attributeSelection.OneRAttributeEval -S 1 -F 10 -B 6", 
				"-S", "weka.attributeSelection.Ranker -T -1.7976931348623157E308 -N -1",
				"-W", "weka.classifiers.functions.Logistic", "-do-not-check-capabilities", "--", "-R", "1.0E-6", "-M", "-1"
			},
			
			{	// 7. Logistic + PrincipalComponents + Ranker
				"-E", "ACC", 
				"-search", "weka.core.setupgenerator.ListParameter -property classifier.evaluator.centerData -list \"True False\"",
				"-search", "weka.core.setupgenerator.ListParameter -property classifier.evaluator.transformBackToOriginal -list \"True False\"",
				"-search", "weka.core.setupgenerator.ListParameter -property classifier.evaluator.maximumAttributeNames -list \"5 8 12 16 20 30 40 50\"",
				"-search", "weka.core.setupgenerator.ListParameter -property classifier.evaluator.varianceCovered -list \"0.6 0.8 0.9 0.95 0.99\"",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.maxIts -min -1 -max 12.0 -step 1.0 -base 10.0 -expression I",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.ridge -min -10.0 -max 10.0 -step 1.0 -base 10.0 -expression pow(BASE,I)",
				"-sample-size", "100.0", 
				"-log-file", "/projects/huanlab/wei/results/Logistic_PrincipalComponents_Ranker.txt", 
				"-initial-folds", "10", 
				"-subsequent-folds", "10", 
				"-num-slots", "8", 
				"-S", "1", 
				"-W", "weka.classifiers.meta.AttributeSelectedClassifier", "-do-not-check-capabilities", "--", 
				"-E", "weka.attributeSelection.PrincipalComponents -R 0.95 -A 5", 
				"-S", "weka.attributeSelection.Ranker -T -1.7976931348623157E308 -N -1",
				"-W", "weka.classifiers.functions.Logistic", "-do-not-check-capabilities", "--", "-R", "1.0E-6", "-M", "-1"
			},
			
			{	// 8. Logistic + ReliefFAttributeEval + Ranker
				"-E", "ACC", 
				"-search", "weka.core.setupgenerator.ListParameter -property classifier.evaluator.numNeighbours -list \"5 10 15 20 30 50\"",
				"-search", "weka.core.setupgenerator.ListParameter -property classifier.evaluator.weightByDistance -list \"True False\"",
				"-search", "weka.core.setupgenerator.ListParameter -property classifier.evaluator.sigma -list \"1 2 3 4 6 10\"",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.maxIts -min -1 -max 12.0 -step 1.0 -base 10.0 -expression I",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.ridge -min -10.0 -max 10.0 -step 1.0 -base 10.0 -expression pow(BASE,I)",
				"-sample-size", "100.0", 
				"-log-file", "/projects/huanlab/wei/results/Logistic_ReliefFAttributeEval_Ranker.txt", 
				"-initial-folds", "10", 
				"-subsequent-folds", "10", 
				"-num-slots", "8", 
				"-S", "1", 
				"-W", "weka.classifiers.meta.AttributeSelectedClassifier", "-do-not-check-capabilities", "--", 
				"-E", "weka.attributeSelection.ReliefFAttributeEval -M -1 -D 1 -K 10", 
				"-S", "weka.attributeSelection.Ranker -T -1.7976931348623157E308 -N -1",
				"-W", "weka.classifiers.functions.Logistic", "-do-not-check-capabilities", "--", "-R", "1.0E-6", "-M", "-1"
			},
			
			{	// 9. Logistic + SymmetricalUncertAttributeEval + Ranker
				"-E", "ACC", 
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.maxIts -min -1 -max 12.0 -step 1.0 -base 10.0 -expression I",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.ridge -min -10.0 -max 10.0 -step 1.0 -base 10.0 -expression pow(BASE,I)",
				"-sample-size", "100.0", 
				"-log-file", "/projects/huanlab/wei/results/Logistic_SymmetricalUncertAttributeEval_BestFirst.txt", 
				"-initial-folds", "10", 
				"-subsequent-folds", "10", 
				"-num-slots", "8", 
				"-S", "1", 
				"-W", "weka.classifiers.meta.AttributeSelectedClassifier", "-do-not-check-capabilities", "--", 
				"-E", "weka.attributeSelection.SymmetricalUncertAttributeEval ", 
				"-S", "weka.attributeSelection.Ranker -T -1.7976931348623157E308 -N -1",
				"-W", "weka.classifiers.functions.Logistic", "-do-not-check-capabilities", "--", "-R", "1.0E-6", "-M", "-1"
			},

			{	// 10. RBFNetwork + CfsSubsetEval + BestFirst
				"-E", "ACC", 
				"-search", "weka.core.setupgenerator.ListParameter -property classifier.maxIts -list \"-1 1 5 10 20 50\"",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.ridge -min -4.0 -max 4.0 -step 2.0 -base 10.0 -expression pow(BASE,I)",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.minStdDev -min 0.1 -max 0.9 -step 0.1 -base 10.0 -expression I", 
				"-search", "weka.core.setupgenerator.ListParameter -property classifier.numClusters -list \"5 10 20 50 100 200\"",
				"-sample-size", "100.0", 
				"-log-file", "/projects/huanlab/wei/results/RBFNetwork_CfsSubsetEval_BestFirst.txt", 
				"-initial-folds", "10", 
				"-subsequent-folds", "10", 
				"-num-slots", "8", 
				"-S", "1", 
				"-W", "weka.classifiers.meta.AttributeSelectedClassifier", "-do-not-check-capabilities", "--", 
				"-E", "weka.attributeSelection.CfsSubsetEval -P 1 -E 1", 
				"-S", "weka.attributeSelection.BestFirst -D 2 -N 6",
				"-W", "weka.classifiers.functions.Logistic", "-do-not-check-capabilities", "--", "-R", "1.0E-6", "-M", "-1"
			},
			
			{	// 11. RBFNetwork + CfsSubsetEval + GreedyStepwise
				"-E", "ACC", 
				"-search", "weka.core.setupgenerator.ListParameter -property classifier.maxIts -list \"-1 1 5 10 20 50\"",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.ridge -min -4.0 -max 4.0 -step 2.0 -base 10.0 -expression pow(BASE,I)",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.minStdDev -min 0.1 -max 0.9 -step 0.1 -base 10.0 -expression I", 
				"-search", "weka.core.setupgenerator.ListParameter -property classifier.numClusters -list \"5 10 20 50 100 200\"",
				"-sample-size", "100.0", 
				"-log-file", "/projects/huanlab/wei/results/RBFNetwork_CfsSubsetEval_GreedyStepwise.txt", 
				"-initial-folds", "10", 
				"-subsequent-folds", "10", 
				"-num-slots", "8", 
				"-S", "1", 
				"-W", "weka.classifiers.meta.AttributeSelectedClassifier", "-do-not-check-capabilities", "--", 
				"-E", "weka.attributeSelection.CfsSubsetEval -P 1 -E 1", 
				"-S", "weka.attributeSelection.GreedyStepwise -T -1.7976931348623157E308 -N -1 -num-slots 1",
				"-W", "weka.classifiers.functions.Logistic", "-do-not-check-capabilities", "--", "-R", "1.0E-6", "-M", "-1"
			},
			
			{	// 12. RBFNetwork + CorrelationAttributeEval + Ranker
				"-E", "ACC", 
				"-search", "weka.core.setupgenerator.ListParameter -property classifier.maxIts -list \"-1 1 5 10 20 50\"",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.ridge -min -4.0 -max 4.0 -step 2.0 -base 10.0 -expression pow(BASE,I)",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.minStdDev -min 0.1 -max 0.9 -step 0.1 -base 10.0 -expression I", 
				"-search", "weka.core.setupgenerator.ListParameter -property classifier.numClusters -list \"5 10 20 50 100 200\"",
				"-sample-size", "100.0", 
				"-log-file", "/projects/huanlab/wei/results/RBFNetwork_CorrelationAttributeEval_Ranker.txt", 
				"-initial-folds", "10", 
				"-subsequent-folds", "10", 
				"-num-slots", "8", 
				"-S", "1", 
				"-W", "weka.classifiers.meta.AttributeSelectedClassifier", "-do-not-check-capabilities", "--", 
				"-E", "weka.attributeSelection.CorrelationAttributeEval ", 
				"-S", "weka.attributeSelection.Ranker -T -1.7976931348623157E308 -N -1",
				"-W", "weka.classifiers.functions.Logistic", "-do-not-check-capabilities", "--", "-R", "1.0E-6", "-M", "-1"
			},
			
			{	// 13. RBFNetwork + GainRatioAttributeEval + Ranker
				"-E", "ACC", 
				"-search", "weka.core.setupgenerator.ListParameter -property classifier.maxIts -list \"-1 1 5 10 20 50\"",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.ridge -min -4.0 -max 4.0 -step 2.0 -base 10.0 -expression pow(BASE,I)",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.minStdDev -min 0.1 -max 0.9 -step 0.1 -base 10.0 -expression I", 
				"-search", "weka.core.setupgenerator.ListParameter -property classifier.numClusters -list \"5 10 20 50 100 200\"",
				"-sample-size", "100.0", 
				"-log-file", "/projects/huanlab/wei/results/RBFNetwork_GainRatioAttributeEval_Ranker.txt", 
				"-initial-folds", "10", 
				"-subsequent-folds", "10", 
				"-num-slots", "8", 
				"-S", "1", 
				"-W", "weka.classifiers.meta.AttributeSelectedClassifier", "-do-not-check-capabilities", "--", 
				"-E", "weka.attributeSelection.GainRatioAttributeEval ", 
				"-S", "weka.attributeSelection.Ranker -T -1.7976931348623157E308 -N -1",
				"-W", "weka.classifiers.functions.Logistic", "-do-not-check-capabilities", "--", "-R", "1.0E-6", "-M", "-1"
			},
			
			{	// 14. RBFNetwork + InfoGainAttributeEval + Ranker
				"-E", "ACC", 
				"-search", "weka.core.setupgenerator.ListParameter -property classifier.maxIts -list \"-1 1 5 10 20 50\"",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.ridge -min -4.0 -max 4.0 -step 2.0 -base 10.0 -expression pow(BASE,I)",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.minStdDev -min 0.1 -max 0.9 -step 0.1 -base 10.0 -expression I", 
				"-search", "weka.core.setupgenerator.ListParameter -property classifier.numClusters -list \"5 10 20 50 100 200\"",
				"-sample-size", "100.0", 
				"-log-file", "/projects/huanlab/wei/results/RBFNetwork_InfoGainAttributeEval_Ranker.txt", 
				"-initial-folds", "10", 
				"-subsequent-folds", "10", 
				"-num-slots", "8", 
				"-S", "1", 
				"-W", "weka.classifiers.meta.AttributeSelectedClassifier", "-do-not-check-capabilities", "--", 
				"-E", "weka.attributeSelection.InfoGainAttributeEval ", 
				"-S", "weka.attributeSelection.Ranker -T -1.7976931348623157E308 -N -1",
				"-W", "weka.classifiers.functions.Logistic", "-do-not-check-capabilities", "--", "-R", "1.0E-6", "-M", "-1"
			},
			
			{	// 15. RBFNetwork + OneRAttributeEval + Ranker
				"-E", "ACC", 
				"-search", "weka.core.setupgenerator.ListParameter -property classifier.evaluator.minimumBucketSize -list \"6 10 15 20 30 50 100\"",
				"-search", "weka.core.setupgenerator.ListParameter -property classifier.maxIts -list \"-1 1 5 10 20 50\"",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.ridge -min -4.0 -max 4.0 -step 2.0 -base 10.0 -expression pow(BASE,I)",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.minStdDev -min 0.1 -max 0.9 -step 0.1 -base 10.0 -expression I", 
				"-search", "weka.core.setupgenerator.ListParameter -property classifier.numClusters -list \"5 10 20 50 100 200\"",
				"-sample-size", "100.0", 
				"-log-file", "/projects/huanlab/wei/results/RBFNetwork_OneRAttributeEval_Ranker.txt", 
				"-initial-folds", "10", 
				"-subsequent-folds", "10", 
				"-num-slots", "8", 
				"-S", "1", 
				"-W", "weka.classifiers.meta.AttributeSelectedClassifier", "-do-not-check-capabilities", "--", 
				"-E", "weka.attributeSelection.OneRAttributeEval -S 1 -F 10 -B 6", 
				"-S", "weka.attributeSelection.Ranker -T -1.7976931348623157E308 -N -1",
				"-W", "weka.classifiers.functions.Logistic", "-do-not-check-capabilities", "--", "-R", "1.0E-6", "-M", "-1"
			},
			
			{	// 16. RBFNetwork + PrincipalComponents + Ranker
				"-E", "ACC", 
				"-search", "weka.core.setupgenerator.ListParameter -property classifier.evaluator.centerData -list \"True False\"",
				"-search", "weka.core.setupgenerator.ListParameter -property classifier.evaluator.transformBackToOriginal -list \"True False\"",
				"-search", "weka.core.setupgenerator.ListParameter -property classifier.evaluator.maximumAttributeNames -list \"5 8 12 16 20 30 40 50\"",
				"-search", "weka.core.setupgenerator.ListParameter -property classifier.evaluator.varianceCovered -list \"0.6 0.8 0.9 0.95 0.99\"",
				"-search", "weka.core.setupgenerator.ListParameter -property classifier.maxIts -list \"-1 1 5 10 20 50\"",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.ridge -min -4.0 -max 4.0 -step 2.0 -base 10.0 -expression pow(BASE,I)",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.minStdDev -min 0.1 -max 0.9 -step 0.1 -base 10.0 -expression I", 
				"-search", "weka.core.setupgenerator.ListParameter -property classifier.numClusters -list \"5 10 20 50 100 200\"",
				"-sample-size", "100.0", 
				"-log-file", "/projects/huanlab/wei/results/RBFNetwork_PrincipalComponents_Ranker.txt", 
				"-initial-folds", "10", 
				"-subsequent-folds", "10", 
				"-num-slots", "8", 
				"-S", "1", 
				"-W", "weka.classifiers.meta.AttributeSelectedClassifier", "-do-not-check-capabilities", "--", 
				"-E", "weka.attributeSelection.PrincipalComponents -R 0.95 -A 5", 
				"-S", "weka.attributeSelection.Ranker -T -1.7976931348623157E308 -N -1",
				"-W", "weka.classifiers.functions.Logistic", "-do-not-check-capabilities", "--", "-R", "1.0E-6", "-M", "-1"
			},
			
			{	// 17. RBFNetwork + ReliefFAttributeEval + Ranker
				"-E", "ACC", 
				"-search", "weka.core.setupgenerator.ListParameter -property classifier.evaluator.numNeighbours -list \"5 10 15 20 30 50\"",
				"-search", "weka.core.setupgenerator.ListParameter -property classifier.evaluator.weightByDistance -list \"True False\"",
				"-search", "weka.core.setupgenerator.ListParameter -property classifier.evaluator.sigma -list \"1 2 3 4 6 10\"",
				"-search", "weka.core.setupgenerator.ListParameter -property classifier.maxIts -list \"-1 1 5 10 20 50\"",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.ridge -min -4.0 -max 4.0 -step 2.0 -base 10.0 -expression pow(BASE,I)",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.minStdDev -min 0.1 -max 0.9 -step 0.1 -base 10.0 -expression I", 
				"-search", "weka.core.setupgenerator.ListParameter -property classifier.numClusters -list \"5 10 20 50 100 200\"",
				"-sample-size", "100.0", 
				"-log-file", "/projects/huanlab/wei/results/RBFNetwork_ReliefFAttributeEval_Ranker.txt", 
				"-initial-folds", "10", 
				"-subsequent-folds", "10", 
				"-num-slots", "8", 
				"-S", "1", 
				"-W", "weka.classifiers.meta.AttributeSelectedClassifier", "-do-not-check-capabilities", "--", 
				"-E", "weka.attributeSelection.ReliefFAttributeEval -M -1 -D 1 -K 10", 
				"-S", "weka.attributeSelection.Ranker -T -1.7976931348623157E308 -N -1",
				"-W", "weka.classifiers.functions.Logistic", "-do-not-check-capabilities", "--", "-R", "1.0E-6", "-M", "-1"
			},
			
			{	// 18. RBFNetwork + SymmetricalUncertAttributeEval + Ranker
				"-E", "ACC", 
				"-search", "weka.core.setupgenerator.ListParameter -property classifier.maxIts -list \"-1 1 5 10 20 50\"",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.ridge -min -4.0 -max 4.0 -step 2.0 -base 10.0 -expression pow(BASE,I)",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.minStdDev -min 0.1 -max 0.9 -step 0.1 -base 10.0 -expression I", 
				"-search", "weka.core.setupgenerator.ListParameter -property classifier.numClusters -list \"5 10 20 50 100 200\"",
				"-sample-size", "100.0", 
				"-log-file", "/projects/huanlab/wei/results/RBFNetwork_SymmetricalUncertAttributeEval_BestFirst.txt", 
				"-initial-folds", "10", 
				"-subsequent-folds", "10", 
				"-num-slots", "8", 
				"-S", "1", 
				"-W", "weka.classifiers.meta.AttributeSelectedClassifier", "-do-not-check-capabilities", "--", 
				"-E", "weka.attributeSelection.CorrelationAttributeEval ", 
				"-S", "weka.attributeSelection.Ranker -T -1.7976931348623157E308 -N -1",
				"-W", "weka.classifiers.functions.Logistic", "-do-not-check-capabilities", "--", "-R", "1.0E-6", "-M", "-1"
			}
		};
		
		System.out.println("Testing...");
		multisearch.setOptions(parameters[8]);
		multisearch.buildClassifier(data_train);
		System.out.println("Done");
	}
}
