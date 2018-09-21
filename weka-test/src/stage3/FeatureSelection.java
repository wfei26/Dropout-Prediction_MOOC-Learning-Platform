package stage3;

import java.io.File;
import weka.classifiers.meta.MultiSearch;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
 
public class FeatureSelection {

	public static void main(String[] args) throws Exception {
		
		CSVLoader loader_train = new CSVLoader();
		loader_train.setSource(new File(
				"/Users/syan/Dropbox/Study/2015Fall/EECS 738/Final Project/EECS738_Train_Yes_No.csv"));

		Instances data_train = loader_train.getDataSet();

		// Tell weka that the second column contains the class labels
		data_train.setClassIndex(1);
		
		MultiSearch multisearch = new MultiSearch();

		String[][] parameters = {

			{	// Logistic + CfsSubsetEval + BestFirst
				"-E", "ACC", 
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.maxIts -min -1 -max 12.0 -step 1.0 -base 10.0 -expression I",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.ridge -min -10.0 -max 10.0 -step 1.0 -base 10.0 -expression pow(BASE,I)",
				"-sample-size", "10.0", 
				"-log-file", "/Users/syan/Documents/Logistic_CfsSubsetEval_BestFirst.txt", 
				"-initial-folds", "10", 
				"-subsequent-folds", "10", 
				"-num-slots", "8", 
				"-S", "1", 
				"-W", "weka.classifiers.meta.AttributeSelectedClassifier", "-do-not-check-capabilities", "--", 
				"-E", "weka.attributeSelection.CfsSubsetEval -P 1 -E 1", 
				"-S", "weka.attributeSelection.BestFirst -D 2 -N 6",
				"-W", "weka.classifiers.functions.Logistic", "-do-not-check-capabilities", "--", "-R", "1.0E-6", "-M", "-1"
			},
			
			{	// Logistic + CfsSubsetEval + GreedyStepwise
				"-E", "ACC", 
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.maxIts -min -1 -max 12.0 -step 1.0 -base 10.0 -expression I",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.ridge -min -10.0 -max 10.0 -step 1.0 -base 10.0 -expression pow(BASE,I)",
				"-sample-size", "100.0", 
				"-log-file", "/Users/syan/Documents/Logistic_CfsSubsetEval_GreedyStepwise.txt", 
				"-initial-folds", "10", 
				"-subsequent-folds", "10", 
				"-num-slots", "8", 
				"-S", "1", 
				"-W", "weka.classifiers.meta.AttributeSelectedClassifier", "-do-not-check-capabilities", "--", 
				"-E", "weka.attributeSelection.CfsSubsetEval -P 1 -E 1", 
				"-S", "weka.attributeSelection.GreedyStepwise -T -1.7976931348623157E308 -N -1 -num-slots 1",
				"-W", "weka.classifiers.functions.Logistic", "-do-not-check-capabilities", "--", "-R", "1.0E-6", "-M", "-1"
			},
			
			{	// Logistic + CorrelationAttributeEval + Ranker
				"-E", "ACC", 
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.maxIts -min -1 -max 12.0 -step 1.0 -base 10.0 -expression I",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.ridge -min -10.0 -max 10.0 -step 1.0 -base 10.0 -expression pow(BASE,I)",
				"-sample-size", "100.0", 
				"-log-file", "/Users/syan/Documents/Logistic_CorrelationAttributeEval_Ranker.txt", 
				"-initial-folds", "10", 
				"-subsequent-folds", "10", 
				"-num-slots", "8", 
				"-S", "1", 
				"-W", "weka.classifiers.meta.AttributeSelectedClassifier", "-do-not-check-capabilities", "--", 
				"-E", "weka.attributeSelection.CorrelationAttributeEval ", 
				"-S", "weka.attributeSelection.Ranker -T -1.7976931348623157E308 -N -1",
				"-W", "weka.classifiers.functions.Logistic", "-do-not-check-capabilities", "--", "-R", "1.0E-6", "-M", "-1"
			},
			
			{	// Logistic + GainRatioAttributeEval + Ranker
				"-E", "ACC", 
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.maxIts -min -1 -max 12.0 -step 1.0 -base 10.0 -expression I",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.ridge -min -10.0 -max 10.0 -step 1.0 -base 10.0 -expression pow(BASE,I)",
				"-sample-size", "100.0", 
				"-log-file", "/Users/syan/Documents/Logistic_GainRatioAttributeEval_Ranker.txt", 
				"-initial-folds", "10", 
				"-subsequent-folds", "10", 
				"-num-slots", "8", 
				"-S", "1", 
				"-W", "weka.classifiers.meta.AttributeSelectedClassifier", "-do-not-check-capabilities", "--", 
				"-E", "weka.attributeSelection.GainRatioAttributeEval ", 
				"-S", "weka.attributeSelection.Ranker -T -1.7976931348623157E308 -N -1",
				"-W", "weka.classifiers.functions.Logistic", "-do-not-check-capabilities", "--", "-R", "1.0E-6", "-M", "-1"
			},
			
			{	// Logistic + InfoGainAttributeEval + Ranker
				"-E", "ACC", 
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.maxIts -min -1 -max 12.0 -step 1.0 -base 10.0 -expression I",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.ridge -min -10.0 -max 10.0 -step 1.0 -base 10.0 -expression pow(BASE,I)",
				"-sample-size", "100.0", 
				"-log-file", "/Users/syan/Documents/Logistic_InfoGainAttributeEval_Ranker.txt", 
				"-initial-folds", "10", 
				"-subsequent-folds", "10", 
				"-num-slots", "8", 
				"-S", "1", 
				"-W", "weka.classifiers.meta.AttributeSelectedClassifier", "-do-not-check-capabilities", "--", 
				"-E", "weka.attributeSelection.InfoGainAttributeEval ", 
				"-S", "weka.attributeSelection.Ranker -T -1.7976931348623157E308 -N -1",
				"-W", "weka.classifiers.functions.Logistic", "-do-not-check-capabilities", "--", "-R", "1.0E-6", "-M", "-1"
			},
			
			{	// Logistic + OneRAttributeEval + Ranker
				"-E", "ACC", 
				"-search", "weka.core.setupgenerator.ListParameter -property classifier.evaluator.minimumBucketSize -list \"6 10 15 20 30 50 100\"",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.maxIts -min -1 -max 12.0 -step 1.0 -base 10.0 -expression I",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.ridge -min -10.0 -max 10.0 -step 1.0 -base 10.0 -expression pow(BASE,I)",
				"-sample-size", "100.0", 
				"-log-file", "/Users/syan/Documents/Logistic_OneRAttributeEval_Ranker.txt", 
				"-initial-folds", "10", 
				"-subsequent-folds", "10", 
				"-num-slots", "8", 
				"-S", "1", 
				"-W", "weka.classifiers.meta.AttributeSelectedClassifier", "-do-not-check-capabilities", "--", 
				"-E", "weka.attributeSelection.OneRAttributeEval -S 1 -F 10 -B 6", 
				"-S", "weka.attributeSelection.Ranker -T -1.7976931348623157E308 -N -1",
				"-W", "weka.classifiers.functions.Logistic", "-do-not-check-capabilities", "--", "-R", "1.0E-6", "-M", "-1"
			},
			
			{	// Logistic + PrincipalComponents + Ranker
				"-E", "ACC", 
				"-search", "weka.core.setupgenerator.ListParameter -property classifier.evaluator.centerData -list \"True False\"",
				"-search", "weka.core.setupgenerator.ListParameter -property classifier.evaluator.transformBackToOriginal -list \"True False\"",
				"-search", "weka.core.setupgenerator.ListParameter -property classifier.evaluator.maximumAttributeNames -list \"5 8 12 16 20 30 40 50\"",
				"-search", "weka.core.setupgenerator.ListParameter -property classifier.evaluator.varianceCovered -list \"0.6 0.8 0.9 0.95 0.99\"",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.maxIts -min -1 -max 12.0 -step 1.0 -base 10.0 -expression I",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.ridge -min -10.0 -max 10.0 -step 1.0 -base 10.0 -expression pow(BASE,I)",
				"-sample-size", "100.0", 
				"-log-file", "/Users/syan/Documents/Logistic_PrincipalComponents_Ranker.txt", 
				"-initial-folds", "10", 
				"-subsequent-folds", "10", 
				"-num-slots", "8", 
				"-S", "1", 
				"-W", "weka.classifiers.meta.AttributeSelectedClassifier", "-do-not-check-capabilities", "--", 
				"-E", "weka.attributeSelection.PrincipalComponents -R 0.95 -A 5", 
				"-S", "weka.attributeSelection.Ranker -T -1.7976931348623157E308 -N -1",
				"-W", "weka.classifiers.functions.Logistic", "-do-not-check-capabilities", "--", "-R", "1.0E-6", "-M", "-1"
			},
			
			{	// Logistic + ReliefFAttributeEval + Ranker
				"-E", "ACC", 
				"-search", "weka.core.setupgenerator.ListParameter -property classifier.evaluator.numNeighbours -list \"5 10 15 20 30 50\"",
				"-search", "weka.core.setupgenerator.ListParameter -property classifier.evaluator.weightByDistance -list \"True False\"",
				"-search", "weka.core.setupgenerator.ListParameter -property classifier.evaluator.sigma -list \"1 2 3 4 6 10\"",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.maxIts -min -1 -max 12.0 -step 1.0 -base 10.0 -expression I",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.ridge -min -10.0 -max 10.0 -step 1.0 -base 10.0 -expression pow(BASE,I)",
				"-sample-size", "100.0", 
				"-log-file", "/Users/syan/Documents/Logistic_ReliefFAttributeEval_Ranker.txt", 
				"-initial-folds", "10", 
				"-subsequent-folds", "10", 
				"-num-slots", "8", 
				"-S", "1", 
				"-W", "weka.classifiers.meta.AttributeSelectedClassifier", "-do-not-check-capabilities", "--", 
				"-E", "weka.attributeSelection.ReliefFAttributeEval -M -1 -D 1 -K 10", 
				"-S", "weka.attributeSelection.Ranker -T -1.7976931348623157E308 -N -1",
				"-W", "weka.classifiers.functions.Logistic", "-do-not-check-capabilities", "--", "-R", "1.0E-6", "-M", "-1"
			},
			
			{	// Logistic + SymmetricalUncertAttributeEval + Ranker
				"-E", "ACC", 
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.maxIts -min -1 -max 12.0 -step 1.0 -base 10.0 -expression I",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.ridge -min -10.0 -max 10.0 -step 1.0 -base 10.0 -expression pow(BASE,I)",
				"-sample-size", "100.0", 
				"-log-file", "/Users/syan/Documents/Logistic_SymmetricalUncertAttributeEval_BestFirst.txt", 
				"-initial-folds", "10", 
				"-subsequent-folds", "10", 
				"-num-slots", "8", 
				"-S", "1", 
				"-W", "weka.classifiers.meta.AttributeSelectedClassifier", "-do-not-check-capabilities", "--", 
				"-E", "weka.attributeSelection.CorrelationAttributeEval ", 
				"-S", "weka.attributeSelection.Ranker -T -1.7976931348623157E308 -N -1",
				"-W", "weka.classifiers.functions.Logistic", "-do-not-check-capabilities", "--", "-R", "1.0E-6", "-M", "-1"
			},

			{	// RBFNetwork + CfsSubsetEval + BestFirst
				"-E", "ACC", 
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.maxIts -min -1 -max 9.0 -step 1.0 -base 10.0 -expression I",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.ridge -min -8.0 -max 4 -step 2.0 -base 10.0 -expression pow(BASE,I)",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.minStdDev -min 0.1 -max 0.5 -step 0.1 -base 10.0 -expression I", 
				"-sample-size", "100.0", 
				"-log-file", "/Users/syan/Documents/RBFNetwork_CfsSubsetEval_BestFirst.txt", 
				"-initial-folds", "10", 
				"-subsequent-folds", "10", 
				"-num-slots", "8", 
				"-S", "1", 
				"-W", "weka.classifiers.meta.AttributeSelectedClassifier", "-do-not-check-capabilities", "--", 
				"-E", "weka.attributeSelection.CfsSubsetEval -P 1 -E 1", 
				"-S", "weka.attributeSelection.BestFirst -D 2 -N 6",
				"-W", "weka.classifiers.functions.RBFNetwork", "-do-not-check-capabilities", "--", "-R", "1.0E-6", "-M", "-1"
			},
			
			{	// RBFNetwork + CfsSubsetEval + GreedyStepwise
				"-E", "ACC", 
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.maxIts -min -1 -max 9.0 -step 1.0 -base 10.0 -expression I",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.ridge -min -8.0 -max 4 -step 2.0 -base 10.0 -expression pow(BASE,I)",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.minStdDev -min 0.1 -max 0.5 -step 0.1 -base 10.0 -expression I", 
				"-sample-size", "100.0", 
				"-log-file", "/Users/syan/Documents/RBFNetwork_CfsSubsetEval_GreedyStepwise.txt", 
				"-initial-folds", "10", 
				"-subsequent-folds", "10", 
				"-num-slots", "8", 
				"-S", "1", 
				"-W", "weka.classifiers.meta.AttributeSelectedClassifier", "-do-not-check-capabilities", "--", 
				"-E", "weka.attributeSelection.CfsSubsetEval -P 1 -E 1", 
				"-S", "weka.attributeSelection.GreedyStepwise -T -1.7976931348623157E308 -N -1 -num-slots 1",
				"-W", "weka.classifiers.functions.RBFNetwork", "-do-not-check-capabilities", "--", "-R", "1.0E-6", "-M", "-1"
			},
			
			{	// RBFNetwork + CorrelationAttributeEval + Ranker
				"-E", "ACC", 
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.maxIts -min -1 -max 9.0 -step 1.0 -base 10.0 -expression I",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.ridge -min -8.0 -max 4 -step 2.0 -base 10.0 -expression pow(BASE,I)",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.minStdDev -min 0.1 -max 0.5 -step 0.1 -base 10.0 -expression I", 
				"-sample-size", "100.0", 
				"-log-file", "/Users/syan/Documents/RBFNetwork_CorrelationAttributeEval_Ranker.txt", 
				"-initial-folds", "10", 
				"-subsequent-folds", "10", 
				"-num-slots", "8", 
				"-S", "1", 
				"-W", "weka.classifiers.meta.AttributeSelectedClassifier", "-do-not-check-capabilities", "--", 
				"-E", "weka.attributeSelection.CorrelationAttributeEval ", 
				"-S", "weka.attributeSelection.Ranker -T -1.7976931348623157E308 -N -1",
				"-W", "weka.classifiers.functions.RBFNetwork", "-do-not-check-capabilities", "--", "-R", "1.0E-6", "-M", "-1"
			},
			
			{	// RBFNetwork + GainRatioAttributeEval + Ranker
				"-E", "ACC", 
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.maxIts -min -1 -max 9.0 -step 1.0 -base 10.0 -expression I",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.ridge -min -8.0 -max 4 -step 2.0 -base 10.0 -expression pow(BASE,I)",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.minStdDev -min 0.1 -max 0.5 -step 0.1 -base 10.0 -expression I", 
				"-sample-size", "100.0", 
				"-log-file", "/Users/syan/Documents/RBFNetwork_GainRatioAttributeEval_Ranker.txt", 
				"-initial-folds", "10", 
				"-subsequent-folds", "10", 
				"-num-slots", "8", 
				"-S", "1", 
				"-W", "weka.classifiers.meta.AttributeSelectedClassifier", "-do-not-check-capabilities", "--", 
				"-E", "weka.attributeSelection.GainRatioAttributeEval ", 
				"-S", "weka.attributeSelection.Ranker -T -1.7976931348623157E308 -N -1",
				"-W", "weka.classifiers.functions.RBFNetwork", "-do-not-check-capabilities", "--", "-R", "1.0E-6", "-M", "-1"
			},
			
			{	// RBFNetwork + InfoGainAttributeEval + Ranker
				"-E", "ACC", 
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.maxIts -min -1 -max 9.0 -step 1.0 -base 10.0 -expression I",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.ridge -min -8.0 -max 4 -step 2.0 -base 10.0 -expression pow(BASE,I)",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.minStdDev -min 0.1 -max 0.5 -step 0.1 -base 10.0 -expression I", 
				"-sample-size", "100.0", 
				"-log-file", "/Users/syan/Documents/RBFNetwork_InfoGainAttributeEval_Ranker.txt", 
				"-initial-folds", "10", 
				"-subsequent-folds", "10", 
				"-num-slots", "8", 
				"-S", "1", 
				"-W", "weka.classifiers.meta.AttributeSelectedClassifier", "-do-not-check-capabilities", "--", 
				"-E", "weka.attributeSelection.InfoGainAttributeEval ", 
				"-S", "weka.attributeSelection.Ranker -T -1.7976931348623157E308 -N -1",
				"-W", "weka.classifiers.functions.RBFNetwork", "-do-not-check-capabilities", "--", "-R", "1.0E-6", "-M", "-1"
			},
			
			{	// RBFNetwork + OneRAttributeEval + Ranker
				"-E", "ACC", 
				"-search", "weka.core.setupgenerator.ListParameter -property classifier.evaluator.minimumBucketSize -list \"6 10 15 20 30 50 100\"",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.maxIts -min -1 -max 9.0 -step 1.0 -base 10.0 -expression I",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.ridge -min -8.0 -max 4 -step 2.0 -base 10.0 -expression pow(BASE,I)",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.minStdDev -min 0.1 -max 0.5 -step 0.1 -base 10.0 -expression I", 
				"-sample-size", "100.0", 
				"-log-file", "/Users/syan/Documents/RBFNetwork_OneRAttributeEval_Ranker.txt", 
				"-initial-folds", "10", 
				"-subsequent-folds", "10", 
				"-num-slots", "8", 
				"-S", "1", 
				"-W", "weka.classifiers.meta.AttributeSelectedClassifier", "-do-not-check-capabilities", "--", 
				"-E", "weka.attributeSelection.OneRAttributeEval -S 1 -F 10 -B 6", 
				"-S", "weka.attributeSelection.Ranker -T -1.7976931348623157E308 -N -1",
				"-W", "weka.classifiers.functions.RBFNetwork", "-do-not-check-capabilities", "--", "-R", "1.0E-6", "-M", "-1"
			},
			
			{	// RBFNetwork + PrincipalComponents + Ranker
				"-E", "ACC", 
				"-search", "weka.core.setupgenerator.ListParameter -property classifier.evaluator.centerData -list \"True False\"",
				"-search", "weka.core.setupgenerator.ListParameter -property classifier.evaluator.transformBackToOriginal -list \"True False\"",
				"-search", "weka.core.setupgenerator.ListParameter -property classifier.evaluator.maximumAttributeNames -list \"5 8 12 16 20 30 40 50\"",
				"-search", "weka.core.setupgenerator.ListParameter -property classifier.evaluator.varianceCovered -list \"0.6 0.8 0.9 0.95 0.99\"",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.maxIts -min -1 -max 9.0 -step 1.0 -base 10.0 -expression I",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.ridge -min -8.0 -max 4 -step 2.0 -base 10.0 -expression pow(BASE,I)",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.minStdDev -min 0.1 -max 0.5 -step 0.1 -base 10.0 -expression I", 
				"-sample-size", "100.0", 
				"-log-file", "/Users/syan/Documents/RBFNetwork_PrincipalComponents_Ranker.txt", 
				"-initial-folds", "10", 
				"-subsequent-folds", "10", 
				"-num-slots", "8", 
				"-S", "1", 
				"-W", "weka.classifiers.meta.AttributeSelectedClassifier", "-do-not-check-capabilities", "--", 
				"-E", "weka.attributeSelection.PrincipalComponents -R 0.95 -A 5", 
				"-S", "weka.attributeSelection.Ranker -T -1.7976931348623157E308 -N -1",
				"-W", "weka.classifiers.functions.RBFNetwork", "-do-not-check-capabilities", "--", "-R", "1.0E-6", "-M", "-1"
			},
			
			{	// RBFNetwork + ReliefFAttributeEval + Ranker
				"-E", "ACC", 
				"-search", "weka.core.setupgenerator.ListParameter -property classifier.evaluator.numNeighbours -list \"5 10 15 20 30 50\"",
				"-search", "weka.core.setupgenerator.ListParameter -property classifier.evaluator.weightByDistance -list \"True False\"",
				"-search", "weka.core.setupgenerator.ListParameter -property classifier.evaluator.sigma -list \"1 2 3 4 6 10\"",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.maxIts -min -1 -max 9.0 -step 1.0 -base 10.0 -expression I",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.ridge -min -8.0 -max 4 -step 2.0 -base 10.0 -expression pow(BASE,I)",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.minStdDev -min 0.1 -max 0.5 -step 0.1 -base 10.0 -expression I", 
				"-sample-size", "100.0", 
				"-log-file", "/Users/syan/Documents/RBFNetwork_ReliefFAttributeEval_Ranker.txt", 
				"-initial-folds", "10", 
				"-subsequent-folds", "10", 
				"-num-slots", "8", 
				"-S", "1", 
				"-W", "weka.classifiers.meta.AttributeSelectedClassifier", "-do-not-check-capabilities", "--", 
				"-E", "weka.attributeSelection.ReliefFAttributeEval -M -1 -D 1 -K 10", 
				"-S", "weka.attributeSelection.Ranker -T -1.7976931348623157E308 -N -1",
				"-W", "weka.classifiers.functions.RBFNetwork", "-do-not-check-capabilities", "--", "-R", "1.0E-6", "-M", "-1"
			},
			
			{	// RBFNetwork + SymmetricalUncertAttributeEval + Ranker
				"-E", "ACC", 
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.maxIts -min -1 -max 9.0 -step 1.0 -base 10.0 -expression I",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.ridge -min -8.0 -max 4 -step 2.0 -base 10.0 -expression pow(BASE,I)",
				"-search", "weka.core.setupgenerator.MathParameter -property classifier.classifier.minStdDev -min 0.1 -max 0.5 -step 0.1 -base 10.0 -expression I", 
				"-sample-size", "100.0", 
				"-log-file", "/Users/syan/Documents/RBFNetwork_SymmetricalUncertAttributeEval_BestFirst.txt", 
				"-initial-folds", "10", 
				"-subsequent-folds", "10", 
				"-num-slots", "8", 
				"-S", "1", 
				"-W", "weka.classifiers.meta.AttributeSelectedClassifier", "-do-not-check-capabilities", "--", 
				"-E", "weka.attributeSelection.CorrelationAttributeEval ", 
				"-S", "weka.attributeSelection.Ranker -T -1.7976931348623157E308 -N -1",
				"-W", "weka.classifiers.functions.RBFNetwork", "-do-not-check-capabilities", "--", "-R", "1.0E-6", "-M", "-1"
			}
		};
		
		System.out.println("Testing...");
		multisearch.setOptions(parameters[0]);
		multisearch.buildClassifier(data_train);
		System.out.println("Done");
	}
}