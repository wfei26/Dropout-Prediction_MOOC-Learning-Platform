import java.io.File;
import java.util.Random;

import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.CorrelationAttributeEval;
import weka.attributeSelection.GainRatioAttributeEval;
import weka.attributeSelection.GreedyStepwise;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.OneRAttributeEval;
import weka.attributeSelection.PrincipalComponents;
import weka.attributeSelection.Ranker;
import weka.attributeSelection.ReliefFAttributeEval;
import weka.attributeSelection.SymmetricalUncertAttributeEval;
import weka.classifiers.Evaluation;

import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.RBFNetwork;
import weka.classifiers.trees.RandomForest;

import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
 
public class MCC_AUC {

	public static void main(String[] args) throws Exception {

		CSVLoader loader_train = new CSVLoader();
		loader_train.setSource(new File("EECS738_Train.csv"));
		
		// Have weka process each buffer as a "data instance"
		Instances data_train = loader_train.getDataSet();
		
		// Tell weka that the second column contains the class labels
		data_train.setClassIndex(0);
				
		// Use a set of classifiers
		RandomForest rf = new RandomForest();
		
		String [] numTrees = { "50" , "100", "150", "200" };
		String [] numFeature = { "10", "12", "14", "16", "18", "20" }; 
		String rf_param_1 = "-I ";
		String rf_param_2 = " -K 0 -S ";
		String rf_param_3 = " -num-slots 16 -do-not-check-capabilities";
		
		// Run each model 20 times on data
		for (int i1 = 0; i1 < numTrees.length; i1++) {
			for (int i2 = 0; i2 < numFeature.length; i2++) {
				for (int i = 0; i < 20; i++) {
					rf.setOptions(weka.core.Utils.splitOptions(
						rf_param_1 + numTrees[i1] + rf_param_2 + numFeature[i2] + rf_param_3));
					rf.buildClassifier(data_train);

					Random rand = new Random(i);
					Evaluation eval = new Evaluation(data_train);

					eval.crossValidateModel(rf, data_train, 10, rand);
			
					System.out.println("\n"	+ "Round " + (i+1) +", numTrees = " + numTrees[i1] + ", numFeature = " + numFeature[i2] + "\n");
					System.out.println(eval.toClassDetailsString());
					System.out.println("\n-----------------------------------------------------\n\n");
				}
			}
			System.out.println("\n================================================================================================\n\n");
		}
	}
}
