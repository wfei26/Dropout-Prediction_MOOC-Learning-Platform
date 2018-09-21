package stage3;

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
 
public class AucTest {

	public static void main(String[] args) throws Exception {

		CSVLoader loader_train = new CSVLoader();
		loader_train.setSource(new File("EECS738_Train_reduced.csv"));
		
		// Have weka process each buffer as a "data instance"
		Instances data_train = loader_train.getDataSet();
		
		// Tell weka that the second column contains the class labels
		data_train.setClassIndex(0);
		
		// generate attribute-selected data through filters
		AttributeSelection filter = new AttributeSelection();
		
		ASEvaluation[] evaluators = {
				new CfsSubsetEval(),
				new CorrelationAttributeEval(),
				new GainRatioAttributeEval(),
				new InfoGainAttributeEval(),
				new OneRAttributeEval(),
				new PrincipalComponents(),
				new ReliefFAttributeEval(),
				new SymmetricalUncertAttributeEval()
		};
		
		BestFirst bf = new BestFirst();
		GreedyStepwise gsw = new GreedyStepwise();
		Ranker rk = new Ranker();
		
		Instances[] data = new Instances[9];
		
		for (int i = 0; i < evaluators.length; i++) {
			if (i == 0) {
				filter.setEvaluator(evaluators[i]);
				filter.setSearch(bf);
				filter.setInputFormat(data_train);
				data[0] = Filter.useFilter(data_train, filter);
				
				filter.setEvaluator(evaluators[i]);
				filter.setSearch(gsw);
				filter.setInputFormat(data_train);
				data[1] = Filter.useFilter(data_train, filter);
			}
			else {
				filter.setEvaluator(evaluators[i]);
				filter.setSearch(rk);
				filter.setInputFormat(data_train);
				data[i + 1] = Filter.useFilter(data_train, filter);
			}
		}
		
		// Use a set of classifiers
		Logistic log = new Logistic();
		RBFNetwork rbf = new RBFNetwork();
		RandomForest rf = new RandomForest();
		
		String log_params = "-R 1.0E-4 -M -1 -do-not-check-capabilities";
		String rbf_params = "-B 200 -S 1 -R 1.0E2 -M -1 -W 0.7 -do-not-check-capabilities";
		String rf_params = "-I 200 -K 0 -S 14 -num-slots 4 -do-not-check-capabilities";

		// Run each model on each filtered data
		for (int i = 0; i < data.length; i++) {
			log.setOptions(weka.core.Utils.splitOptions(log_params));
			log.buildClassifier(data[i]);	
			
			Random rand = new Random(1);
			Evaluation eval = new Evaluation(data[i]);

			eval.crossValidateModel(log, data[i], 10, rand);
			
			System.out.println("==============================================================================================\n"
					+ "Classifier: Logistic, params: " + log_params + ", Dataset Index: "+ (i + 1));
			System.out.println(eval.toClassDetailsString());
			System.out.println("==============================================================================================\n\n");
		}
		
		// Run each model on each filtered data
		for (int i = 0; i < data.length; i++) {
			rf.setOptions(weka.core.Utils.splitOptions(rf_params));
			rf.buildClassifier(data[i]);	
			
			Random rand = new Random(1);
			Evaluation eval = new Evaluation(data[i]);

			eval.crossValidateModel(rf, data[i], 10, rand);
					
			System.out.println("==============================================================================================\n"
					+ "Classifier: RandomForest, params: " + rf_params + ", Dataset Index: "+ (i + 1));
			System.out.println(eval.toClassDetailsString());
			System.out.println("==============================================================================================\n\n");
		}
		
		// Run each model on each filtered data
		for (int i = 0; i < data.length; i++) {
			rbf.setOptions(weka.core.Utils.splitOptions(rbf_params));
			rbf.buildClassifier(data[i]);	
					
			Random rand = new Random(1);
			Evaluation eval = new Evaluation(data[i]);

			eval.crossValidateModel(rbf, data[i], 10, rand);
							
			System.out.println("==============================================================================================\n"
					+ "Classifier: RBFNetwork, params: " + rbf_params + ", Dataset Index: "+ (i + 1));
			System.out.println(eval.toClassDetailsString());
			System.out.println("==============================================================================================\n\n");
		}
	}
}