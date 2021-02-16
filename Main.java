package ehu.weka;



import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.ChebyshevDistance;
import weka.core.EuclideanDistance;
import weka.core.FilteredDistance;
import weka.core.Instances;
import weka.core.ManhattanDistance;
import weka.core.MinkowskiDistance;
import weka.core.SelectedTag;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;




public class Main {
	public static void main(String[] args) throws Exception {
		
			Double maxpct = 0.0;
			int maxi=0;
			int maxj=0;
			int maxz=0;
			DataSource loader = new DataSource(args[0]);
			Instances data = loader.getDataSet();
			data.setClassIndex(data.numAttributes()-1);
			/*Randomize randomize = new Randomize();
			randomize.setSeed(941999);
			randomize.setInputFormat(data);
			Instances randomData = Filter.useFilter(data, randomize);
			RemovePercentage rmp = new RemovePercentage();
			rmp.setInputFormat(randomData);
			rmp.setPercentage(30);
			Instances train = Filter.useFilter(randomData, rmp);
			
			rmp = new RemovePercentage();
			rmp.setInputFormat(randomData);
			rmp.setPercentage(30);
			rmp.setInvertSelection(true);
			Instances test = Filter.useFilter(randomData, rmp);
			System.out.println(test.numInstances());
			System.out.println(train.numInstances());
			//Datuak
			train.setClassIndex(train.numAttributes()-1);
			test.setClassIndex(test.numAttributes()-1);
			*/
			System.out.println(data.classAttribute().numValues());
			FileWriter fw = new FileWriter("errors.txt");
			SelectedTag sim = new SelectedTag(IBk.WEIGHT_SIMILARITY, IBk.TAGS_WEIGHTING);
			SelectedTag none = new SelectedTag(IBk.WEIGHT_NONE, IBk.TAGS_WEIGHTING);
			SelectedTag inverse = new SelectedTag(IBk.WEIGHT_INVERSE, IBk.TAGS_WEIGHTING);
			ManhattanDistance manDistance = new ManhattanDistance();
			EuclideanDistance euDistance = new EuclideanDistance();
			ChebyshevDistance cheDistance = new ChebyshevDistance();
			FilteredDistance filDistance = new FilteredDistance();
			MinkowskiDistance minkDistance = new MinkowskiDistance();

			for(int i = 1; i<data.numInstances();i++) { 
				for(int j = 1; j<6; j++) {
					IBk ibk = new IBk();
					ibk.setKNN(i);
					switch(j) {
					case 1:
						ibk.getNearestNeighbourSearchAlgorithm().setDistanceFunction(manDistance);
						break;
					case 2:
						ibk.getNearestNeighbourSearchAlgorithm().setDistanceFunction(euDistance);
						break;
					case 3:
						ibk.getNearestNeighbourSearchAlgorithm().setDistanceFunction(cheDistance);
						break;
					case 4: 
						ibk.getNearestNeighbourSearchAlgorithm().setDistanceFunction(filDistance);
						break;
					case 5: 
						ibk.getNearestNeighbourSearchAlgorithm().setDistanceFunction(minkDistance);
						break;
					default:
			    		break;
					}
					for(int z= 1; z<4; z++) {
						switch(z) {
						case 1:
							ibk.setDistanceWeighting(inverse);
							break;
						case 2:
							ibk.setDistanceWeighting(none);
							break;
						case 3:
							ibk.setDistanceWeighting(sim);
							break;
						default:
				    		break;
						}try {
							Evaluation eval = new Evaluation(data);
							eval.crossValidateModel(ibk, data, 10, new Random(1));
							System.out.println("\nCaso actual i,j,z "+i+","+j+","+z);
							if(eval.pctCorrect()>maxpct) {
								maxpct=eval.pctCorrect();
								maxi=i;
								maxj=j;
								maxz=z;
							}
							
						}catch(Exception e) {
							StringWriter sw = new StringWriter();
							PrintWriter pw = new PrintWriter(sw);
							e.printStackTrace(pw);
							String sStackTrace = sw.toString(); 
							fw.write("\nCaso actual i,j,z "+i+","+j+","+z);
							fw.write(sStackTrace);
						}

					}

				}

			}

			fw.close();
			System.out.println("Casu hoberenean pctCorrect " + maxpct + " izan da");
			System.out.println("i = " + maxi + ", j = "+ maxj + ", z = "+ maxz);

	}
}