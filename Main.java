package ehu.weka;



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
			DataSource loader = new DataSource("balance-scale.arff");
			Instances data = loader.getDataSet();
			Randomize randomize = new Randomize();
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

			SelectedTag sim = new SelectedTag(IBk.WEIGHT_SIMILARITY, IBk.TAGS_WEIGHTING);
			SelectedTag none = new SelectedTag(IBk.WEIGHT_NONE, IBk.TAGS_WEIGHTING);
			SelectedTag inverse = new SelectedTag(IBk.WEIGHT_INVERSE, IBk.TAGS_WEIGHTING);
			ManhattanDistance manDistance = new ManhattanDistance();
			EuclideanDistance euDistance = new EuclideanDistance();
			ChebyshevDistance cheDistance = new ChebyshevDistance();
			FilteredDistance filDistance = new FilteredDistance();
			MinkowskiDistance minkDistance = new MinkowskiDistance();

			for(int i = 1; i<train.numInstances()-1;i++) { //1-en hasten bagara 100 pctCorrect dago
				for(int j = 2; j<6; j++) {
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
						System.out.println("Caso actual i,j,z "+i+","+j+","+z);
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
						}
						ibk.buildClassifier(train);
						Evaluation eval = new Evaluation(train);
						eval.evaluateModel(ibk, test);
						if(eval.pctCorrect()>maxpct) {
							maxpct=eval.pctCorrect();
							maxi=i;
							maxj=j;
							maxz=z;
						}
						
						if(i%25== 0 && j == 5 && z == 3) {
							System.out.println("asu hoberenean pctCorrect " + maxpct + " izan da");
							System.out.println("i = " + maxi + ", j = "+ maxj + ", z = "+ maxz);		 
						}
					}

				}

			}


			System.out.println("Casu hoberenean pctCorrect " + maxpct + " izan da");
			System.out.println("i = " + maxi + ", j = "+ maxj + ", z = "+ maxz);

	}
}