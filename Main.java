package ehu.weka;


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
import weka.core.converters.ConverterUtils;
import weka.core.neighboursearch.LinearNNSearch;




public class Main {
	public static void main(String[] args) throws Exception {
	
            ConverterUtils.DataSource source = new ConverterUtils.DataSource(args[0]);
            Instances data = source.getDataSet();
            data.setClassIndex(data.numAttributes()-1);

            LinearNNSearch manDistance = new LinearNNSearch();
            manDistance.setDistanceFunction(new ManhattanDistance());
            LinearNNSearch euDistance = new LinearNNSearch();
            euDistance.setDistanceFunction(new EuclideanDistance());
            LinearNNSearch cheDistance = new LinearNNSearch();
            cheDistance.setDistanceFunction(new ChebyshevDistance());
            LinearNNSearch filDistance = new LinearNNSearch();
            filDistance.setDistanceFunction(new FilteredDistance());
            LinearNNSearch minkDistance = new LinearNNSearch();
            MinkowskiDistance minko = new MinkowskiDistance();
            minko.setOrder(3.0);
            minkDistance.setDistanceFunction(minko);

            
            LinearNNSearch[] distances = new LinearNNSearch[]{manDistance, euDistance, cheDistance, filDistance, minkDistance};

            SelectedTag[] tags = new SelectedTag[]{new SelectedTag(IBk.WEIGHT_NONE, IBk.TAGS_WEIGHTING), new SelectedTag(IBk.WEIGHT_INVERSE, IBk.TAGS_WEIGHTING), new SelectedTag(IBk.WEIGHT_SIMILARITY, IBk.TAGS_WEIGHTING)};

            Evaluation eval = null;
            IBk ibk = new IBk();
            Double maxf = 0.0;
            int maxk = 0;
            Evaluation maxeval = new Evaluation(data);
            LinearNNSearch maxd;
            SelectedTag maxw= null;
            int index = 0;
            int maxindex = 0;
            for (int k = 1; k <= data.numInstances()*0.9; k++) {
                ibk.setKNN(k);
                for (LinearNNSearch d : distances) {
                    if(index==distances.length){
                        index = 0;
                    }
                    ibk.setNearestNeighbourSearchAlgorithm(d);
                	int pisua = 0;
                    for (SelectedTag w : tags) {
                    	try {
                    		eval = new Evaluation(data);
                    		ibk.setDistanceWeighting(w);
                    		eval.crossValidateModel(ibk, data, 10, new Random());
                    		Double fmeasure = eval.weightedFMeasure();
                    		System.out.println("k = " + k + " distantzia = " + printeatudistantzia(index) + " eta pisua = " + pisua(pisua));
                    		pisua++;
                    		if (maxf < fmeasure) {
                    			maxf = fmeasure;
                    			maxk = k;
                    			maxd = d;
                    			maxw = w;
                    			maxindex = index;
                    			maxeval=eval;
                    		}
                    		
                    	}catch(Exception e) {
                    		System.out.println("k = " + k + " distantzia = " + printeatudistantzia(index) + " eta pisua = " + w + " ERROREA");
                    	}
               
                    }
                    index ++;
                }
            }

            System.out.println("EMAITZAK: k-ren balio optimoa = " + maxk + " da, " + printeatudistantzia(maxindex)+ " distantziarekin eta " + maxw + " pisuarekin non fmeasure = " + maxf + " den.");
            System.out.println(maxeval.toMatrixString());
    }



	private static String pisua(int pisua) {
		switch(pisua) {
		case 0:
			return "None";
		case 1: 
			return "Inverse";
		case 2:
			return "Similarity";
		}
		return null;
	}



	private static String printeatudistantzia(int i) {
        String emaitza ="";
        switch (i){
            case 0:
                emaitza = "Manhattan";
                break;
            case 1:
                emaitza = "Euclidean";
                break;
            case 2:
                emaitza = "Chevishev";
                break;
            case 3:
                emaitza = "Filtered";
                break;
            case 4:
                emaitza = "Minkowski";
                break;
            default:
                emaitza = "Manhattan";
                break;
        }
        return emaitza;
    
	}
}