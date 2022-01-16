import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.deeplearning4j.util.ModelSerializer;

//import org.deeplearning4j.api.storage.StatsStorage;
//import org.deeplearning4j.ui.storage.FileStatsStorage;
//import org.deeplearning4j.ui.api.UIServer;
//import org.deeplearning4j.ui.stats.StatsListener;


import java.io.File;

public class baselinemodel {

    public static void main(String[] args) throws Exception {

        int seed = 100;
        double learningRate = 0.001;
        int batchSize = 50;
        int nEpochs = 5;
//        int batchSize = 10;
//        int nEpochs = 1;

//        int numInputs = 45;
//        int numOutputs = 10;
//        int numHiddenNodes = 1000;

        int numInputs = 561;
        int numOutputs = 6;
        int numHiddenNodes = 1000;


//        final String filenameTrain = "res/dataset/train.csv";
//        final String filenameTest = "res/dataset/test.csv";
        final String filenameTrain = "res/UCI-HAR/train_final.csv";
        final String filenameTest = "res/UCI-HAR/test_final.csv";

        //Load the training data:
        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new File(filenameTrain)));
        DataSetIterator trainIter = new RecordReaderDataSetIterator(rr, batchSize, 561, 6);

        //Load the test/evaluation data:
        RecordReader rrTest = new CSVRecordReader();
        rrTest.initialize(new FileSplit(new File(filenameTest)));
        DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest, batchSize, 561, 6);
        System.out.println("data finish");

        // https://deeplearning4j.org/docs/latest/deeplearning4j-nn-multilayernetwork
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(learningRate, 0.9))
                .list()
                .layer(new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                        .activation(Activation.RELU)
                        .build())
                .layer(new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .nIn(numHiddenNodes).nOut(numOutputs).build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        System.out.println("config finish");

        model.init();
        model.setListeners(new ScoreIterationListener(10));  //Print score every 10 parameter updates
        System.out.println("init finish");

        model.fit(trainIter, nEpochs);
        System.out.println("fit finish");
//        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork("res/model/trained_nn.zip");


        System.out.println("Evaluate model....");
        Evaluation eval = new Evaluation(numOutputs);
        while (testIter.hasNext()) {
            DataSet t = testIter.next();
            INDArray features = t.getFeatures();
            INDArray labels = t.getLabels();
            INDArray predicted = model.output(features, false);
            System.out.println("label:");
            System.out.println(labels);

            System.out.println(predicted);
            System.out.println("\n\n");
            eval.eval(labels, predicted);
        }
        // Print the evaluation statistics
        System.out.println(eval.stats());

        // save model
        boolean saveUpdate = true;
        File locationToSave = new File("res/serverModel/trained_nn.zip");
        ModelSerializer.writeModel(model, locationToSave, saveUpdate);
        System.out.println("save model finish");
    }


}
