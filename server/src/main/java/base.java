import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Random;

import ch.qos.logback.classic.BasicConfigurator;
import server.FileServer;

public class base {

    public static int numOutputs = 6;
    //public static int batchSize = 16;
    private static final int HEIGHT = 50;
    private static final int WIDTH = 50;
    private static final int CHANNELS = 3;
    private static final int N_OUTCOMES = 6;
    private static long t0 = System.currentTimeMillis();

    public static final String filenameTrain = "C:\\Users\\souvik\\Downloads\\dataset-resized\\dataset-resized\\trashnetdataset\\data_c\\train\\";
    public static final String filenameTest = "C:\\Users\\souvik\\Downloads\\dev\\FLDrone_gray\\server\\res\\data_c\\test\\";


    private static DataSetIterator getDataSetIterator(String folderPath) throws IOException {
        try {
            File folder = new File(folderPath);
            File[] digitFolders = folder.listFiles();
            int count = 0;
            for (File sublist : digitFolders) {
                File[] iF = sublist.listFiles();
                for (File img : iF) {
                    count = count + 1;
                }
                System.out.println("The :" + count);
            }
            int nSamples = count;


            NativeImageLoader nativeImageLoader = new NativeImageLoader(HEIGHT, WIDTH, CHANNELS);
            ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
            INDArray input = Nd4j.create(new int[]{nSamples, HEIGHT * WIDTH * CHANNELS});
            INDArray output = Nd4j.create(new int[]{nSamples, N_OUTCOMES});

            int n = 0;
            //scan all 0 to 9 digit subfolders
            for (File digitFolder : digitFolders) {

//                System.out.println("the name is " + digitFolder.getName());
                int labelDigit = Integer.parseInt(digitFolder.getName());
                File[] imageFiles = digitFolder.listFiles();

                for (File imgFile : imageFiles) {
                    INDArray img = nativeImageLoader.asRowVector(imgFile);
                    scaler.transform(img);
                    input.putRow(n, img);
                    output.put(n, labelDigit, 1.0);
                    n++;
                }
            }//End of For-loop

            //Joining input and output matrices into a dataset
            DataSet dataSet = new DataSet(input, output);
            //Convert the dataset into a list
            List<DataSet> listDataSet = dataSet.asList();
            //Shuffle content of list randomly
            Collections.shuffle(listDataSet, new Random(System.currentTimeMillis()));
            //int batchSize = 50;
            int batchSize = 16;
            //Build and return a dataset iterator
            DataSetIterator dsi = new ListDataSetIterator<DataSet>(listDataSet, batchSize);
            return dsi;
        } catch (Exception e) {
            System.out.println(e.getLocalizedMessage());
            return null;
        }
    } //End of DataIterator Method
    //Load the training data:

    public static void main(String[] args) throws Exception {

      //  BasicConfigurator.configure();

        t0 = System.currentTimeMillis();
        //System.out.print(RESOURCES_FOLDER_PATH + "/training");
        DataSetIterator dataSetIterator = getDataSetIterator(filenameTrain);

        buildModel(dataSetIterator);


    }

    private static void buildModel(DataSetIterator dsi) throws IOException {
        int seed = 100;
        int nEpochs = 50;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .l2(0.0005)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(1e-3))
                .list()
                .layer(new ConvolutionLayer.Builder(3, 3)
                        //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
                        .nIn(CHANNELS)
                        .stride(1, 1)
                        .nOut(28)
                        .activation(Activation.LEAKYRELU)
                        .build())
                .layer(new SubsamplingLayer.Builder(PoolingType.MAX)
                        .kernelSize(2, 2)
                        .build())
                .layer(new ConvolutionLayer.Builder(3, 3)
                        //Note that nIn need not be specified in later layers
                        .stride(1, 1)
                        .nOut(56)
                        .activation(Activation.LEAKYRELU)
                        .build())
                .layer(new SubsamplingLayer.Builder(PoolingType.MAX)
                        .kernelSize(2, 2)
                        .build())
                .layer(new ConvolutionLayer.Builder(3, 3)
                        //Note that nIn need not be specified in later layers
                        .stride(1, 1)
                        .nOut(56)
                        .activation(Activation.LEAKYRELU)
                        .build())
                .layer(new SubsamplingLayer.Builder(PoolingType.MAX)
                        .kernelSize(2, 2)
                        .build())
                .layer(new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(56).build())

                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY)
                        .nOut(numOutputs)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(HEIGHT, WIDTH, CHANNELS)) //See note below
                .build();


        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        //Print score every 500 interaction
        model.setListeners(new ScoreIterationListener(10));

        System.out.print("Train Model...");
        model.fit(dsi,nEpochs);

        //Evaluation
        DataSetIterator testDsi = getDataSetIterator(filenameTest);
        System.out.print("Evaluating Model...");
        Evaluation eval = model.evaluate(testDsi);
        System.out.print(eval.stats());

        long t1 = System.currentTimeMillis();
        double t = (double)(t1-t0)/1000.0;
        System.out.print("\n\nTotal time: "+t+" seconds");


    }

}