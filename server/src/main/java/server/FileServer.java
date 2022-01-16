package server;



import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.io.IOException;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class FileServer {

    public static String onDeviceModelPath = "res/clientModel";
    public static Map<Integer, Map<String, INDArray>> cache = new HashMap<>();
    public static Map<Integer, Integer> map = new HashMap<>();
    public static FederatedModel federatedmodel = new FederatedModel();
    private static ServerSocket serverSocket;
    public static int clientNum = 4;

    private void init(int port, int timeout) {
        try {
            serverSocket = new ServerSocket(port);
            serverSocket.setSoTimeout(timeout);
        } catch (Exception e) {
            System.out.println("Couldn't open socket. " + e.getMessage());
            return;
        }
        System.out.println("Server started on port " + port + " with timeout " + timeout + "ms");
    }

    public void run() throws InterruptedException {


        int curID = 1;
        map.put(1, 0);
        map.put(2, 0);
        map.put(3, 0);
        map.put(4, 0);
        // here plus 2
        while ((curID < (clientNum + 1))) {
            try {
                Socket clientSocket = serverSocket.accept();
                Thread t1 = new Thread((Runnable) new ServerConnection(clientSocket, curID));
                t1.start();
                System.out.println("client " + curID + " connected!");
                System.out.println(map);
                t1.join();
                curID++;
            } catch (IOException e) {
                System.out.println("Error accepting client connection: " + e.getMessage());
            }

        }

        System.out.println("exit the while loop");
    }

    public static void fedAvgRun() throws InterruptedException {
        int i = 1;
        if ((map.get(i) == 1) & (map.get(i+1) == 1) & (map.get(i+2) == 1)  & (map.get(i+3) == 1)){
            System.out.println(map.get(i+1));
            System.out.println("the cache size is " + cache.size());
            map.clear();
            try {
                map.clear();
               federatedmodel.fedavg(2, cache);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        else{
            System.out.println(map.size());
        }
    }


    public static void main(String[] args) throws IOException, InterruptedException {
        // Run server
        int DEFAULT_PORT = 8000;
        int DEFAULT_TIMEOUT = 60 * 1000;//30 seconds
        int round = 5;

        //to do: select client
        federatedmodel.initModel();

        FileServer fileserver = new FileServer();
        fileserver.init(DEFAULT_PORT, DEFAULT_TIMEOUT);

        for (int r = 0; r < round; r++) {
            System.out.println("\n\nround:" + r);
            fileserver.run();
            try {
                FileServer.fedAvgRun();
            }
            catch (InterruptedException e) {
                e.printStackTrace();
            }

            }
        }

}


