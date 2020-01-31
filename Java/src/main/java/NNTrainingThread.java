import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.function.Function;

public class NNTrainingThread extends Thread {
    protected final NeuralNetwork nn;
    INDArray[] inputs;
    INDArray[] targets;
    int iterations;
    public NNTrainingThread(NeuralNetwork nn, int iterations, INDArray[] inputs, INDArray[] targets) {
        this.nn = nn;
        this.inputs = inputs;
        this.targets = targets;
        this.iterations = iterations;
    }


    @Override
    public void run() {
        nn.train(this.inputs, this.targets, this.iterations);
        System.out.println(String.format("Thread %s finished.", this.getId()));
    }
}
