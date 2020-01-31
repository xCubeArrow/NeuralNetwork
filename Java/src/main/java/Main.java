import org.nd4j.linalg.factory.Nd4j;

public class Main {


    public static void main(String[] args) {
        int[] layers = {2, 2, 1};
        float[][][] inputs = {{{0, 0}},
                {{0, 1}},
                {{1, 0}},
                {{1, 1}}};
        float[][][] targets = {{{0}},
                {{1}},
                {{1}},
                {{0}}
        };

        NeuralNetwork nn = new NeuralNetwork(layers);
        nn.setActivationFunction(NeuralNetwork::sigmoid);
        nn.setdActivationFunction(NeuralNetwork::dSigmoid);
        nn.setLearningRate(0.1f);

        try {
            nn.trainThreads(inputs, targets, 10000, 6);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        nn.test(inputs);

    }
}
