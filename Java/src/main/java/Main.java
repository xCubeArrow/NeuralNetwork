public class Main {


    public static void main(String[] args) {
        int[] layers = {2, 5, 5, 1};
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

        nn.train(inputs, targets, 2100000000);
        nn.test(inputs);

    }
}
