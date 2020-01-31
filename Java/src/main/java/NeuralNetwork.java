import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Arrays;
import java.util.Random;
import java.util.function.Function;

public class NeuralNetwork {

    Function activationFunction;
    Function dActivationFunction;
    float learningRate;
    INDArray[] weights;
    INDArray[] biases;
    INDArray[] out;
    INDArray[] net;
    public NeuralNetwork(Function activationFunction, Function dActivationFunction, float learningRate, INDArray[] weights, INDArray[] biases) {
        this.activationFunction = activationFunction;
        this.dActivationFunction = dActivationFunction;
        this.learningRate = learningRate;
        this.weights = weights;
        this.biases = biases;

        this.out = new INDArray[weights.length];
        this.net = new INDArray[weights.length];

    }

    public NeuralNetwork(int[] layers) {

        this.weights = new INDArray[layers.length - 1];
        this.biases = new INDArray[layers.length - 1];
        this.out = new INDArray[layers.length - 1];
        this.net = new INDArray[layers.length - 1];
        for (int i = 1; i < layers.length; i++) {
            this.weights[i - 1] = Nd4j.rand(layers[i - 1], layers[i]);
            this.biases[i - 1] = Nd4j.rand(1, layers[i]);
        }
    }

    public static Object sigmoid(Object x) {
        return (float) x / (1 + Math.abs((float) x));
    }

    public static Object dSigmoid(Object x) {
        return (float) (1 / (1 + Math.pow(Math.abs((float) x), 2)));
    }

    INDArray mapFunction(Function function, INDArray array) {
        INDArray result = Nd4j.create(array.rows(), array.columns());
        for (int i = 0; i < result.rows(); i++) {
            for (int j = 0; j < result.columns(); j++) {
                result.putScalar(i, j, (float) function.apply(array.getFloat(i, j)));
            }
        }
        return result;
    }

    public void setActivationFunction(Function activationFunction) {
        this.activationFunction = activationFunction;
    }


    public void setdActivationFunction(Function dActivationFunction) {
        this.dActivationFunction = dActivationFunction;
    }


    public void setLearningRate(float learningRate) {
        this.learningRate = learningRate;
    }

    public void feedForward(INDArray inputs) {
        INDArray layerInputs = inputs;
        for (int i = 0; i < weights.length; i++) {
            // Multiply the previous output with the current weights and add the bias
            INDArray x = layerInputs.mmul(this.weights[i]);
            x = x.add(this.biases[i]);

            // Save both the output and the output without the activation function
            this.net[i] = x;
            out[i] = Transforms.sigmoid(x);
            layerInputs = out[i];
        }
    }

    public void backPropagation(INDArray inputs, INDArray targets) {
        INDArray[] deltas = new INDArray[this.weights.length];
        // Calculate the delta of the output layer
        deltas[deltas.length - 1] = targets.sub(this.out[out.length - 1])
                .mmul(Transforms.sigmoidDerivative(this.net[net.length - 1]));

        // Calculate the deltas of the hidden layers using the previous delta
        for (int i = weights.length - 2; i >= 0; i--) {
            INDArray error = deltas[i + 1].mmul(this.weights[i + 1].transpose());
            deltas[i] = error.mul(Transforms.sigmoidDerivative(net[i]));

        }
        // Update the weights and the biases
        for (int i = 0; i < this.weights.length; i++) {
            // Set the previous output to the inputs if there is no previous outputs
            INDArray input = inputs;
            if (i != 0) {
                input = this.out[i - 1];
            }

            // Update both the weights and the biases
            this.weights[i] = weights[i].add(input.transpose().mmul(deltas[i]).mul(this.learningRate));
            this.biases[i] = biases[i].add(deltas[i].mul(this.learningRate));
        }
    }


    public void train(INDArray[] inputs, INDArray[] targets, int iterations) {
        for (int i = 0; i < iterations; i++) {

            int randInt = new Random().nextInt(inputs.length);
            this.feedForward(inputs[randInt]);
            this.backPropagation(inputs[randInt], targets[randInt]);

        }
    }

    public void train(float[][][] inputs, float[][][] targets, int iterations) {

        train(Utilities.floatArrayToMatrixArray(inputs), Utilities.floatArrayToMatrixArray(targets), iterations);
    }

    public void test(INDArray[] inputs) {
        for (INDArray input : inputs) {
            this.feedForward(input);
            System.out.println("Outputs: " + Arrays.deepToString(this.out[out.length - 1].toFloatMatrix()));

        }
    }

    public void test(float[][][] inputs) {
        test(Utilities.floatArrayToMatrixArray(inputs));
    }



    public void trainThreads(INDArray[] inputs, INDArray[] targets, int iterations, int n) throws InterruptedException {
        NNTrainingThread[] threads = new NNTrainingThread[n];
        for (int i = 0; i < n; i++) {
            threads[i] = new NNTrainingThread(this, iterations, inputs, targets);
            threads[i].start();
        }
        INDArray[] biases = new INDArray[this.biases.length];
        INDArray[] weights = new INDArray[this.weights.length];
        for (int i = 0; i < weights.length; i++) {
            weights[i] = Nd4j.zeros(this.weights[i].shape());
            biases[i] = Nd4j.zeros(this.biases[i].shape());
        }

        for(NNTrainingThread thread : threads){
            thread.join();
            for (int i = 0; i < this.weights.length; i++) {
                weights[i] = weights[i].add(thread.nn.weights[i]);
                biases[i] = biases[i].add(thread.nn.biases[i]);
            }
        }

        for (int i = 0; i < this.weights.length; i++) {
            this.weights[i] = weights[i].div(n);
            this.biases[i] = biases[i].div(n);
        }
    }
    public void trainThreads(float[][][] inputs, float[][][] targets, int iterations, int n) throws InterruptedException{
        trainThreads(Utilities.floatArrayToMatrixArray(inputs), Utilities.floatArrayToMatrixArray(targets), iterations, n);
    }
}
