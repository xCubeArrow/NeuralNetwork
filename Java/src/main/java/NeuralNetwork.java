import java.util.Arrays;
import java.util.Random;
import java.util.function.Function;

public class NeuralNetwork {

    Function activationFunction;
    Function dActivationFunction;
    float learningRate;
    Matrix[] weights;
    Matrix[] biases;
    Matrix[] out;
    Matrix[] net;

    public NeuralNetwork(Function activationFunction, Function dActivationFunction, float learningRate, Matrix[] weights, Matrix[] biases) {
        this.activationFunction = activationFunction;
        this.dActivationFunction = dActivationFunction;
        this.learningRate = learningRate;
        this.weights = weights;
        this.biases = biases;

        this.out = new Matrix[weights.length];
        this.net = new Matrix[weights.length];

    }

    public NeuralNetwork(int[] layers) {

        this.weights = new Matrix[layers.length - 1];
        this.biases = new Matrix[layers.length - 1];
        this.out = new Matrix[layers.length - 1];
        this.net = new Matrix[layers.length - 1];
        for (int i = 1; i < layers.length; i++) {
            this.weights[i - 1] = new Matrix(layers[i - 1], layers[i]).randomize();
            this.biases[i - 1] = new Matrix(1, layers[i]).randomize();
        }
    }

    public static Object sigmoid(Object x) {
        return (float) x / (1 + Math.abs((float) x));
    }

    public static Object dSigmoid(Object x) {
        return (float) (1 / (1 + Math.pow(Math.abs((float) x), 2)));
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

    public Matrix[] getWeights() {
        return weights;
    }

    public void setWeights(Matrix[] weights) {
        this.weights = weights;
    }

    public Matrix[] getBiases() {
        return biases;
    }

    public void setBiases(Matrix[] biases) {
        this.biases = biases;
    }

    public void feedForward(Matrix inputs) {
        Matrix layerInputs = inputs;
        for (int i = 0; i < weights.length; i++) {
            // Multiply the previous output with the current weights and add the bias
            Matrix x = layerInputs.dot(this.weights[i]);
            x = x.addMatrix(this.biases[i]);

            // Save both the output and the output without the activation function
            this.net[i] = x;
            this.out[i] = x.applyFunction(this.activationFunction);
            layerInputs = out[i];
        }
    }

    public void backPropagation(Matrix inputs, Matrix targets) {
        Matrix[] deltas = new Matrix[this.weights.length];
        // Calculate the delta of the output layer
        deltas[deltas.length - 1] = targets.subtractMatrix(this.out[out.length - 1])
                .dot(this.net[net.length - 1].applyFunction(dActivationFunction));

        // Calculate the deltas of the hidden layers using the previous delta
        for (int i = weights.length - 2; i >= 0; i--) {
            Matrix error = deltas[i + 1].dot(this.weights[i + 1].transpose());
            deltas[i] = error.multiplyByMatrix(net[i].applyFunction(dActivationFunction));
        }
        // Update the weights and the biases
        for (int i = 0; i < this.weights.length; i++) {
            // Set the previous output to the inputs if there is no previous outputs
            Matrix input = inputs;
            if (i != 0) {
                input = this.out[i - 1];
            }

            // Update both the weights and the biases
            this.weights[i] = input.transpose().dot(deltas[i]).multiplyByN(this.learningRate).addMatrix(this.weights[i]);
            this.biases[i] = deltas[i].multiplyByN(this.learningRate).addMatrix(biases[i]);
        }


    }


    public void train(Matrix[] inputs, Matrix[] targets, int iterations) {
        for (int i = 0; i < iterations; i++) {
            int randInt = new Random().nextInt(inputs.length);
            this.feedForward(inputs[randInt]);
            this.backPropagation(inputs[randInt], targets[randInt]);

        }
    }

    public void train(float[][][] inputs, float[][][] targets, int iterations) {
        Matrix[] inputsMatrix = new Matrix[inputs.length];
        Matrix[] targetsMatrix = new Matrix[targets.length];

        for (int i = 0; i < inputs.length; i++) {
            inputsMatrix[i] = new Matrix(inputs[i]);
            targetsMatrix[i] = new Matrix(targets[i]);
        }
        train(inputsMatrix, targetsMatrix, iterations);
    }

    public void test(Matrix[] inputs) {
        for (Matrix input : inputs) {
            this.feedForward(input);
            System.out.println("Outputs: " + Arrays.deepToString(this.out[out.length - 1].data));
        }
    }

    public void test(float[][][] inputs){
        Matrix[] inputsMatrix = new Matrix[inputs.length];

        for (int i = 0; i < inputs.length; i++) {
            inputsMatrix[i] = new Matrix(inputs[i]);
        }
        test(inputsMatrix);
    }
}
