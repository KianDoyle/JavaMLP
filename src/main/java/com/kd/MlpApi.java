package com.kd;

import java.util.Random;

public class MlpApi {
    private int inputSize = 2;
    private int hiddenSize = 8;
    private int outputSize = 1;
    private double learningRate = 0.0001;

    private double[][] weightsInputHidden;
    private double[] weightsHiddenOutput;
    private double[] hiddenBias;
    private double outputBias;

    private Random random = new Random();

    public MlpApi() {
        initializeWeights();
    }

    private void initializeWeights() {
        weightsInputHidden = new double[inputSize][hiddenSize];
        weightsHiddenOutput = new double[hiddenSize];
        hiddenBias = new double[hiddenSize];

        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                weightsInputHidden[i][j] = random.nextGaussian() * Math.sqrt(2.0 / inputSize);
            }
        }

        for (int j = 0; j < hiddenSize; j++) {
            weightsHiddenOutput[j] = random.nextGaussian() * Math.sqrt(2.0 / hiddenSize);
            hiddenBias[j] = 0.0;
        }

        outputBias = 0.0;
    }

    private double leakyReLU(double x) {
        return x > 0 ? x : 0.01 * x;
    }

    private double leakyReLUDerivative(double x) {
        return x > 0 ? 1.0 : 0.01;
    }

    public double forward(double daysBetweenComps, double weightChange) {
        double[] hiddenLayer = new double[hiddenSize];

        // Normalize inputs
        double normDays = daysBetweenComps / 200.0;
        double normWeightChange = weightChange / 5.0;

        // Compute hidden layer activations
        for (int j = 0; j < hiddenSize; j++) {
            hiddenLayer[j] = weightsInputHidden[0][j] * normDays + weightsInputHidden[1][j] * normWeightChange + hiddenBias[j];
            hiddenLayer[j] = leakyReLU(hiddenLayer[j]);
        }

        // Compute output layer activation
        double output = outputBias;
        for (int j = 0; j < hiddenSize; j++) {
            output += weightsHiddenOutput[j] * hiddenLayer[j];
        }

        return output;
    }

    public void train(double[][] inputs, double[] targets, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalLoss = 0;

            for (int i = 0; i < inputs.length; i++) {
                double normDays = inputs[i][0] / 200.0;
                double normWeightChange = inputs[i][1] / 5.0;
                double target = targets[i];

                // Forward pass
                double[] hiddenLayer = new double[hiddenSize];
                for (int j = 0; j < hiddenSize; j++) {
                    hiddenLayer[j] = weightsInputHidden[0][j] * normDays + weightsInputHidden[1][j] * normWeightChange + hiddenBias[j];
                    hiddenLayer[j] = leakyReLU(hiddenLayer[j]);
                }

                double output = outputBias;
                for (int j = 0; j < hiddenSize; j++) {
                    output += weightsHiddenOutput[j] * hiddenLayer[j];
                }

                // Compute loss
                double error = target - output;
                totalLoss += error * error;

                // Backpropagation
                double outputGradient = -2 * error;

                double[] hiddenGradients = new double[hiddenSize];
                for (int j = 0; j < hiddenSize; j++) {
                    hiddenGradients[j] = outputGradient * weightsHiddenOutput[j] * leakyReLUDerivative(hiddenLayer[j]);
                }

                for (int j = 0; j < hiddenSize; j++) {
                    weightsHiddenOutput[j] -= learningRate * outputGradient * hiddenLayer[j];
                }

                outputBias -= learningRate * outputGradient;

                for (int j = 0; j < hiddenSize; j++) {
                    weightsInputHidden[0][j] -= learningRate * hiddenGradients[j] * normDays;
                    weightsInputHidden[1][j] -= learningRate * hiddenGradients[j] * normWeightChange;
                    hiddenBias[j] -= learningRate * hiddenGradients[j];
                }
            }

            if (epoch % 100 == 0) {
                System.out.println("Epoch " + epoch + " Loss: " + totalLoss / inputs.length);
            }
        }
    }
}
