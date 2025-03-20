package com.kd;

import java.util.Random;

public class Api {

    private static final Random random = new Random();

    public static void initializeWeightsAndBiases(Mlp mlp) {

        for (int i = 0; i < mlp.inputSize; i++) {
            for (int j = 0; j < mlp.hiddenSize; j++) {
                mlp.weightsInputHidden[i][j] = random.nextGaussian() * Math.sqrt(2.0 / mlp.inputSize);
            }
        }

        for (int j = 0; j < mlp.hiddenSize; j++) {
            mlp.weightsHiddenOutput[j] = random.nextGaussian() * Math.sqrt(2.0 / mlp.hiddenSize);
            mlp.hiddenBias[j] = random.nextGaussian() * 0.01;
        }

        mlp.outputBias = random.nextGaussian() * 0.01;
    }

    private static double leakyReLU(double x) {
        return x > 0 ? x : 0.01 * x;
    }

    private static double leakyReLUDerivative(double x) {
        return x > 0 ? 1.0 : 0.01;
    }

    public static double forward(Mlp mlp, double daysBetweenComps, double weightChange) {
        double[] hiddenLayer = new double[mlp.hiddenSize];

        // Normalize inputs
        double normDays = daysBetweenComps / 365.0;
        double normWeightChange = weightChange / 5.0;

        // Compute hidden layer activations
        for (int j = 0; j < mlp.hiddenSize; j++) {
            hiddenLayer[j] = mlp.weightsInputHidden[0][j] * normDays + mlp.weightsInputHidden[1][j] * normWeightChange + mlp.hiddenBias[j];
            hiddenLayer[j] = leakyReLU(hiddenLayer[j]);
        }

        // Compute output layer activation
        double output = mlp.outputBias;
        for (int j = 0; j < mlp.hiddenSize; j++) {
            output += mlp.weightsHiddenOutput[j] * hiddenLayer[j];
        }

        return output;
    }

//    public static void train(Mlp mlp, double[][] inputs, double[] targets, int epochs) {
//        for (int epoch = 0; epoch < epochs; epoch++) {
//            double totalLoss = 0;
//
//            for (int i = 0; i < inputs.length; i++) {
//                double normDays = inputs[i][0] / 200.0;
//                double normWeightChange = inputs[i][1] / 5.0;
//                double target = targets[i];
//
//                // Forward pass
//                double[] hiddenLayer = new double[mlp.hiddenSize];
//                for (int j = 0; j < mlp.hiddenSize; j++) {
//                    hiddenLayer[j] = mlp.weightsInputHidden[0][j] * normDays + mlp.weightsInputHidden[1][j] * normWeightChange + mlp.hiddenBias[j];
//                    hiddenLayer[j] = leakyReLU(hiddenLayer[j]);
//                }
//
//                double output = mlp.outputBias;
//                for (int j = 0; j < mlp.hiddenSize; j++) {
//                    output += mlp.weightsHiddenOutput[j] * hiddenLayer[j];
//                }
//
//                // Compute loss
//                double error = target - output;
//                totalLoss += error * error;
//
//                // Backpropagation
//                double outputGradient = -2 * error;
//
//                double[] hiddenGradients = new double[mlp.hiddenSize];
//                for (int j = 0; j < mlp.hiddenSize; j++) {
//                    hiddenGradients[j] = outputGradient * mlp.weightsHiddenOutput[j] * leakyReLUDerivative(hiddenLayer[j]);
//                }
//
//                for (int j = 0; j < mlp.hiddenSize; j++) {
//                    mlp.weightsHiddenOutput[j] -= mlp.learningRate * outputGradient * hiddenLayer[j];
//                }
//
//                mlp.outputBias -= mlp.learningRate * outputGradient;
//
//                for (int j = 0; j < mlp.hiddenSize; j++) {
//                    mlp.weightsInputHidden[0][j] -= mlp.learningRate * hiddenGradients[j] * normDays;
//                    mlp.weightsInputHidden[1][j] -= mlp.learningRate * hiddenGradients[j] * normWeightChange;
//                    mlp.hiddenBias[j] -= mlp.learningRate * hiddenGradients[j];
//                }
//            }
//
//            if (epoch % 100 == 0) {
//                System.out.println("Epoch " + epoch + " Loss: " + totalLoss / inputs.length);
//            }
//        }
//    }

    /** Normalize the input days */
    private static double normalizeDays(double days) {
        return days / 200.0;
    }

    /** Normalize the input weight change */
    private static double normalizeWeightChange(double weightChange) {
        return weightChange / 5.0;
    }

    /** Compute hidden layer activations */
    private static double[] computeHiddenLayer(Mlp mlp, double normDays, double normWeightChange) {
        double[] hiddenLayer = new double[mlp.hiddenSize];

        for (int j = 0; j < mlp.hiddenSize; j++) {
            hiddenLayer[j] = mlp.weightsInputHidden[0][j] * normDays
                    + mlp.weightsInputHidden[1][j] * normWeightChange
                    + mlp.hiddenBias[j];
            hiddenLayer[j] = leakyReLU(hiddenLayer[j]);
        }

        return hiddenLayer;
    }

    /** Compute output layer activation */
    private static double computeOutput(Mlp mlp, double[] hiddenLayer) {
        double output = mlp.outputBias;

        for (int j = 0; j < mlp.hiddenSize; j++) {
            output += mlp.weightsHiddenOutput[j] * hiddenLayer[j];
        }

        return output;
    }

    /** Perform backpropagation to update weights and biases */
    private static void backpropagate(Mlp mlp, double[] hiddenLayer, double normDays, double normWeightChange, double error) {
        double outputGradient = -2 * error;

        double[] hiddenGradients = new double[mlp.hiddenSize];
        for (int j = 0; j < mlp.hiddenSize; j++) {
            hiddenGradients[j] = outputGradient * mlp.weightsHiddenOutput[j] * leakyReLUDerivative(hiddenLayer[j]);
        }

        for (int j = 0; j < mlp.hiddenSize; j++) {
            mlp.weightsHiddenOutput[j] -= mlp.learningRate * outputGradient * hiddenLayer[j];
        }

        mlp.outputBias -= mlp.learningRate * outputGradient;

        for (int j = 0; j < mlp.hiddenSize; j++) {
            mlp.weightsInputHidden[0][j] -= mlp.learningRate * hiddenGradients[j] * normDays;
            mlp.weightsInputHidden[1][j] -= mlp.learningRate * hiddenGradients[j] * normWeightChange;
            mlp.hiddenBias[j] -= mlp.learningRate * hiddenGradients[j];
        }
    }

    public static void train(Mlp mlp, double[][] inputs, double[] targets, int epochs, double learningRate) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalLoss = 0;

            for (int i = 0; i < inputs.length; i++) {
                double normDays = normalizeDays(inputs[i][0]);
                double normWeightChange = normalizeWeightChange(inputs[i][1]);
                double target = targets[i];

                // Forward pass
                double[] hiddenLayer = computeHiddenLayer(mlp, normDays, normWeightChange);
                double output = computeOutput(mlp, hiddenLayer);

                // Compute loss
                double error = target - output;
                totalLoss += error * error;

                mlp.learningRate = learningRate;
                // Backpropagation
                backpropagate(mlp, hiddenLayer, normDays, normWeightChange, error);
            }

            if (epoch % 100 == 0) {
                System.out.println("Epoch " + epoch + " Loss: " + totalLoss / inputs.length);
            }
        }
    }
}
