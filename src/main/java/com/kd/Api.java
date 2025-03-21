//package com.kd;
//
//import com.fasterxml.jackson.databind.JsonNode;
//
//import java.util.ArrayList;
//import java.util.Arrays;
//import java.util.List;
//import java.util.Random;
//
//public class Api {
//
//    private static final Random random = new Random();
//
//    public static void initializeWeightsAndBiases(Mlp mlp) {
//
//        for (int i = 0; i < mlp.inputSize; i++) {
//            for (int j = 0; j < mlp.hiddenSize; j++) {
//                mlp.weightsInputHidden[i][j] = random.nextGaussian() * Math.sqrt(2.0 / mlp.inputSize);
//            }
//        }
//
//        for (int j = 0; j < mlp.hiddenSize; j++) {
//            mlp.weightsHiddenOutput[j] = random.nextGaussian() * Math.sqrt(2.0 / mlp.hiddenSize);
//            mlp.hiddenBias[j] = random.nextGaussian() * 0.01;
//        }
//
//        mlp.outputBias = random.nextGaussian() * 0.01;
//    }
//
//    private static double leakyReLU(double x) {
//        return x > 0 ? x : 0.01 * x;
//    }
//
//    private static double leakyReLUDerivative(double x) {
//        return x > 0 ? 1.0 : 0.01;
//    }
//
//    public static double forward(Mlp mlp, double daysBetweenComps, double weightChange) {
//        double[] hiddenLayer = new double[mlp.hiddenSize];
//
//        // Normalize inputs
//        double normDays = normalise(daysBetweenComps, mlp.inMins[0], mlp.inMaxs[0]);
//        double normWeightChange = normalise(weightChange, mlp.inMins[1], mlp.inMaxs[1]);
//
//        // Compute hidden layer activations
//        for (int j = 0; j < mlp.hiddenSize; j++) {
//            hiddenLayer[j] = mlp.weightsInputHidden[0][j] * normDays + mlp.weightsInputHidden[1][j] * normWeightChange + mlp.hiddenBias[j];
//            hiddenLayer[j] = leakyReLU(hiddenLayer[j]);
//        }
//
//        // Compute output layer activation
//        double output = mlp.outputBias;
//        for (int j = 0; j < mlp.hiddenSize; j++) {
//            output += mlp.weightsHiddenOutput[j] * hiddenLayer[j];
//        }
//
//        return output;
//    }
//
//    private static double findMax(double[] arr) {
//        return Arrays.stream(arr).max().orElseThrow();
//    }
//
//    private static double findMin(double[] arr) {
//        return Arrays.stream(arr).min().orElseThrow();
//    }
//
//    public static double denormalise(double normalizedValue, double min, double max) {
//        return ((normalizedValue + 1) / 2) * (max - min) + min;
//    }
//
//    public static double[] splitInputs(double[][] inputs, int index) {
//        double[] splitInput = new double[inputs.length];
//        for (int i=0; i<inputs.length; i++) {
//            splitInput[i] = inputs[i][index];
//        }
//        return splitInput;
//    }
//
//    public static void normalPreReq(Mlp mlp, double[][] inputs, double[] targets) {
//        for (int i=0; i<inputs[0].length; i++) {
//            mlp.inMaxs[i] = findMax(splitInputs(inputs, i));
//            mlp.inMins[i] = findMin(splitInputs(inputs, i));
//        }
//        mlp.outMax = findMax(targets);
//        mlp.outMin = findMin(targets);
//    }
//
//    public static double normalise(double value, double min, double max) {
//        double range = max - min;
//        if (range == 0) {
//            return 0;
//        }
//        return 2 * ((value - min) / range) - 1;
//    }
//
//    private static double[] computeHiddenLayer(Mlp mlp, double normDays, double normWeightChange) {
//        double[] hiddenLayer = new double[mlp.hiddenSize];
//
//        for (int j = 0; j < mlp.hiddenSize; j++) {
//            hiddenLayer[j] = mlp.weightsInputHidden[0][j] * normDays
//                    + mlp.weightsInputHidden[1][j] * normWeightChange
//                    + mlp.hiddenBias[j];
//            hiddenLayer[j] = leakyReLU(hiddenLayer[j]);
//        }
//
//        return hiddenLayer;
//    }
//
//    private static double computeOutput(Mlp mlp, double[] hiddenLayer) {
//        double output = mlp.outputBias;
//
//        for (int j = 0; j < mlp.hiddenSize; j++) {
//            output += mlp.weightsHiddenOutput[j] * hiddenLayer[j];
//        }
//
//        return output;
//    }
//
//    private static void backpropagate(Mlp mlp, double[] hiddenLayer, double normDays, double normWeightChange, double error) {
//        double outputGradient = -2 * error;
//
//        double[] hiddenGradients = new double[mlp.hiddenSize];
//        for (int j = 0; j < mlp.hiddenSize; j++) {
//            hiddenGradients[j] = outputGradient * mlp.weightsHiddenOutput[j] * leakyReLUDerivative(hiddenLayer[j]);
//        }
//
//        for (int j = 0; j < mlp.hiddenSize; j++) {
//            mlp.weightsHiddenOutput[j] -= mlp.learningRate * outputGradient * hiddenLayer[j];
//        }
//
//        mlp.outputBias -= mlp.learningRate * outputGradient;
//
//        for (int j = 0; j < mlp.hiddenSize; j++) {
//            mlp.weightsInputHidden[0][j] -= mlp.learningRate * hiddenGradients[j] * normDays;
//            mlp.weightsInputHidden[1][j] -= mlp.learningRate * hiddenGradients[j] * normWeightChange;
//            mlp.hiddenBias[j] -= mlp.learningRate * hiddenGradients[j];
//        }
//    }
//
//    public static void train(Mlp mlp, double[][] inputs, double[] targets, int epochs, double learningRate) {
//        normalPreReq(mlp, inputs, targets);
//
//        double[] normDays = new double[inputs.length];
//        double[] normWeights = new double[inputs.length];
//        double[] normTargets = new double[targets.length];
//
//        for (int i = 0; i < inputs.length; i++) {
//            normDays[i] = normalise(inputs[i][0], mlp.inMins[0], mlp.inMaxs[0]);
//            normWeights[i] = normalise(inputs[i][1], mlp.inMins[1], mlp.inMaxs[1]);
//            normTargets[i] = normalise(targets[i], mlp.outMin, mlp.outMax);
//        }
//
//        for (int epoch = 0; epoch < epochs; epoch++) {
//            double totalLoss = 0;
//
//            for (int i = 0; i < inputs.length; i++) {
//                // Forward pass
//                double[] hiddenLayer = computeHiddenLayer(mlp, normDays[i], normWeights[i]);
//                double output = computeOutput(mlp, hiddenLayer);
//
//                // Compute loss
//                double error = normTargets[i] - output;
//                totalLoss += error * error;
//
//                mlp.learningRate = learningRate;
//                // Backpropagation
//                backpropagate(mlp, hiddenLayer, normDays[i], normWeights[i], error);
//            }
//
//            if (epoch % 100 == 0) {
//                System.out.println("Epoch " + epoch + " Loss: " + totalLoss / inputs.length);
//            }
//        }
//    }
//
//    public static double[][] parseData(JsonNode jsonNode) {
//        double[][] resultArray = new double[jsonNode.size()][2];
//
//        // Populate the array with data from the JSON node
//        for (int i = 0; i < jsonNode.size(); i++) {
//            JsonNode pairNode = jsonNode.get(i);
//            resultArray[i][0] = pairNode.get(0).asDouble(); // First element in pair (e.g., 0)
//            resultArray[i][1] = pairNode.get(1).asDouble(); // Second element in pair (e.g., 0.0)
//        }
//
//        return resultArray;
//    }
//
//    public static double[] parseTotalKgDifferences(JsonNode jsonNode) {
//        double[] resultArray = new double[jsonNode.size()];
//
//        // Populate the array with data from the JSON node
//        for (int i = 0; i < jsonNode.size(); i++) {
//            resultArray[i] = jsonNode.get(i).asDouble();
//        }
//
//        return resultArray;
//    }
//}

package com.kd;

import com.fasterxml.jackson.databind.JsonNode;

import java.util.Arrays;
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

    // Modified forward pass to accept day change and body weight change
    public static double forward(Mlp mlp, int dayChange, double weightChange) {
        double[] hiddenLayer = new double[mlp.hiddenSize];

        // Normalize inputs
        double normDays = normalise(dayChange, mlp.inMins[0], mlp.inMaxs[0]);
        double normWeightChange = normalise(weightChange, mlp.inMins[1], mlp.inMaxs[1]);

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

    private static double findMax(double[] arr) {
        return Arrays.stream(arr).max().orElseThrow();
    }

    private static double findMin(double[] arr) {
        return Arrays.stream(arr).min().orElseThrow();
    }

    public static double denormalise(double normalizedValue, double min, double max) {
        return ((normalizedValue + 1) / 2) * (max - min) + min;
    }

    public static double[] splitInputs(double[][] inputs, int index) {
        double[] splitInput = new double[inputs[0].length];
        for (int i=0; i<inputs[0].length; i++) {
            splitInput[i] = inputs[index][i];
        }
        return splitInput;
    }

    public static void normalPreReq(Mlp mlp, double[][] inputs, double[] targets) {
        // Inputs are now [dayChanges[], weightChanges[]], so split appropriately
        mlp.inMaxs[0] = findMax(inputs[0]);
        mlp.inMins[0] = findMin(inputs[0]);
        mlp.inMaxs[1] = findMax(inputs[1]);
        mlp.inMins[1] = findMin(inputs[1]);

        mlp.outMax = findMax(targets);
        mlp.outMin = findMin(targets);
    }

    public static double normalise(double value, double min, double max) {
        double range = max - min;
        if (range == 0) {
            return 0;
        }
        return 2 * ((value - min) / range) - 1;
    }

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

    private static double computeOutput(Mlp mlp, double[] hiddenLayer) {
        double output = mlp.outputBias;

        for (int j = 0; j < mlp.hiddenSize; j++) {
            output += mlp.weightsHiddenOutput[j] * hiddenLayer[j];
        }

        return output;
    }

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
        normalPreReq(mlp, inputs, targets);

        double[] normDays = new double[inputs[0].length];
        double[] normWeights = new double[inputs[1].length];
        double[] normTargets = new double[targets.length];

        for (int i = 0; i < inputs[0].length; i++) {
            normDays[i] = normalise(inputs[0][i], mlp.inMins[0], mlp.inMaxs[0]);
            normWeights[i] = normalise(inputs[1][i], mlp.inMins[1], mlp.inMaxs[1]);
            normTargets[i] = normalise(targets[i], mlp.outMin, mlp.outMax);
        }

        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalLoss = 0;

            for (int i = 0; i < inputs[0].length; i++) {
                // Forward pass
                double[] hiddenLayer = computeHiddenLayer(mlp, normDays[i], normWeights[i]);
                double output = computeOutput(mlp, hiddenLayer);

                // Compute loss
                double error = normTargets[i] - output;
                totalLoss += error * error;

                mlp.learningRate = learningRate;
                // Backpropagation
                backpropagate(mlp, hiddenLayer, normDays[i], normWeights[i], error);
            }

            if (epoch % 100 == 0) {
                System.out.println("Epoch " + epoch + " Loss: " + totalLoss / inputs[0].length);
            }
        }
    }

    public static double[][] parseData(JsonNode jsonNode) {
        double[][] resultArray = new double[2][jsonNode.size()];

        // Populate the array with data from the JSON node
        for (int i = 0; i < jsonNode.size(); i++) {
            JsonNode pairNode = jsonNode.get(i);
            resultArray[0][i] = pairNode.get(0).asDouble(); // Day change
            resultArray[1][i] = pairNode.get(1).asDouble(); // Bodyweight change
        }

        return resultArray;
    }

    public static double[] parseTotalKgDifferences(JsonNode jsonNode) {
        double[] resultArray = new double[jsonNode.size()];

        // Populate the array with data from the JSON node
        for (int i = 0; i < jsonNode.size(); i++) {
            resultArray[i] = jsonNode.get(i).asDouble();
        }

        return resultArray;
    }
}

