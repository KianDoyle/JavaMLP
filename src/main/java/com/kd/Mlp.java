package com.kd;

public class Mlp {
    public int inputSize;
    public int hiddenSize;
    public int outputSize;
    public double learningRate;

    public double[][] weightsInputHidden;
    public double[] weightsHiddenOutput;
    public double[] hiddenBias;
    public double outputBias;

    double outMax;
    double outMin;
    double[] inMaxs;
    double[] inMins;

    public Mlp(int inputSize, int hiddenSize, int outputSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        this.weightsInputHidden = new double[inputSize][hiddenSize];
        this.weightsHiddenOutput = new double[hiddenSize];
        this.hiddenBias = new double[hiddenSize];
        this.outMax = 0;
        this.outMin = 0;
        this.inMaxs = new double[inputSize];
        this.inMins = new double[inputSize];
    }

    public Mlp(int inputSize, int hiddenSize, int outputSize, double learningRate, double[][] weightsInputHidden, double[] weightsHiddenOutput, double[] hiddenBias, double outputBias, double outMax, double outMin, double[] inMaxs, double[] inMins) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        this.learningRate = learningRate;
        this.weightsInputHidden = weightsInputHidden;
        this.weightsHiddenOutput = weightsHiddenOutput;
        this.hiddenBias = hiddenBias;
        this.outputBias = outputBias;
        this.outMax = outMax;
        this.outMin = outMin;
        this.inMaxs = inMaxs;
        this.inMins = inMins;
    }
}
