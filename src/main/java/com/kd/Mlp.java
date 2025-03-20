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

    public Mlp(int inputSize, int hiddenSize, int outputSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        this.weightsInputHidden = new double[inputSize][hiddenSize];
        this.weightsHiddenOutput = new double[hiddenSize];
        this.hiddenBias = new double[hiddenSize];
    }

    public Mlp(int inputSize, int hiddenSize, int outputSize, double learningRate, double[][] weightsInputHidden, double[] weightsHiddenOutput, double[] hiddenBias, double outputBias) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        this.learningRate = learningRate;
        this.weightsInputHidden = weightsInputHidden;
        this.weightsHiddenOutput = weightsHiddenOutput;
        this.hiddenBias = hiddenBias;
        this.outputBias = outputBias;
    }
}
