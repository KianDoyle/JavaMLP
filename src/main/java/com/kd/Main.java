package com.kd;

import com.kd.Api;

public class Main {
    public static void main(String[] args) {
        // Training data: {days between comps, weight change}, target = strength increase
        double[][] inputs = {
                {90, 0.0},  // 90 days, no weight change
                {120, 1.5},  // 120 days, gained 1.5kg
                {150, -2.0}, // 150 days, lost 2.0kg
                {200, 2.5},  // 200 days, gained 2.5kg
                {250, -3.0}, // 250 days, lost 3.0kg
                {180, 0.5},  // 180 days, slight weight gain
                {300, -1.0}, // 300 days, slight weight loss
                {50, 1.0},  // 50 days, small gain
                {100, -1.5}, // 100 days, small loss
                {270, 3.0}   // 270 days, large gain
        };

        // Target strength increase (hypothetical data)
        double[] targets = {10.0, 12.5, 8.0, 15.0, 6.0, 11.0, 7.5, 5.0, 9.0, 14.0};

        Mlp mlp = new Mlp(2, 8, 1);

        Api.initializeWeightsAndBiases(mlp);

        // Train the model
        System.out.println("Training...");
        Api.train(mlp, inputs, targets, 2000, 0.0001);

        // Test predictions
        System.out.println("\nPredictions:");
        double pred1 = Api.forward(mlp, 150, 2.0);
        double pred2 = Api.forward(mlp, 200, -1.5);
        double pred3 = Api.forward(mlp, 1412, 21.5);

        System.out.println("Predicted weight increase for (150 days, +2.0kg): " + pred1);
        System.out.println("Predicted weight increase for (200 days, -1.5kg): " + pred2);
        System.out.println("Predicted weight increase for (1412 days, +21.5): " + pred3);
    }
}

