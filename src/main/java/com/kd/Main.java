package com.kd;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.kd.Api;

import java.util.Arrays;

public class Main {
    public static void main(String[] args) {

        String json = "[[[0,0.0],[225,-1.6599999999999966],[93,-0.30000000000001137],[124,0.6000000000000085],[770,-1.7999999999999972],[110,-0.980000000000004],[22,0.28000000000000114],[85,-0.7999999999999972],[105,-1.4500000000000028],[0,0.0],[35,0.6499999999999915],[0,0.0],[217,-6.959999999999994],[143,2.8499999999999943],[0,0.0],[67,-0.28999999999999204],[0,0.0],[35,-0.29999999999999716],[0,0.0],[85,-1.0],[132,-2.0],[51,-0.4000000000000057],[0,0.0],[175,-1.8400000000000034],[53,-0.7599999999999909],[85,1.0],[53,1.5499999999999972],[0,0.0],[31,-1.1500000000000057],[155,0.0],[125,-2.3999999999999915],[51,-2.1000000000000085],[0,0.0],[182,-0.29999999999999716],[117,-2.0],[218,-0.09999999999999432],[0,0.0],[160,2.1999999999999886],[96,-3.1999999999999886],[46,2.0],[0,0.0],[111,-3.1000000000000085],[111,-0.7999999999999972],[70,-0.7000000000000028],[161,-3.0999999999999943]],[0.0,45.5,0.5,-33.5,5.0,30.5,-68.0,63.0,12.5,0.0,-30.5,0.0,-38.5,5.5,0.0,3.0,0.0,10.0,0.0,-24.5,-13.0,-30.0,0.0,10.0,-25.0,-500.0,522.5,0.0,-657.5,717.5,0.0,-107.5,0.0,35.0,-87.5,27.5,0.0,-32.5,32.5,-70.0,0.0,37.5,-94.5,-33.0,-47.5]]";

        JsonNode jsonDaysBodyweightsNode = null;
        JsonNode jsonLiftDiffNode = null;

        ObjectMapper objectMapper = new ObjectMapper();

        double[][] daysbw;
        double[][] inputs = new double[2][];
        double[] targets = null;

        try {
            jsonDaysBodyweightsNode = objectMapper.readTree(json).get(0);
            daysbw = Api.parseData(jsonDaysBodyweightsNode);

            // Fix: Initialize inputs arrays
            inputs[0] = Api.splitInputs(daysbw, 0); // Array of day changes
            inputs[1] = Api.splitInputs(daysbw, 1); // Array of bodyweight changes

            jsonLiftDiffNode = objectMapper.readTree(json).get(1);
            targets = Api.parseTotalKgDifferences(jsonLiftDiffNode);
        } catch (Exception e) {
            e.printStackTrace();
        }

        System.out.println("DaysBws: " + Arrays.deepToString(inputs));
        System.out.println("LiftDiffs: " + Arrays.toString(targets));

        Mlp mlp = new Mlp(2, 8, 1);

        Api.initializeWeightsAndBiases(mlp);

        // Train the model
        System.out.println("Training...");
        Api.train(mlp, inputs, targets, 2000, 0.0001);

        // Test predictions
        System.out.println("\nPredictions:");
        double pred1 = Api.denormalise(Api.forward(mlp, 150, 2.0), mlp.outMin, mlp.outMax);
        double pred2 = Api.denormalise(Api.forward(mlp, 200, -1.5), mlp.outMin, mlp.outMax);
        double pred3 = Api.denormalise(Api.forward(mlp, 1412, 21.5), mlp.outMin, mlp.outMax);

        System.out.println("Predicted weight increase for (150 days, +2.0kg): " + pred1);
        System.out.println("Predicted weight increase for (200 days, -1.5kg): " + pred2);
        System.out.println("Predicted weight increase for (1412 days, +21.5): " + pred3);
    }
}
