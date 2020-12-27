/*  Copyright 2020, Christian Flender

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.

    Dieses Programm ist Freie Software: Sie können es unter den Bedingungen
    der GNU General Public License, wie von der Free Software Foundation,
    Version 3 der Lizenz oder (nach Ihrer Wahl) jeder neueren
    veröffentlichten Version, weiter verteilen und/oder modifizieren.

    Dieses Programm wird in der Hoffnung bereitgestellt, dass es nützlich sein wird, 
    jedoch OHNE JEDE GEWÄHR; sogar ohne die implizite Gewähr der MARKTFÄHIGKEIT 
    oder EIGNUNG FÜR EINEN BESTIMMTEN ZWECK.
    
    Siehe die GNU General Public License für weitere Einzelheiten. */

package com.siba.timeseries.connectionism;

import java.lang.Math;
import java.util.Random;

public class NeuralNetwork {

    private int nodes1;
    private int nodes2;
    private int nodes3;
    private double lr;
    private double layer1[][];
    private double layer2[][];
    private double layer3[][];
    private double error3[][];
    private double error2[][];
    private double weights12[][];
    private double weights23[][];

    public NeuralNetwork(int inodes, int hnodes, int onodes, double rate) {
        
        nodes1 = inodes;
        nodes2 = hnodes;
        nodes3 = onodes;
        lr = rate;

        layer1 = new double[nodes1][1];
        layer2 = new double[nodes2][1];
        layer3 = new double[nodes3][1];
        error3 = new double[nodes3][1];
        error2 = new double[nodes2][1];

        weights12 = init(nodes2, nodes1);
        weights23 = init(nodes3, nodes2);
    }

    public void train(double[][] input, double[][] target) {

        layer1 = input;
        layer2 = multiply(weights12, layer1);
        layer3 = multiply(weights23, sigmoid(layer2));
        error3 = subtract(target, sigmoid(layer3));
        error2 = multiply(transpose(weights23), error3);
         
        weights23 = update(lr,weights23, multiply(product(product(error3,sigmoid(layer3)), inverse(sigmoid(layer3))),transpose(sigmoid(layer2))));
        weights12 = update(lr,weights12, multiply(product(product(error2,sigmoid(layer2)), inverse(sigmoid(layer2))),transpose(layer1)));
    }

    public double[][] query(double[][] input) {

        layer1 = input;
        layer2 = multiply(weights12, layer1);
        layer3 = multiply(weights23, sigmoid(layer2));

        return sigmoid(layer3);
    }

    private static double[][] multiply(double[][] matrix1, double[][] matrix2) {

        double result[][] = new double[matrix1.length][matrix2[0].length]; 

        for (int i = 0; i < matrix1.length; i++) {

            for (int j = 0; j < matrix2[0].length; j++) {

                result[i][j] = 0;

                for (int k = 0; k < matrix1[0].length; k++) {

                    result[i][j] += matrix1[i][k] * matrix2[k][j];

                } 
            } 
        } 
        return result;
    }

    private static double[][] product(double[][] matrix1, double[][] matrix2){

        double result[][]=new double[matrix1.length][1];
    
        for(int i=0;i<matrix1.length;i++) {    
           
            result[i][0] = matrix1[i][0] * matrix2[i][0];     
          
        }
        return result;     
    }

    private static double[][] subtract(double[][] matrix1, double[][] matrix2) {

        double result[][] = new double[matrix1.length][matrix1[0].length];

        for (int i = 0; i < matrix1.length; i++) {

            result[i][0] = matrix1[i][0] - matrix2[i][0];

        } 
        return result;
    }

    private static double[][] update(double lr, double[][] matrix1, double[][] matrix2) {

        double result[][] = new double[matrix1.length][matrix1[0].length];

        for (int i = 0; i < matrix1.length; i++) {

            for (int j = 0; j < matrix1[0].length; j++) {

                result[i][j] = matrix1[i][j] + lr * (matrix2[i][j]);

            }
        }
        return result;
    }

    private static double[][] inverse(double[][] matrix) {

        double result[][] = new double[matrix.length][matrix[0].length];

        for (int i = 0; i < matrix.length; i++) {

            result[i][0] = 1 - matrix[i][0];

        }
        return result;
    }

    private static double[][] transpose(double[][] matrix) {

        double result[][] = new double[matrix[0].length][matrix.length];
        for (int i = 0; i < matrix.length; i++) {

            for (int j = 0; j < matrix[0].length; j++) {
                result[j][i] = matrix[i][j];
            }
        }
        return result;
    }

    private static double[][] sigmoid(double[][] matrix) {

        double result[][] = new double[matrix.length][matrix[0].length];
        for (int i = 0; i < matrix.length; i++) {

            for (int j = 0; j < matrix[0].length; j++) {

                result[i][j] = 1 / (1 + Math.exp(-matrix[i][j]));

            }

        }
        return result;
    }

    private static double[][] init(int rows, int cols) {

        double result[][] = new double[rows][cols];
        double rand;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                do {
                    rand = 0.0 + new Random().nextGaussian() * (1/Math.pow(rows,-0.5));
                    result[i][j] = rand;
                } while (rand < -0.5 || rand > 0.5);
            }
        }
        return result;
    }
}