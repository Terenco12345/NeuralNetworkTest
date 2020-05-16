package main;

import main.neuralnet.InvalidInputSizeException;
import main.neuralnet.NeuralNetwork;

public class Main {
	
	public static void main(String[] args) throws InvalidInputSizeException {
		NeuralNetwork neuralNetwork = new NeuralNetwork();
		
		int dataSize = 10000;
		
		double[] input = new double[dataSize];
		double[] output = new double[dataSize];
		
		for(int i = 0; i < dataSize; i++) {
			double random = Math.random();
			
			input[i] = random;
			output[i] = (random > 0.5) ? 1 : 0 ;
		}
		
		// Train
		for(int i = 0; i < dataSize; i++) {
			neuralNetwork.trainNetwork(new double[] {input[i]}, new double[] {output[i]}, 2);
		}
		
		// Predict
		System.out.println("Neural network output: " +neuralNetwork.getOutput(new double[] {0.2})[0]+" | Output should be 0");
		System.out.println("Neural network output: " +neuralNetwork.getOutput(new double[] {0.6})[0]+" | Output should be 1");
		System.out.println("Neural network output: " +neuralNetwork.getOutput(new double[] {0.7})[0]+" | Output should be 1");
		System.out.println("Neural network output: " +neuralNetwork.getOutput(new double[] {0.5})[0]+" | Output should be 0");
		System.out.println("Neural network output: " +neuralNetwork.getOutput(new double[] {0.51})[0]+" | Output should be 1");
	}
}
