package main.neuralnet;

import java.util.ArrayList;
import java.util.List;

import main.function.Function;

/**
 * This is a neuron/node in a neural network.
 * A neuron takes a weighted sum from it's inputs and passes it through an activation function,
 * which will then return it's value to be used by neurons in the next layer.
 */
public class Neuron {
	private List<Synapse> incomingSynapses;
	private List<Synapse> outgoingSynapses;
	
	private double currentValue;
	private double derivative;
	
	private Function activationFunction;
	
	
	public Neuron(Function activationFunction) {
		incomingSynapses = new ArrayList<Synapse>();
		outgoingSynapses = new ArrayList<Synapse>();
		currentValue = 0;
		
		this.activationFunction = activationFunction;
	}
	
	/**
	 * Applies forward propagation to calculate this neuron's current value.
	 * Calculate weighted sum of incoming synapses and applies an activation function to this weighted sum.
	 * The value obtained will be stored in currentValue.
	 */
	public void updateCurrentValue() {
		if(incomingSynapses.size() == 0) {
			return;
		}
		
		// Calculate the weighted sum of all incoming connections
		double weightedSum = 0;
		for(int i = 0; i < incomingSynapses.size(); i++) {
			Synapse synapse = incomingSynapses.get(i);
			weightedSum += synapse.getWeight()*synapse.getFromNeuron().getCurrentValue();
		}
		
		// Apply the activation function to the weighted sum
		currentValue = activationFunction.getOutput(weightedSum);
	}
	
	/**
	 * Updating partial derivative value and storing it. This value is based on the p. div. of the activation function.
	 * @param error
	 */
	public void updateDerivative(double error) {
		derivative = error*activationFunction.getDerivativeOutput(this.getCurrentValue());
	}
	
	// Getters and setters
	public List<Synapse> getIncomingSynapses() {
		return incomingSynapses;
	}
	
	public List<Synapse> getOutgoingSynapses() {
		return outgoingSynapses;
	}
	
	public void setCurrentValue(double value) {
		currentValue = value;
	}
	
	public double getCurrentValue() {
		return currentValue;
	}
	
	public Function getFunction() {
		return activationFunction;
	}
	
	public double getDerivative() {
		return derivative;
	}
}
