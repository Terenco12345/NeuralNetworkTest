package main.neuralnet;

import java.util.Random;

/**
 * This represents a synapse/directional connection that exists between two neurons.
 * A synapse has a specific weighting, that will change as the neural network trains.
 */
public class Synapse {
	private Neuron fromNeuron;
	private Neuron toNeuron;
	private double weight;
	
	/**
	 * A synapse will be constructed that connects one neuron to another.
	 * @param fromNeuron
	 * @param toNeuron
	 */
	public Synapse(Neuron fromNeuron, Neuron toNeuron, double weight) {
		this.setFromNeuron(fromNeuron);
		this.setToNeuron(toNeuron);
		this.setWeight(weight);
	}

	/**
	 * A synapse will be constructed that connects one neuron to another.
	 * A constructor with no weight specified will have a random weight from 0.0 to 1.0 assigned automatically.
	 * @param fromNeuron
	 * @param toNeuron
	 */
	public Synapse(Neuron fromNeuron, Neuron toNeuron) {
		this.setFromNeuron(fromNeuron);
		this.setToNeuron(toNeuron);
		this.setWeight(new Random().nextDouble());
	}
	
	public Neuron getFromNeuron() {
		return fromNeuron;
	}

	public void setFromNeuron(Neuron fromNeuron) {
		this.fromNeuron = fromNeuron;
	}

	public Neuron getToNeuron() {
		return toNeuron;
	}

	public void setToNeuron(Neuron toNeuron) {
		this.toNeuron = toNeuron;
	}

	public double getWeight() {
		return weight;
	}

	public void setWeight(double weight) {
		this.weight = weight;
	}
}
