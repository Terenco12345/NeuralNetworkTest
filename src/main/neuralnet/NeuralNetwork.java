package main.neuralnet;

import java.util.ArrayList;
import java.util.List;

import main.function.Function;
import main.function.SigmoidFunction;

/**
 * This will represent a fully connected neural network.
 * This network can be customized in many ways.
 */
public class NeuralNetwork {

	List<Neuron> inputLayer;
	List<Neuron> hiddenLayer1;
	List<Neuron> outputLayer;
	
	private Function activationFunction;

	public NeuralNetwork() {
		inputLayer = new ArrayList<Neuron>();
		hiddenLayer1 = new ArrayList<Neuron>();
		outputLayer = new ArrayList<Neuron>();
		
		activationFunction = new SigmoidFunction();
		
		initialize();
	}

	/**
	 * Initializes all neurons and synapses for this neural network. Weights will be completely random.
	 */
	private void initialize() {
		int inputLayerSize = 1;
		int hiddenLayer1Size = 10;
		int outputLayerSize = 1;

		// Initialize input layer neurons.
		for(int i = 0; i < inputLayerSize; i++) {
			inputLayer.add(new Neuron(activationFunction));
		}

		initializeLayer(hiddenLayer1, inputLayer, hiddenLayer1Size);
		initializeLayer(outputLayer, hiddenLayer1, outputLayerSize);

	}

	/**
	 * Initializes all synapse connections for currentLayer, based on previousLayer.
	 * @param currentLayer
	 * @param previousLayer
	 * @param layerSize
	 */
	private void initializeLayer(List<Neuron> currentLayer, List<Neuron> previousLayer, int layerSize) {
		// Bias neuron that feeds into current layer
		Neuron biasNeuron = new Neuron(activationFunction);
		biasNeuron.setCurrentValue(1.0);
		
		// Initialize output layer neurons.
		for(int i = 0; i < layerSize; i++) {
			Neuron neuron = new Neuron(activationFunction);

			// Create all synapses for this neuron, and all neurons going into it
			for(int j = 0; j < previousLayer.size(); j++) {
				Neuron previousLayerNeuron = previousLayer.get(j);
				Synapse synapse = new Synapse(previousLayerNeuron, neuron);

				previousLayerNeuron.getOutgoingSynapses().add(synapse);
				neuron.getIncomingSynapses().add(synapse);
			}
			
			// Add bias neuron's connections
			Synapse biasSynapse = new Synapse(biasNeuron, neuron, 0);
			biasNeuron.getOutgoingSynapses().add(biasSynapse);
			neuron.getIncomingSynapses().add(biasSynapse);
			
			currentLayer.add(neuron);
		}
	}

	/**
	 * Returns a set of output data for the current weight configuration, when a set of input data is supplied.
	 * Forward propagation is used to update the neural network's values.
	 * @param input
	 * @return output
	 * @throws InvalidInputSizeException
	 */
	public double[] getOutput(double[] input) throws InvalidInputSizeException {
		if(input.length != inputLayer.size()) {
			throw new InvalidInputSizeException();
		}

		// Set the neural network's input
		for(int i = 0; i < input.length; i++) {
			inputLayer.get(i).setCurrentValue(input[i]);
		}
		// Update neurons layer by layer
		for(int i = 0; i < hiddenLayer1.size(); i++) {
			hiddenLayer1.get(i).updateCurrentValue();
		}
		for(int i = 0; i < outputLayer.size(); i++) {
			outputLayer.get(i).updateCurrentValue();
		}

		// Obtain the output from the output layer's data.
		double[] output = new double[outputLayer.size()];
		for(int i = 0; i < outputLayer.size(); i++) {
			output[i] = outputLayer.get(i).getCurrentValue();
		}

		return output;
	}

	/**
	 * Update the synapse weights using back propagation.
	 * @param input
	 * @param expectedOutput
	 * @throws InvalidInputSizeException 
	 */
	public void trainNetwork(double[] input, double[] expectedOutput, double learningRate) throws InvalidInputSizeException {
		// Calculate the cost of the network through forward propagation
		double[] overallOutput = this.getOutput(input);
		for(int i = 0; i < outputLayer.size(); i++) {
			Neuron neuron = outputLayer.get(i);
			double synapseError = neuron.getCurrentValue() - expectedOutput[i];
			neuron.updateDerivative(synapseError);
		}
		
		// Layer 1 back prop
		for(int i = 0; i < hiddenLayer1.size(); i++) {
			Neuron neuron = hiddenLayer1.get(i);
			for(int j = 0; j < neuron.getOutgoingSynapses().size(); j++) {
				Synapse synapse = neuron.getOutgoingSynapses().get(j);
				
				// Adjust weight by error
				synapse.setWeight(
						synapse.getWeight()
						-learningRate*neuron.getCurrentValue()*synapse.getToNeuron().getDerivative()
				);
			}
		}
		
		// Input layer back prop
		for(int i = 0; i < inputLayer.size(); i++) {
			Neuron neuron = inputLayer.get(i);
			for(int j = 0; j < neuron.getOutgoingSynapses().size(); j++) {
				Synapse synapse = neuron.getOutgoingSynapses().get(j);
				
				// Adjust weight by error
				synapse.setWeight(
						synapse.getWeight()
						-learningRate*neuron.getCurrentValue()*synapse.getToNeuron().getDerivative()
				);
			}
		}
	}
}
