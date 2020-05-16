package main.function;

public class SigmoidFunction extends Function{
	
	public double getOutput(double input) {
		return (1/( 1 + Math.pow(Math.E, (-1*input))));
	}

	public double getDerivativeOutput(double input) {
		double fx = getOutput(input);
		return fx*(1-fx);
	}
}
