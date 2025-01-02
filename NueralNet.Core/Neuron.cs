namespace NueralNet.Core;

/// <summary>
/// Represents a neuron in a neural network.
/// </summary>
public class Neuron
{
    private static readonly Random _random = new();

    /// <summary>
    /// Initializes a new instance of the Neuron class with a specified number of inputs.
    /// </summary>
    /// <param name="inputCount"></param>
    public Neuron(int inputCount)
    {
        Weights = new(inputCount);

        for (int i = 0; i < inputCount; i++)
        {
            Weights.Add(_random.NextDouble() * 2 - 1);
        }

        Bias = _random.NextDouble() * 2 - 1;
    }

    /// <summary>
    /// Gets the weights of the neuron.
    /// </summary>
    public List<double> Weights { get; private set; }

    /// <summary>
    /// Gets the bias of the neuron.
    /// </summary>
    public double Bias { get; private set; }

    /// <summary>
    /// Gets the output of the neuron.
    /// </summary>
    public double Output { get; private set; }

    /// <summary>
    /// Calculates the output of the neuron given a list of inputs and an activation function.
    /// </summary>
    /// <param name="inputs"></param>
    /// <param name="activationFunction"></param>
    /// <returns></returns>
    public double CalculateOutput(List<double> inputs, ActivationFunction activationFunction)
    {
        double sum = 0;

        for (int i = 0; i < inputs.Count; i++)
        {
            sum += inputs[i] * Weights[i];
        }

        sum += Bias;

        Output = activationFunction.Function(sum);

        return Output;
    }

    /// <summary>
    /// Updates the weights and bias of the neuron given the gradients and a learning rate.
    /// </summary>
    /// <param name="weightGradients"></param>
    /// <param name="biasGradients"></param>
    /// <param name="learningRate"></param>
    public void UpdateParameters(List<double> weightGradients, List<double> biasGradients, double learningRate)
    {
        for (int i = 0; i < Weights.Count; i++)
        {
            Weights[i] -= learningRate * weightGradients[i];
        }

        Bias += learningRate * biasGradients[0];
    }
}
