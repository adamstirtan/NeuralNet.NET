namespace NeuralNet.Net;

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
    public double Bias { get; set; }

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
}