﻿namespace NeuralNet.Net;

/// <summary>
/// Represents a layer in a neural network.
/// </summary>
public class Layer
{
    /// <summary>
    /// Initializes a new instance of the Layer class with a specified number of neurons, inputs, and activation function.
    /// </summary>
    /// <param name="neuronCount"></param>
    /// <param name="inputCount"></param>
    /// <param name="activationFunction"></param>
    public Layer(int neuronCount, int inputCount, ActivationFunction activationFunction)
    {
        ActivationFunction = activationFunction;
        Neurons = new(neuronCount);

        for (int i = 0; i < neuronCount; i++)
        {
            Neurons.Add(new(inputCount));
        }
    }

    /// <summary>
    /// Gets the activation function of the layer.
    /// </summary>
    public ActivationFunction ActivationFunction { get; private set; }

    /// <summary>
    /// Gets the neurons in the layer.
    /// </summary>
    public List<Neuron> Neurons { get; private set; }

    /// <summary>
    /// Calculates the output of the layer given a list of inputs.
    /// </summary>
    /// <param name="inputs"></param>
    /// <returns></returns>
    public List<double> CalculateOutputs(List<double> inputs)
    {
        List<double> outputs = [];

        foreach (var neuron in Neurons)
        {
            outputs.Add(neuron.CalculateOutput(inputs, ActivationFunction));
        }

        return outputs;
    }
}