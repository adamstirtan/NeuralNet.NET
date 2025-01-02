namespace NueralNet.Core;

public class NeuralNetwork
{
    private readonly List<Layer> _layers;

    /// <summary>
    /// Initializes a new instance of the NeuralNetwork class with a specified structure and activation functions.
    /// </summary>
    /// <param name="structure"></param>
    /// <param name="activationFunctions"></param>
    /// <exception cref="ArgumentException"></exception>
    public NeuralNetwork(int[] structure, ActivationFunction[] activationFunctions)
    {
        if (structure.Length - 1 != activationFunctions.Length)
        {
            throw new ArgumentException("The number of activation functions must be one less than the number of layers.");
        }

        _layers = [];

        for (int i = 0; i < structure.Length - 1; i++)
        {
            _layers.Add(new(structure[i + 1], structure[i], activationFunctions[i]));
        }
    }

    /// <summary>
    /// Performs a forward pass through the network
    /// </summary>
    /// <param name="inputs">The inputs of the network.</param>
    /// <returns>The final output of the network.</returns>
    public List<double> Forward(List<double> inputs)
    {
        List<double> outputs = inputs;

        foreach (Layer layer in _layers)
        {
            outputs = layer.CalculateOutputs(outputs);
        }

        return outputs;
    }

    /// <summary>
    /// Trains the network using backpropagation and gradient descent.
    /// </summary>
    /// <param name="trainingData"></param>
    /// <param name="epochs"></param>
    /// <param name="learningRate"></param>
    public void Train(IEnumerable<(List<double> Inputs, List<double> Targets)> trainingData, int epochs, double learningRate)
    {
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            foreach (var (inputs, targets) in trainingData)
            {
                List<double> outputs = Forward(inputs);

                Backpropagate(inputs, targets, learningRate);
            }
        }
    }

    /// <summary>
    /// Predicts the output of the network given a list of inputs.
    /// </summary>
    /// <param name="inputs"></param>
    /// <returns></returns>
    public List<double> Predict(List<double> inputs)
    {
        return Forward(inputs);
    }

    /// <summary>
    /// Backpropogates the error through the network and updates the weights and biases.
    /// </summary>
    /// <param name="outputs">The actual outputs from the forward pass.</param>
    /// <param name="targets">The target outputs.</param>
    /// <param name="learningRate">The learning rate for gradient descent.</param>
    private void Backpropagate(List<double> inputs, List<double> targets, double learningRate)
    {
        // Forward pass to get activations for all layers
        List<List<double>> activations = new() { inputs };
        List<double> currentInputs = inputs;

        foreach (Layer layer in _layers)
        {
            List<double> layerOutputs = layer.CalculateOutputs(currentInputs);
            activations.Add(layerOutputs);
            currentInputs = layerOutputs;
        }

        // Initialize list to hold deltas for each layer
        List<List<double>> deltas = new();

        // Calculate delta for output layer
        int lastLayerIndex = _layers.Count - 1;
        List<double> outputActivations = activations[^1];
        List<double> outputDeltas = new();

        for (int i = 0; i < outputActivations.Count; i++)
        {
            double output = outputActivations[i];
            double error = output - targets[i];
            double delta = error * _layers[lastLayerIndex].ActivationFunction.Derivative(output);
            outputDeltas.Add(delta);
        }

        deltas.Insert(0, outputDeltas);

        // Calculate deltas for hidden layers
        for (int l = _layers.Count - 2; l >= 0; l--)
        {
            Layer currentLayer = _layers[l];
            Layer nextLayer = _layers[l + 1];
            List<double> currentActivations = activations[l + 1];
            List<double> layerDeltas = new();

            for (int i = 0; i < currentLayer.Neurons.Count; i++)
            {
                double sum = 0;
                for (int j = 0; j < nextLayer.Neurons.Count; j++)
                {
                    sum += deltas[0][j] * nextLayer.Neurons[j].Weights[i];
                }
                double activation = currentActivations[i];
                double delta = sum * currentLayer.ActivationFunction.Derivative(activation);
                layerDeltas.Add(delta);
            }

            deltas.Insert(0, layerDeltas);
        }

        // Update weights and biases
        for (int l = 0; l < _layers.Count; l++)
        {
            Layer layer = _layers[l];
            List<double> layerInputs = activations[l];
            List<double> layerDeltas = deltas[l];

            for (int i = 0; i < layer.Neurons.Count; i++)
            {
                Neuron neuron = layer.Neurons[i];

                // Update weights
                for (int j = 0; j < neuron.Weights.Count; j++)
                {
                    double gradient = layerDeltas[i] * layerInputs[j];
                    neuron.Weights[j] -= learningRate * gradient;
                }

                // Update bias
                neuron.Bias -= learningRate * layerDeltas[i];
            }
        }
    }
}