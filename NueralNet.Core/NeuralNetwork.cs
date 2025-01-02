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

                Backpropagate(inputs, outputs, targets, learningRate);
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
    private void Backpropagate(List<double> inputs, List<double> outputs, List<double> targets, double learningRate)
    {
        // Calculate output layer error (delta)
        List<double> outputErrors = new();
        for (int i = 0; i < outputs.Count; i++)
        {
            double error = targets[i] - outputs[i];
            outputErrors.Add(error * _layers[^1].ActivationFunction.Derivative(outputs[i]));
        }

        // Backpropagate the error through the layers
        List<List<double>> layerErrors = new() { outputErrors };
        for (int i = _layers.Count - 2; i >= 0; i--)
        {
            List<double> currentLayerErrors = new();
            Layer currentLayer = _layers[i];
            Layer nextLayer = _layers[i + 1];
            List<double> nextLayerErrors = layerErrors[0];

            for (int j = 0; j < currentLayer.Neurons.Count; j++)
            {
                double error = 0;
                for (int k = 0; k < nextLayer.Neurons.Count; k++)
                {
                    error += nextLayerErrors[k] * nextLayer.Neurons[k].Weights[j];
                }
                currentLayerErrors.Add(error * currentLayer.ActivationFunction.Derivative(currentLayer.Neurons[j].Output));
            }
            layerErrors.Insert(0, currentLayerErrors);
        }

        // Update weights and biases
        List<double> layerInputs = inputs;
        for (int i = 0; i < _layers.Count; i++)
        {
            Layer layer = _layers[i];
            List<double> errors = layerErrors[i];

            if (i > 0)
            {
                layerInputs = _layers[i - 1].CalculateOutputs(layerInputs);
            }

            for (int j = 0; j < layer.Neurons.Count; j++)
            {
                Neuron neuron = layer.Neurons[j];
                List<double> weightGradients = new();
                for (int k = 0; k < neuron.Weights.Count; k++)
                {
                    weightGradients.Add(errors[j] * layerInputs[k]);
                }
                List<double> biasGradients = new() { errors[j] };
                neuron.UpdateParameters(weightGradients, biasGradients, learningRate);
            }
        }
    }
}