# NeuralNet.Net

Simple C# library for creating and training neural networks. Allows building multi-layer networks with customizable activation functions.

## Installation
Install via NuGet or include the source code in your project.

## Example
```csharp
using NeuralNet.Net;

class Program
{
    static void Main()
    {
        // Create a network with 2 input neurons, 2 hidden, and 1 output
        var nn = new ANN(new int[] { 2, 2, 1 }, 
            new ActivationFunction[] { ActivationFunction.Sigmoid, ActivationFunction.Sigmoid });
        
        // Training data (XOR-like)
        var trainingData = new List<(List<double>, List<double>)>
        {
            (new List<double> {0,0}, new List<double> {0}),
            (new List<double> {0,1}, new List<double> {1}),
            (new List<double> {1,0}, new List<double> {1}),
            (new List<double> {1,1}, new List<double> {0}),
        };

        nn.Train(trainingData, epochs: 1000, learningRate: 0.1);

        // Predict
        var result = nn.Predict(new List<double> {1,1});
        Console.WriteLine($"Output: {result[0]}");
    }
}