using NueralNet.Core;

var trainingData = new List<(List<double>, List<double>)>
{
    (new List<double> { 0, 0 }, new List<double> { 0 }),
    (new List<double> { 0, 1 }, new List<double> { 1 }),
    (new List<double> { 1, 0 }, new List<double> { 1 }),
    (new List<double> { 1, 1 }, new List<double> { 0 })
};

var nn = new NeuralNetwork(
    new[] { 2, 3, 1 },
    new[] { ActivationFunction.Sigmoid, ActivationFunction.Sigmoid });

nn.Train(trainingData, epochs: 10000, learningRate: 0.5);

Console.WriteLine("Testing XOR Predictions:");
Console.WriteLine($"0 XOR 0: {nn.Predict(new List<double> { 0, 0 })[0]}");
Console.WriteLine($"0 XOR 1: {nn.Predict(new List<double> { 0, 1 })[0]}");
Console.WriteLine($"1 XOR 0: {nn.Predict(new List<double> { 1, 0 })[0]}");
Console.WriteLine($"1 XOR 1: {nn.Predict(new List<double> { 1, 1 })[0]}");