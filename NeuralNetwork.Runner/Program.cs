using System.Diagnostics;

using NeuralNetwork.Runner;
using NueralNet.Net;

XorProblem problem = new();

ANN network = new(
    [problem.InputSize, 3, problem.OutputSize],
    [ActivationFunction.Sigmoid, ActivationFunction.Sigmoid]
);

var trainingData = problem.GetTrainingData();

Stopwatch stopwatch = new();
stopwatch.Start();

network.Train(trainingData, epochs: 10000, learningRate: 0.2);

stopwatch.Stop();

Console.WriteLine($"Training Time: {stopwatch.ElapsedMilliseconds}ms\n");
Console.WriteLine("Testing Predictions:");

foreach (var (inputs, targets) in trainingData)
{
    var processedInputs = problem.PreprocessInputs(inputs);
    var outputs = network.Predict(processedInputs);
    var processedOutputs = problem.PostprocessOutputs(outputs);

    Console.WriteLine($"Inputs: {string.Join(", ", inputs)} | Predicted: {string.Join(", ", processedOutputs)} | Actual: {string.Join(", ", targets)}");
}