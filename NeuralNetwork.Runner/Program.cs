using System.Diagnostics;

using NeuralNetwork.Runner;
using NueralNet.Net;

const int EPOCHS = 10000;
const double LEARNING_RATE = 0.2;

XorProblem problem = new();

ANN network = new(
    [problem.InputSize, 3, problem.OutputSize],
    [ActivationFunction.Sigmoid, ActivationFunction.Sigmoid]
);

var trainingData = problem.GetTrainingData();

Stopwatch stopwatch = new();
stopwatch.Start();

network.Train(trainingData, EPOCHS, LEARNING_RATE);

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