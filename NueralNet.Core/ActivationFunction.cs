﻿namespace NueralNet.Core;

/// <summary>
/// Represents an activation function used by a neuron in a neural network.
/// </summary>
public class ActivationFunction
{
    public string Name { get; }
    public Func<double, double> Function { get; }
    public Func<double, double> Derivative { get; }

    public ActivationFunction(string name, Func<double, double> function, Func<double, double> derivative)
    {
        Name = name;
        Function = function;
        Derivative = derivative;
    }

    public static ActivationFunction Sigmoid { get; } = new ActivationFunction(
        "Sigmoid",
        x => 1.0 / (1.0 + Math.Exp(-x)),
        y => y * (1 - y)
    );

    public static ActivationFunction Tanh { get; } = new ActivationFunction(
        "Tanh",
        x => Math.Tanh(x),
        y => 1 - y * y
    );

    public static ActivationFunction ReLU { get; } = new ActivationFunction(
        "ReLU",
        x => Math.Max(0, x),
        y => y > 0 ? 1 : 0
    );
}