namespace NueralNet.Core;

public class ActivationFunction
{
    private string _name;

    public ActivationFunction(string name, Func<double, double> function, Func<double, double> derivative)
    {
        _name = name;

        Function = function;
        Derivative = derivative;
    }

    public Func<double, double> Function { get; }
    public Func<double, double> Derivative { get; }

    public static ActivationFunction Sigmoid => new("Sigmoid", x => 1 / (1 + Math.Exp(-x)), x => x * (1 - x));

    public static ActivationFunction Tanh => new("Tanh", x => Math.Tanh(x), x => 1 - Math.Pow(x, 2));

    public static ActivationFunction ReLU => new("ReLU", x => Math.Max(0, x), x => x > 0 ? 1 : 0);
}
