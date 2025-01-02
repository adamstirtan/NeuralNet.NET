using NueralNet.Net;

namespace NeuralNetwork.Runner;

public class SineWaveProblem : IProblem
{
    public int InputSize => 1;
    public int OutputSize => 1;

    public IEnumerable<(List<double> Inputs, List<double> Targets)> GetTrainingData()
    {
        var data = new List<(List<double>, List<double>)>();

        for (double x = 0; x <= 2 * Math.PI; x += 0.1)
        {
            data.Add(([x], [Math.Sin(x)]));
        }

        return data;
    }

    public List<double> PreprocessInputs(List<double> inputs)
    {
        return inputs;
    }

    public List<double> PostprocessOutputs(List<double> outputs)
    {
        return outputs;
    }
}