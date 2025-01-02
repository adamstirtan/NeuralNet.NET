using NueralNet.Net;

namespace NeuralNetwork.Runner;

public class XorProblem : IProblem
{
    public int InputSize => 2;
    public int OutputSize => 1;

    public IEnumerable<(List<double> Inputs, List<double> Targets)> GetTrainingData()
    {
        return new List<(List<double>, List<double>)>
        {
            ([0, 0], [0]),
            ([0, 1], [1]),
            ([1, 0], [1]),
            ([1, 1], [0])
        };
    }

    public List<double> PreprocessInputs(List<double> inputs)
    {
        return inputs;
    }

    public List<double> PostprocessOutputs(List<double> outputs)
    {
        return outputs.Select(output => output >= 0.5 ? 1.0 : 0.0).ToList();
    }
}