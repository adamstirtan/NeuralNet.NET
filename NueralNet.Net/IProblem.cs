namespace NueralNet.Net;

public interface IProblem
{
    /// <summary>
    /// Gets the number of input features.
    /// </summary>
    int InputSize { get; }

    /// <summary>
    ///  Gets the number of output values.
    /// </summary>
    int OutputSize { get; }

    /// <summary>
    /// Retrieves the training data as an enumerable of input-target pairs.
    /// </summary>
    IEnumerable<(List<double> Inputs, List<double> Targets)> GetTrainingData();

    /// <summary>
    /// Optionally preprocesses input data if necessary.
    /// </summary>
    /// <param name="inputs">The raw input data.</param>
    /// <returns>The preprocessed input data.</returns>
    List<double> PreprocessInputs(List<double> inputs);

    /// <summary>
    /// Optionally postprocesses output data if necessary.
    /// </summary>
    /// <param name="outputs">The raw output data from the network.</param>
    /// <returns>The postprocessed output data.</returns>
    List<double> PostprocessOutputs(List<double> outputs);
}