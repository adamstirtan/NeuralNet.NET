namespace NeuralNet.Net.Tests;

[TestClass]
public sealed class Test1
{
    [TestMethod]
    public void ForwardTests()
    {
        ANN sut = new([2, 2, 1], [ActivationFunction.Sigmoid, ActivationFunction.Sigmoid]);

        var result = sut.Forward([1.0, 0.0]);

        Assert.IsNotNull(result);
        Assert.AreEqual(1, result.Count);
    }

    [TestMethod]
    public void TrainAndPredictTest()
    {
        ANN sut = new([2, 2, 1], [ActivationFunction.Sigmoid, ActivationFunction.Sigmoid]);

        var trainingData = new List<(List<double>, List<double>)>
        {
            ([0.0, 0.0], [0.0]),
            ([0.0, 1.0], [1.0])
        };

        sut.Train(trainingData, epochs: 1000, learningRate: 0.2);

        var output00 = sut.Predict([0.0, 0.0]);
        var output01 = sut.Predict([0.0, 1.0]);

        Assert.IsTrue(output00[0] < 0.5, "Expected near 0");
        Assert.IsTrue(output01[0] > 0.5, "Expected near 1");
    }
}
