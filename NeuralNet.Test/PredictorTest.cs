using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNet.Prediction;
using NeuralNet.Training;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace NeuralNet.Test
{
    [TestClass]
    public class PredictorTest
    {
        [TestMethod]
        public void PerformTrainingTest()
        {
            var config  = new TrainingConfig { Epochs = 5 };
            var source = new TextDataSource(config, @"..\..\..\..\Resources\Data\tinyshakespeare.txt");
            
            var model = new ModelTrainer().PerformTraining(config, source);
            model.Save(@"..\..\..\..\Resources\Model\shakespeare_epoch5.dnn");
        }

        [TestMethod]
        public void EvaluateTest()
        {
            var modelPath = @"..\..\..\..\Resources\Model\shakespeare_epoch41.dnn";
            var symbols = LoadSymbols( @"..\..\..\..\Resources\Data\tinyshakespeare.txt");

            var predictor = new Predictor<char>(modelPath, symbols, false);

            var textToTest = "He";
            var targetLength = 102;
            var sentence = string.Join("", predictor.Evaluate(textToTest, targetLength).ToArray());
            
            var expected = "He! one's this is stabunt.\n\nMENENIUS:\nGo, behind this is must Yet outionvios!\n\nMARCIUS:\nThey find you\n";
            Assert.AreEqual(expected, sentence);
        }

        static IEnumerable<char> LoadSymbols(string filename)
        {
            var corpus = File.ReadAllText(filename);
            var symbols = corpus.Where(c => c != '\r').Distinct().ToList();
            symbols.Sort();

            return symbols;
        }
    }
}
