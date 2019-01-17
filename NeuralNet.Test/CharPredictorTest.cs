using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNet.Prediction;
using NeuralNet.Training;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace NeuralNet.Test
{
    [TestClass]
    public class CharPredictorTest
    {
        [TestMethod]
        public void EvaluateTest()
        {
            var modelPath = @"..\..\..\..\Resources\Model\shakespeare_epoch41.dnn";
            var symbols = LoadSymbols( @"..\..\..\..\Resources\Data\tinyshakespeare.txt");

            var predictor = new CharPredictor(modelPath, symbols, false);

            var textToTest = "He";
            var targetLength = 102;
            var sentence = predictor.Evaluate(textToTest, targetLength);
            
            var expected = "He! one's this is stabunt.\n\nMENENIUS:\nGo, behind this is must Yet outionvios!\n\nMARCIUS:\nThey find you\n";
            Assert.AreEqual(expected, sentence);
        }

        [TestMethod]
        public void TrainAndEvaluateTest()
        {
            var symbols = LoadSymbols(@"..\..\..\..\Resources\Data\tinyshakespeare.txt");

            var model = new ModelTrainer().PerformTraining(new TrainingConfig { TrainingFile = @"..\..\..\..\Resources\Data\tinyshakespeare.txt", Epochs = 1 });

            var predictor = new CharPredictor(model, symbols, false);

            var textToTest = "He";
            var targetLength = 102;
            var sentence = predictor.Evaluate(textToTest, targetLength);
            
            var expected = "Herech athy thous\r\nse's;'\r\nEre spacrebbses.\r\n\r\nOMENENIUS:\r\nO, true roce ye crutch and fight with the.\r";
            Assert.AreEqual(expected, sentence);
        }

        static IEnumerable<char> LoadSymbols(string filename)
        {
            var fi = new FileInfo("filename");

            var corpus = File.ReadAllText(filename);
            var symbols = corpus.Where(c => c != '\r').Distinct().ToList();
            symbols.Sort();

            return symbols;
        }
    }
}
