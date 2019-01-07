using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using NeuralNet.Models;

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
