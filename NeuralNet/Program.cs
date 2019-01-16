using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using NeuralNet.Prediction;
using NeuralNet.Training;

namespace NeuralNet
{
    class Program
    {
        static void Main(string[] args)
        {
            new ModelTrainer().PerformTraining(new TrainingConfig { TrainingFile = @"..\Resources\Data\tinyshakespeare.txt" });

            var modelPath = @"..\Resources\Model\shakespeare_epoch41.dnn";
            var symbols = LoadSymbols( @"..\Resources\Data\tinyshakespeare.txt");

            var predictor = new CharPredictor(modelPath, symbols);

            var textToTest = "He";
            var targetLength = 200;
            var sentence = predictor.Evaluate(textToTest, targetLength);
            
            Console.WriteLine(sentence);
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
