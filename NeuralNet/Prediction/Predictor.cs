using CNTK;
using NeuralNet.Utilities;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace NeuralNet.Prediction
{
    public class Predictor<T>
    {
        public Predictor(string modelPath, IEnumerable<T> symbols, bool seed = true)
            : this(symbols, seed)
        {
            SetDevice();
            this.model = Function.Load(modelPath, device);
        }

        public Predictor(Function model, IEnumerable<T> symbols, bool seed = true)
            : this(symbols, seed)
        {
            SetDevice();
            this.model = model;
        }

        Predictor(IEnumerable<T> symbols, bool seed = true)
        {
            this.codec = new Codec<T>(symbols);
            this.random = seed ? new Random() : new Random(0);
        }

        public IEnumerable<T> Evaluate(IEnumerable<T> inputData, int outputLength)
        {
            this.numberOfEvaluatedSymbols = 0;

            List<int> outputData = new List<int>();
            IList<IList<float>> output = null;
            List<float> sequence = new List<float>();

            foreach (var x in inputData)
            {
                var xIndex = codec.Encode(x);
                outputData.Add(xIndex);
            
                var input = new float[codec.Count];
                input[xIndex] = 1;
                sequence.AddRange(input);

                output = Evaluate(sequence);
                numberOfEvaluatedSymbols++;
            }

            for (int i = 0; i < outputLength - inputData.Count(); i++)
            {
                var suggestedSymbolIndex = GetRandomSuggestion(output[0].ToList());
                outputData.Add(suggestedSymbolIndex);

                var input = new float[codec.Count];
                input[suggestedSymbolIndex] = 1;
                sequence.AddRange(input);
                
                output = Evaluate(sequence);
                numberOfEvaluatedSymbols++;
            }

            return outputData.Select(x => codec.Decode(x)).ToArray();
        }

        private IList<IList<float>> Evaluate(List<float> sequence)
        {
            var inputVariable = model.Arguments.Single();
            var inputValue = Value.CreateSequence<float>(inputVariable.Shape, sequence, device);
            var inputs = new Dictionary<Variable, Value>() { { inputVariable, inputValue } };
            var output = new Dictionary<Variable, Value>() { { model.Output, null } };
            model.Evaluate(inputs, output, device);
            return output[model.Output].GetDenseData<float>(model.Output);;
        }

        private void SetDevice(DeviceDescriptor device = null)
        {
            if (device == null)
                this.device = DeviceDescriptor.UseDefaultDevice();
            else
                this.device = device;
        }
        
        private int GetRandomSuggestion(List<float> probabilities)
        {
            probabilities = probabilities.GetRange(codec.Count * (numberOfEvaluatedSymbols - 1), codec.Count);
            probabilities = probabilities.Select(f => (float)Math.Exp(f)).ToList();
            var sumOfProbabilities = probabilities.Sum();
            probabilities = probabilities.Select(x => x / sumOfProbabilities).ToList();
            var selection = 0;
            var randomValue = (float)random.NextDouble();
            foreach (var probability in probabilities)
            {
                randomValue -= probability;
                if (randomValue < 0) break;
                selection++;
            }
            return selection;
        }

        private int numberOfEvaluatedSymbols;

        private Random random;
        
        private Function model;

        private Codec<T> codec;

        private DeviceDescriptor device;
    }
}
