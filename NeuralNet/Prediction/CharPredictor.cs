using CNTK;
using NeuralNet.Utilities;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace NeuralNet.Prediction
{
    public class CharPredictor
    {
        public CharPredictor(string modelPath, IEnumerable<char> symbols, bool seed = true)
            : this(symbols, seed)
        {
            SetDevice();
            this.model = Function.Load(modelPath, device);
        }

        public CharPredictor(Function model, IEnumerable<char> symbols, bool seed = true)
            : this(symbols, seed)
        {
            SetDevice();
            this.model = model;
        }

        CharPredictor(IEnumerable<char> symbols, bool seed = true)
        {
            this.codec = new Codec<char>(symbols);
            this.random = seed ? new Random() : new Random(0);
        }

        public string Evaluate(string inputText, int outputLength)
        {
            this.numberOfEvaluatedCharacters = 0;

            List<int> textOutput = new List<int>();
            IList<IList<float>> output = null;
            List<float> sequence = new List<float>();

            for (int i = 0; i < inputText.Length; i++)
            {
                var c = inputText[i];
                var cAsIndex = codec.Encode(c);
                textOutput.Add(cAsIndex);
            
                var input = new float[codec.Count];
                input[cAsIndex] = 1;
                sequence.AddRange(input);

                output = Evaluate(sequence);
                numberOfEvaluatedCharacters++;
            }

            for (int i = 0; i < outputLength - inputText.Length; i++)
            {
                var suggestedCharIndex = GetRandomSuggestion(output[0].ToList());
                textOutput.Add(suggestedCharIndex);

                var input = new float[codec.Count];
                input[suggestedCharIndex] = 1;
                sequence.AddRange(input);
                
                output = Evaluate(sequence);
                numberOfEvaluatedCharacters++;
            }

            List<char> sentenceAsChar = textOutput.Select(x => codec.Decode(x)).ToList();
            string sentence = string.Join("", sentenceAsChar.ToArray());
            return sentence;
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
            probabilities = probabilities.GetRange(codec.Count * (numberOfEvaluatedCharacters - 1), codec.Count);
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

        private int numberOfEvaluatedCharacters;

        private Random random;
        
        private Function model;

        private Codec<char> codec;

        private DeviceDescriptor device;
    }
}
