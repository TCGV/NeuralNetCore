using NeuralNet.Utilities;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace NeuralNet.Training
{
    public class TextDataSource : TrainingDataSource
    {
        public TextDataSource(TrainingConfig config, string filePath)
            : base(config)
        {
            LoadData(filePath);
        }

        public override TrainingDataSource.MinibatchData GetData(int index)
        {
            var inputString = text.Substring(index, MinibatchLength);
            var outputString = text.Substring(index + 1, MinibatchLength);

            //  Handle EOF
            if (outputString.Length < MinibatchLength)
                MinibatchLength = outputString.Length;
            inputString = inputString.Substring(0, MinibatchLength);

            List<float> inputSequence = new List<float>();
            List<float> outputSequence = new List<float>();

            for (int i = 0; i < inputString.Length; i++)
            {
                var inputCharacterIndex = codec.Encode(inputString[i]);
                var inputCharOneHot = new float[codec.Count];
                inputCharOneHot[inputCharacterIndex] = 1;
                inputSequence.AddRange(inputCharOneHot);

                var outputCharacterIndex = codec.Encode(outputString[i]);
                var outputCharOneHot = new float[codec.Count];
                outputCharOneHot[outputCharacterIndex] = 1;
                outputSequence.AddRange(outputCharOneHot);
            }

            return new TrainingDataSource.MinibatchData
            {
                InputSequence = inputSequence,
                OutputSequence = outputSequence
            };
        }

        private void LoadData(string filePath)
        {
            this.text = File.ReadAllText(filePath);
            var symbols = text.Distinct().ToList();
            symbols.Sort();
            this.codec = new Codec<char>(symbols);
            this.SymbolsCount = symbols.Count;
            this.Length = text.Length;
        }

        private string text;
        private Codec<char> codec;
    }
}