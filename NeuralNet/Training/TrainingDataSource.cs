using System.Collections.Generic;

namespace NeuralNet.Training
{
    public abstract class TrainingDataSource
    {
        public TrainingDataSource(TrainingConfig config)
        {
            MinibatchLength = config.MinibatchSize;
        }

        public int Length { get; protected set; }

        public int SymbolsCount { get; protected set; }

        public int MinibatchLength { get; protected set; }

        public abstract MinibatchData GetData(int index);

        public struct MinibatchData
        {
            public List<float> InputSequence;
            public List<float> OutputSequence;
        }
    }
}