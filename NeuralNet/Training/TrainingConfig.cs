using System;

namespace NeuralNet.Training
{
    public class TrainingConfig
    {
        public TrainingConfig()
        {
            Epochs = 50;
            MinibatchSize = 100;
            MaxNumberOfMinibatches = int.MaxValue;
            SampleFrequency = 1000;
            LearningRate = 0.01;
            MomentumTimeConstant = 1100;
        }

        public int Epochs { get; set; }
        public int MinibatchSize { get; set; }
        public int MaxNumberOfMinibatches { get; set; }
        public int SampleFrequency { get; set; }
        public double LearningRate { get; internal set; }
        public double MomentumTimeConstant { get; internal set; }
    }
}