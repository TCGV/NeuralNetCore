using CNTK;
using NeuralNet;
using NeuralNet.Functions;
using NeuralNet.Utilities;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace NeuralNet.Training
{
    public class ModelTrainer
    {
        public ModelTrainer()
        {
            SetDevice();
        }

        public void PerformTraining(TrainingConfig config)
        {
            LoadData(config.TrainingFile);

            inputModel = CreateInputs(characters.Count);
            var modelSequence = CreateModel(characters.Count, 2, 256);
            model = modelSequence(inputModel.InputSequence);

            //  Setup the criteria (loss and metric)
            var crossEntropy = CNTKLib.CrossEntropyWithSoftmax(model, inputModel.LabelSequence);
            var errors = CNTKLib.ClassificationError(model, inputModel.LabelSequence);

            //  Instantiate the trainer object to drive the model training
            var learningRatePerSample = new TrainingParameterScheduleDouble(0.001, 1);
            var momentumTimeConstant = CNTKLib.MomentumAsTimeConstantSchedule(1100);
            var additionalParameters = new AdditionalLearningOptions
            {
                gradientClippingThresholdPerSample = 5.0,
                gradientClippingWithTruncation = true
            };
            var learner = Learner.MomentumSGDLearner(model.Parameters(), learningRatePerSample, momentumTimeConstant, true, additionalParameters);
            trainer = Trainer.CreateTrainer(model, crossEntropy, errors, new List<Learner>(){ learner });

            uint minibatchesPerEpoch = (uint) Math.Min(text.Length / config.MinibatchSize, config.MaxNumberOfMinibatches / config.Epochs);

            for (int i = 0; i < config.Epochs; i++)
            {
                TrainMinibatch(config, minibatchesPerEpoch, i);
                
                var modelFilename = $"newmodels/shakespeare_epoch{ i + 1 }.dnn";
                
                model.Save(modelFilename);
            }
        }

        private void TrainMinibatch(TrainingConfig config, uint minibatchesPerEpoch, int i)
        {
            for (int j = 0; j < minibatchesPerEpoch; j++)
            {
                var trainingData = GetData(j, config.MinibatchSize, text, characters.Count);

                var features = Value.CreateSequence<float>(inputModel.InputSequence.Shape,
                    trainingData.InputSequence, device);

                var arguments = new Dictionary<Variable, Value>();
                arguments.Add(inputModel.InputSequence, features);

                var labels = Value.CreateSequence(inputModel.LabelSequence.Shape,
                    trainingData.OutputSequence, device);

                arguments.Add(inputModel.LabelSequence, labels);
                trainer.TrainMinibatch(arguments, device);
            }
        }

        private InputModel CreateInputs(int vocabularyDimension)
        {
            var axis = new Axis("inputAxis");
            var inputSequence = Variable.InputVariable(new int[] { vocabularyDimension }, DataType.Float, "features", new List<Axis> { axis, Axis.DefaultBatchAxis() });
            var labels = Variable.InputVariable(new int[] { vocabularyDimension }, DataType.Float, "labels", new List<Axis> { axis, Axis.DefaultBatchAxis() });

            var inputModel = new InputModel
            {
                InputSequence = inputSequence,
                LabelSequence = labels
            };
            return inputModel;
        }

        private Func<Variable, Function> CreateModel(int numOutputDimension, int numLstmLayer, int numHiddenDimension)
        {
            return (input) =>
            {
                Function model = input;
                for (int i = 0; i < numLstmLayer; i++)
                {
                    model = Stabilizer.Build(model, device);
                    model = LSTM.Build(model, numHiddenDimension, device);
                }
                model = Dense.Build(model, numOutputDimension, device);
                return model;
            };
        }

        private MinibatchData GetData(int index, int minibatchSize, string data, int vocabDimension)
        {
            var inputString = data.Substring(index, minibatchSize);
            var outputString = data.Substring(index + 1, minibatchSize);

            //  Handle EOF
            if (outputString.Length < minibatchSize)
                minibatchSize = outputString.Length;
            inputString = inputString.Substring(0, minibatchSize);

            List<float> inputSequence = new List<float>();
            List<float> outputSequence = new List<float>();

            for (int i = 0; i < inputString.Length; i++)
            {
                var inputCharacterIndex = codec.Encode(inputString[i]);
                var inputCharOneHot = new float[vocabDimension];
                inputCharOneHot[inputCharacterIndex] = 1;
                inputSequence.AddRange(inputCharOneHot);

                var outputCharacterIndex = codec.Encode(outputString[i]);
                var outputCharOneHot = new float[vocabDimension];
                outputCharOneHot[outputCharacterIndex] = 1;
                outputSequence.AddRange(outputCharOneHot);
            }

            return new MinibatchData
            {
                InputSequence = inputSequence,
                OutputSequence = outputSequence
            };
        }

        private void LoadData(string filename)
        {
            text = File.ReadAllText(filename);
            characters = text.Distinct().ToList();
            characters.Sort();

            codec = new Codec<char>(characters);
        }
        
        private void SetDevice(DeviceDescriptor device = null)
        {
            if (device == null) this.device = DeviceDescriptor.UseDefaultDevice();
            else this.device = device;
        }

        internal struct InputModel
        {
            public Variable InputSequence;
            public Variable LabelSequence;
        }

        internal struct MinibatchData
        {
            public List<float> InputSequence;
            public List<float> OutputSequence;
        }
        
        private InputModel inputModel;
        private Function model;
        private Trainer trainer;
        
        private string text;
        private List<char> characters;
        private Codec<char> codec;

        private DeviceDescriptor device;
    }
}
