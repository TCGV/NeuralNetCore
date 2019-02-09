using CNTK;
using NeuralNet.Functions;
using System;
using System.Collections.Generic;

namespace NeuralNet.Training
{
    public class ModelTrainer
    {
        public ModelTrainer()
        {
            SetDevice();
        }

        public Function PerformTraining(TrainingConfig config, TrainingDataSource source)
        {
            this.source = source;

            var modelSequence = CreateModel(source.SymbolsCount, 2, 256);
            inputModel = CreateInputs(source.SymbolsCount);
            model = modelSequence(inputModel.InputSequence);
            
            return PerformTraining(config);
        }

        private Function PerformTraining(TrainingConfig config)
        {
            //  Setup the criteria (loss and metric)
            var crossEntropy = CNTKLib.CrossEntropyWithSoftmax(model, inputModel.LabelSequence);
            var errors = CNTKLib.ClassificationError(model, inputModel.LabelSequence);

            //  Instantiate the trainer object to drive the model training
            var learningRatePerSample = new TrainingParameterScheduleDouble(config.LearningRate, 1);
            var momentumTimeConstant = CNTKLib.MomentumAsTimeConstantSchedule(config.MomentumTimeConstant);
            var additionalParameters = new AdditionalLearningOptions
            {
                gradientClippingThresholdPerSample = 5.0,
                gradientClippingWithTruncation = true
            };
            var learner = Learner.MomentumSGDLearner(model.Parameters(), learningRatePerSample, momentumTimeConstant, true, additionalParameters);
            trainer = Trainer.CreateTrainer(model, crossEntropy, errors, new List<Learner>() { learner });

            for (int i = 0; i < config.Epochs; i++)
                TrainMinibatch(config);

            return model;
        }

        private void TrainMinibatch(TrainingConfig config)
        {
            uint minibatchesPerEpoch = (uint) Math.Min(source.Length / config.MinibatchSize, config.MaxNumberOfMinibatches / config.Epochs);

            for (int j = 0; j < minibatchesPerEpoch; j++)
            {
                var trainingData = source.GetData(j);

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
        
        private DeviceDescriptor device;
        private Function model;
        private Trainer trainer;
        private TrainingDataSource source;
        private InputModel inputModel;
    }
}
