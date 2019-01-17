# NeuralNetCore
Porting of the Microsoft Cognitive Toolkit (CNTK) Recurrent Neural Network (RNN) [python example](https://github.com/Microsoft/CNTK/blob/master/Examples/Text/CharacterLM/char_rnn.py) to .NET Core

a) Download CNTK version 2.6 (Windows+Linux) from:
- https://github.com/Microsoft/CNTK/releases

b) Extract it to `C:\Local\cntk-2.6`

c) Add the binarie path `C:\Local\cntk-2.6\cntk` to the `PATH` environment variable

d) Install `Microsoft C++ Redistributable` 2013 and `Microsoft C++ Redistributable 2017`

e) Run the test project `dotnet test --framework netcoreapp2.1`

Refactored from:
- https://github.com/albertalrisa/cntk-csharp-rnn
