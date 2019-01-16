using System.Collections.Generic;

namespace NeuralNet.Utilities
{
    public class Codec<T>
    {
        public Codec(IEnumerable<T> values)
        {
            int i = 0;
            valueToIndex = new Dictionary<T, int>();
            indexToValue = new Dictionary<int, T>();
            foreach (var v in values)
            {
                valueToIndex.Add(v, i);
                indexToValue.Add(i, v);
                i++;
            }
        }

        public int Count
        {
            get { return valueToIndex.Count; }
        }

        public int Encode(T value)
        {
            return valueToIndex[value];
        }

        public T Decode(int index)
        {
            return indexToValue[index];
        }

        private Dictionary<T, int> valueToIndex;

        private Dictionary<int, T> indexToValue;
    }
}