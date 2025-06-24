### General description

This is a project formed of multiples small projects. The purpose of this project is to gain a basic understanding of languages models and their terms by following the multiple videos made by Andrej Karpathy on [Youtube](https://www.youtube.com/@AndrejKarpathy). More sources may be added later accordingly to understanding and interest.
See the [Youtube playlist](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)

## 1. Build Micrograd
Source : [Youtube](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)

## 2. Build Makemore
Source : [Youtube](https://www.youtube.com/watch?v=PaCmpygFfXo)
Source : [Youtube - Part 2](https://www.youtube.com/watch?v=TCH_1BHY58I)
Source : [Youtube - Part 3](https://www.youtube.com/watch?v=P6sfmUTpUmc)
Source : [Youtube - Part 4](https://www.youtube.com/watch?v=q8SA3rM6ckI)
Source : [Youtube - Part 5](https://www.youtube.com/watch?v=t3YJ5hKiMQ0)
Construction of:
- Bigram (one character predict the enxt one wit a lookup table of counts)
- Bags of words
- MLP
- RNN
- GRU
- Transformer (equivalent to GPT 2;with of course far less data and training)

=> This construct single words (in this case: "name"-like single words)

## 3. Build GPT
Source : [Youtube](https://www.youtube.com/watch?v=kCc8FmEb1nY)
Building a transformer purely decoder which take an input (text from Shakespear) and learn of to complete a given text accoding to its training. 

1. Get a a training set. In this case, a shakespear dataset.
2. Tokenized the dataset into tokens. Here :
    - Get the unique characters
    - Create a lookup table for convertir to and from the tokens' table.
3. Plug the token in the language model by using an embedding table. 65 embedding tokens would make 65 rows. Basically, each rows is a list of parameters for each unique characters. => a vector that feeds into the transformer. The parameters will be created as the vector evolve according to each usage new or repeated of the token in a sentence/string.

=> This construct multiples sentences (in this case: shakespear-like sentences.)

## 4. Build GPT tokenizer
Source : [Youtube](https://www.youtube.com/watch?v=zduSFxRajkE)
Opposed to the previous project, who was using a somewhat naïve approach to tokenization (single characters) in reality, the tokens are almost always chunk of characters. Theses token are composed with algorithms such as the bite-pair algorithm.