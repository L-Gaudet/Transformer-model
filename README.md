# CPSC 406: Algorithm Analysis - Transformer Model Implementation

## Student Contributors:

> Lucas Gaudet - Lead Researcher, Lead Transformer Developer
>
> Liam Propst - Self Attention and Algorithm Research, Unit Testing
>
> Matthew Graham - Transformer Model Research, Presentation
>
> Tyler Lewis - Documentation, Nvidia Docker server, Testing Loop



# How to run:

## Install dependencies:

    pip install -r requirements.txt

#

## To train:

**Note: On first run will download IMDB training set automatically** 


Python file `SentimentTraining.py` initiates the training algorithm 

**We recommend to use a Docker container using the included dockerfile. Our testing was completed on the Chapman nvidia-docker server.**

    docker build -t sentiment-container .

-> then create, start, and attach container to run. 

-> alternatively can run `python3 SentimentTraining.py` without setting up training environment.
#

### Running with a trained model file (TrainedClassifier):

    python3 SentimentRunner.py -f [model filename]

# Description

This repository showcases an implementation of the Transformer architecture, as introduced in the groundbreaking paper "Attention is All You Need" (2017). The main focus of this project is to utilize the Transformer model for sentiment analysis tasks.

The sentiment analysis model is trained on the PyTorch IMDB dataset, which consists of 50,000 movie reviews with labeled sentiments. By leveraging the power of the Transformer's self-attention mechanism, the model is capable of capturing intricate dependencies and patterns within textual data, leading to improved sentiment classification accuracy.

The implementation of the Transformer architecture can be found in the components directory of this repository. It encompasses various components such as attention mechanisms, positional encodings, feed-forward networks, and multi-head attention, all meticulously implemented in Python.

For a comprehensive understanding of the components, related data structures, and algorithms employed in the implementation, please refer to the `transformer-from-paper.ipynb` file located in the main directory of this repository.

This project aims to provide a solid foundation and inspire further advancements in sentiment analysis and Transformer-based models. With its open-source nature, detailed documentation, and carefully organized code, this repository serves as a valuable resource for developers interested in exploring and extending the capabilities of the Transformer architecture.

#

## Self-Attention Mechanism
The self-attention mechanism is a fundamental component of the Transformer architecture, enabling it to capture dependencies and relationships within an input sequence. The self-attention mechanism consists of scaled dot-product attention and multi-head attention.

### Scaled Dot-Product Attention
The scaled dot-product attention computes attention scores between the Query (Q) and Key (K) vectors and scales them by the square root of the dimension of the Query vector:

$$
Attention(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$


### Multi-Head Attention
To capture different relationships, the self-attention mechanism is often applied with multiple attention heads. Each attention head has its own sets of Query (Q), Key (K), and Value (V) vectors. The outputs from the multiple attention heads are concatenated and linearly transformed:


$$
MultiHead(Q, K, V) = \text{Concat}(head_1, ..., head_h)W^O
$$

### Positional Encoding
To incorporate positional information, the Transformer model utilizes sinusoidal functions to encode the position of each word in the input sequence. The positional encoding matrix PE is added to the input embeddings:

$$
PE_{(pos, i)} = 
    \begin{cases} 
        \sin\left(\frac{pos}{10000^{2i/d}}\right), & \text{if } i \text{ is even} \\
        \cos\left(\frac{pos}{10000^{2i/d}}\right), & \text{if } i \text{ is odd}
    \end{cases}
$$

Where:

pos: Position in the sequence
i: Dimension index of the positional encoding
d: Dimension of the input embeddings

These equations represent the key mathematical operations involved in the self-attention mechanism of the Transformer model. By incorporating these equations in your documentation using LaTeX syntax, you can provide a clear understanding of the underlying mathematics behind the sentiment analysis Transformer model.

#

[IMDB training set](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

[Attention is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
<!-- 


The project must be open source and on a public git repository. It must contain a file called LICENSE such as, for example, the MIT License. It also must contain a file .gitignore. See also Git best practices.
The repository must be structured in a way that makes it as easy as possible for a reader to access the relevant information. This includes proper use of markdown syntax.

There should be an introduction motivating the project and explaining why it is interesting (a good project has a convincing narrative).
There should be a literature review, references to related work and theoretical background (a good project describes the wider context in which it is situated). References are more useful when it is clear how they relate to the specifics of the project (just "dumping references at the end" is not useful).

The readme must contain a description of how to deploy and run the software.
The code must run, be well commented and documented.
The documentation should include, for example (adapt as appropriate):

What components does the software consist of? How do components interact?

What programming languages and APIs are used?

What data structures and algorithms did you implement?

How was the work divided between group members? Who was responsible for what?

Beware of Plagiarism: Make sure that if you took some code from somewhere you make clear, both in the code and in the documentation, from where you took it.

Give details of how the software was tested. Most projects should have code that was used for testing. Provide the tests that have been written, as well as a description of how to run the tests and reproduce the test test results.

Depending on the project, there may be other ways of validating the software (eg questionnaires, data analysis, and more).
Describe what works and what does not. Did your plans change? What is left to do?

Suggest directions for future development. Ideally, a list of possible extensions is described and designed, including details of how the current code base should be modified for the extension.

The last point is particularly important to me. A good open source project is one that inspires others to take it further and provides a basis for future developments. -->