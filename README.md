# Sentiment Analysis using Transformers
Chapman University 2023

CPSC 406: Algorithm Analysis

Final Project
## Student Contributors:


> Lucas Gaudet
>
> Liam Propst
>
> Matthew Graham
>
> Tyler Lewis


# How to run:

## To train:

    python SentimentTraining.py

**Alternative: Using NVIDIA Docker:**

    nvidia-docker build -t sentiment-model .
 
**then**

    nvidia-docker run -v /nfshome/tylewis/Transformer-model/Transformer-model sentiment-model python3 SentimentTraining.py

## To run:

    python3 SentimentRunner.py

# Description

This repository contains an implementation of the Transformer architecture proposed in the paper Attention is all You Need (2017). Our implementation of this architecture is contained in the `components` directory of this repo.

#



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

The last point is particularly important to me. A good open source project is one that inspires others to take it further and provides a basis for future developments.