# Advancing POS Tagging: Baseline, Viterbi, and LSTM Models in NLP
Certainly, this problem involves improving Part-of-Speech (POS) tagging accuracy using the Brown Corpus. It has four parts:

1. **Baseline**: Establish a simple baseline accuracy.
2. **HMM**: Implement the Viterbi algorithm to enhance accuracy.
3. **LSTM**: Develop an LSTM model with different configurations to improve accuracy further.
4. **Evaluation**: Compare the results of the three approaches and evaluate them against the baseline.

The aim is to enhance POS tagging accuracy using progressively more advanced methods.

## Description

This project involves tackling the Part-of-Speech (POS) tagging task using various methods and techniques on the Brown Corpus dataset. The primary goal is to improve the accuracy of POS tagging compared to a simple baseline method. The project is divided into four main parts:

1. **Baseline Method (Part A)**: Initially, a straightforward baseline method is established, where each word is tagged with its most frequent POS tag. This basic approach sets the foundation for evaluating the effectiveness of more advanced techniques.

2. **Viterbi Algorithm (Part B)**: The Viterbi algorithm for Hidden Markov Models (HMM) is implemented to enhance POS tagging accuracy. Probability matrices (start_p, trans_p, emit_p) are calculated from the Brown Corpus to guide the tagging process. The accuracy of this algorithm is assessed.

3. **LSTM Model (Part C)**: A deep learning approach is introduced by developing a Long Short-Term Memory (LSTM) neural network for POS tagging using PyTorch. Pretrained GloVe embeddings are employed and fine-tuned with the corpus data. Various configurations of the LSTM layer, including hidden size, bidirectionality, number of layers, and dropout, are explored to optimize tagging accuracy.

4. **Evaluation and Comparison (Part D)**: Finally, the results obtained from the different methods (baseline, HMM, LSTM) are evaluated and compared in the context of the baseline method from Part A. This comprehensive analysis helps in understanding which approach yields the best performance for the POS tagging task.

Throughout the project, a simplified set of POS tags known as the "universal_tagset" is used, and careful attention is given to data preprocessing, hyperparameter tuning, and evaluation metrics. The goal is to uncover the most effective approach for accurately tagging parts of speech in sentences, with a focus on practical applications in natural language processing.

## Getting Started

### Dependencies

Here is a list of libraries and dependencies used in the code snippets you provided:

1. `numpy`: NumPy is used for numerical computations and handling arrays.

2. `nltk`: NLTK (Natural Language Toolkit) is a popular library for natural language processing tasks.

3. `math`: The `math` module provides mathematical functions and constants.

4. `torch`: PyTorch is a deep learning framework for building and training neural networks.

5. `torch.nn`: The `nn` module from PyTorch is used for defining and training neural network models.

6. `torch.optim`: The `optim` module from PyTorch provides various optimization algorithms.

7. `torch.utils.data`: This module from PyTorch is used for data handling and creating data loaders.

8. `sklearn`: Scikit-learn is a machine learning library that provides tools for various machine learning tasks.

9. `tqdm`: TQDM is a library for adding progress bars to loops and tasks.

10. `collections`: The `collections` module provides specialized container datatypes.

11. `nltk.corpus.brown`: The Brown Corpus from NLTK, a dataset used for training and testing natural language processing models.

12. `nltk.tokenize.word_tokenize`: Tokenization function from NLTK for splitting text into words or tokens.

13. `nltk.tag.untag`: Utility function in NLTK for removing tags from tagged text.

14. `nltk.util.ngrams`: Function in NLTK for generating n-grams from a sequence.

15. `collections.Counter`: Counter is used for counting occurrences of elements in a collection.

16. `sklearn.metrics.accuracy_score`: Function for calculating accuracy score in scikit-learn.

17. `sklearn.metrics.f1_score`: Function for calculating F1 score in scikit-learn.

18. `matplotlib.pyplot`: Matplotlib is a library for creating visualizations, and `pyplot` is a sub-library for creating plots and charts.

Please make sure to have these libraries installed in your Python environment to run the code successfully. You can typically install them using package managers like `pip` or `conda`.


### Installing

Before running the program, please follow these steps:

1. **Python Version**: Make sure you are using the latest version of Python. This code is developed and tested with Python 3.

2. **Environment Setup**: Ensure that your Python environment is properly set up.

3. **Dependencies**: You will need to have several Python libraries installed, including NumPy, NLTK, scikit-learn, torch, and matplotlib. You can install them using `pip` or `conda`:

   ```bash
   pip install numpy nltk scikit-learn torch matplotlib
   ```

4. **Data and Files**: Make sure that any downloaded files or datasets that your code relies on are properly linked or stored in the same directory as your master code. In this specific case, you mentioned using the Brown Corpus from NLTK, so ensure that you have downloaded it using:

   ```python
   nltk.download('brown')
   nltk.download('universal_tagset')
   ```

   Ensure that you have the necessary datasets and files in the expected locations for your code to access them without issues.

With these preparations, your program should be ready to run without modifications to files or folders.

Please download glove.6B.200d.txt file [here](https://www.kaggle.com/datasets/incorpes/glove6b200d)

## Authors

contact info

Mohan Thota 
mohant@bu.edu/mohan5thota@gmail.com

## Version History

* 0.1
    * Initial Release

## License

This project is licensed under the [APACHE 2.0] License - see the LICENSE.md file for details

## Acknowledgments

Guidance, code snippets, etc.
* [Proffessor](https://www.bu.edu/cs/profiles/wayne-snyder/)

