🧠 Automatic Classification of Crisis Tweets Using GloVe and BiLSTM
<div align="center">
Deep Learning for Crisis-Related Social Media Text Classification
A deep learning-based approach for automatically classifying crisis-related tweets into informative and non-informative categories using GloVe word embeddings and a Bidirectional Long Short-Term Memory (BiLSTM) network.





</div>
📖 Overview
Social media platforms, particularly Twitter (now X), have become important sources of real-time information during natural disasters and humanitarian crises.
However, the massive volume of crisis-related messages makes manual information extraction and classification challenging. Identifying informative tweets from large collections of social media data can help humanitarian organizations and emergency response teams access relevant information more efficiently.
This research proposes a deep learning approach combining Global Vectors for Word Representation (GloVe) and Bidirectional Long Short-Term Memory (BiLSTM) to automatically classify crisis-related tweets into two categories:
Informative: Tweets containing useful information relevant to crisis response.
Non-informative: Tweets that do not provide actionable or relevant crisis-related information.
The proposed model is evaluated using the CrisisBench dataset and a Stratified 5-Fold Cross-Validation strategy.
🎯 Research Objectives
The main objectives of this research are:
Develop a deep learning model for binary classification of crisis-related tweets.
Integrate GloVe word embeddings with a BiLSTM architecture.
Evaluate the proposed model on the CrisisBench dataset.
Assess model stability using Stratified 5-Fold Cross-Validation.
Analyze classification performance using multiple evaluation metrics.
Examine the potential of deep learning for supporting crisis information management.
📊 Dataset
CrisisBench is a benchmark collection of crisis-related social media datasets designed for humanitarian information processing.
Dataset Property
Description
Dataset
CrisisBench
Number of Samples
109,796 tweets
Language
English
Classification Type
Binary Classification
Target Classes
Informative / Non-informative
Data Type
Crisis-related Twitter messages
The research uses the English-language portion of CrisisBench for binary tweet classification.
⚙️ Methodology
The proposed methodology consists of several stages:
1. Data Preprocessing
Raw tweets undergo preprocessing to reduce noise and standardize textual input.
The preprocessing pipeline includes:
Removing URLs
Removing user mentions
Removing hashtags
Removing stopwords
Tokenization
Sequence length normalization
The maximum sequence length is set to 50 tokens.
2. Word Embedding: GloVe
GloVe (Global Vectors for Word Representation) is used to convert textual tokens into dense numerical vectors.
Configuration:
Embedding dimension: 300
Vocabulary size: 15,000
Embedding matrix: 15,000 × 300
These representations provide the neural network with semantic information about words and their relationships.
3. Deep Learning Architecture: BiLSTM
A Bidirectional LSTM network is employed to capture contextual dependencies in both forward and backward directions.
The proposed architecture consists of:
First BiLSTM layer: 256 units
Dropout: 0.4
Second BiLSTM layer: 128 units
Dropout: 0.4
Dense layer: 64 units
Dropout: 0.4
Output layer: Sigmoid activation
Training configuration:
Optimizer: RMSprop
Loss function: Binary Cross-Entropy
Classification: Binary
Evaluation: Stratified 5-Fold Cross-Validation
🏗 Model Architecture
          Input Tweets
                |
                v
        Text Preprocessing
                |
                v
          Tokenization
                |
                v
       Sequence Padding
         (50 Tokens)
                |
                v
       GloVe Embedding
          (300-Dim)
                |
                v
          BiLSTM Layer
            (256 Units)
                |
                v
           Dropout
             (0.4)
                |
                v
          BiLSTM Layer
            (128 Units)
                |
                v
           Dropout
             (0.4)
                |
                v
          Dense Layer
             (64)
                |
                v
           Dropout
             (0.4)
                |
                v
         Sigmoid Output
                |
                v
       Binary Classification
                |
        +-------+-------+
        |               |
        v               v
    Informative     Non-informative
📈 Evaluation Strategy
To evaluate the proposed model and examine its stability, Stratified 5-Fold Cross-Validation is employed.
The dataset is divided into five folds while maintaining class distribution across the splits.
The following evaluation metrics are considered:
Metric
Purpose
Accuracy
Overall classification correctness
Precision
Reliability of positive predictions
Recall
Ability to identify informative tweets
F1-Score
Balance between precision and recall
AUC-ROC
Discriminative capability of the model
Mean and standard deviation across folds are used to analyze performance consistency.
🔬 Comparative Analysis
The research also investigates the performance of the proposed GloVe + BiLSTM architecture in comparison with a more complex hybrid architecture:
GloVe + BiGRU-LSTM
The comparison focuses on classification performance and model complexity.
The study also discusses previous research while accounting for differences in datasets, classification tasks, and experimental protocols.
📌 Research Contributions
The main contributions of this research include:
Implementation of a GloVe-based BiLSTM architecture for binary classification of crisis-related tweets.
Evaluation on the English-language portion of CrisisBench containing 109,796 samples.
Application of Stratified 5-Fold Cross-Validation to assess model stability.
Analysis of multiple classification metrics.
Comparative evaluation with a more complex recurrent neural network architecture.
Discussion of research limitations and potential directions for future work.
⚠️ Research Limitations
The study identifies several limitations:
Only textual features are considered.
The experiments focus on English-language tweets.
Transformer-based models such as BERT are not included in the experimental comparison.
Ablation studies for hyperparameter optimization have not been conducted.
The evaluation protocol differs from the official CrisisBench train/dev/test split.
🔮 Future Research Directions
Potential extensions of this research include:
Comparing BiLSTM with BERT, DistilBERT, and RoBERTa.
Incorporating network-based features such as followers and retweets.
Extending the classification framework to multilingual crisis datasets.
Applying transfer learning to emerging crisis events.
Conducting ablation studies on model architecture and hyperparameters.
Developing a real-time crisis tweet classification system.
🛠 Technologies & Research Areas
Core Research Areas
Natural Language Processing (NLP)
Deep Learning
Text Classification
Word Embeddings
Recurrent Neural Networks
Crisis Informatics
Social Media Analytics
Methods & Models
GloVe
BiLSTM
BiGRU-LSTM
Stratified Cross-Validation
Binary Classification
📚 References
Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global Vectors for Word Representation. EMNLP 2014.
Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation.
Schuster, M., & Paliwal, K. K. (1997). Bidirectional Recurrent Neural Networks. IEEE Transactions on Signal Processing.
Alam, F., Sajjad, H., Imran, M., & Ofli, F. (2021). CrisisBench: Benchmarking Crisis-related Social Media Datasets for Humanitarian Information Processing. ICWSM 2021.
👩‍💻 Researcher
Bahareh Ghalenoii
Research Interests:
Natural Language Processing
Deep Learning
Large Language Models (LLMs)
Machine Learning
Crisis Informatics
Text Classification
GitHub: https://github.com/BaharehGhalenoii
LinkedIn: https://www.linkedin.com/in/bahareh-ghalenoii
<div align="center">
Advancing Crisis Information Processing Through Deep Learning
</div>
