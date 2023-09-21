## Overview:
This is a personalized repo aiming Artificial Intelligence and Machine Learning Study and Quick Jump Start Guide.

> A.I: The effort to automate intellectual tasks normally performed by humans. From tic-tac-toe games to more complex things that simulate some form of intelligence.
- Narrow AI: AI doing specific kind of tasks are better than human level
- General Inelegance: We're far from the General level Human Intelligence

> M.L: Machine learning is a subset of AI. It is to make computer act without being explicitly programed. It uses data from tons of examples instead of hard coded programme and predict the best case scenarios. It not 100% correct every time. To get the highest accuracy is the goal of machine learning model/algorithm. So Rather than giving the program the rules, the algorithm finds the rule for us.

> Depp Learning || Deep Neural Networks: It's one of the technique to implement Machine Learning. It's a type of Algorithm.


> Neural Network: A form of machine learning that uses layered representation of data. In NN we have multiple layer of data. 
"Neural Network is a multistage information extraction process"

Feature & Label: Feature is the input information. Label is the prediction (output) based on Feature.

Data: Combination of all the Features and labels (training phase). 
### ML and Data Science:
Machine Learning Depends on Data Science to get better understanding of the underlying tons of data.


### Machine Learning Types:
- Unsupervised Learning
    When we only have features and no label. We let model group the features on their position
    - Clustering:
    - Association Rule Learning:

- Supervised Learning:
    It has both Features and Label data for model training. Most common type.
    - Classification: Prediction from 2 or more options. Binary (two options), Multi-class (2+ options)
    - Regression: Prediction of a number (i.e property sell price)

- Reinforcement Learning || Real time learning :
    When we have no data (No Features and No label). Agent, Environment, Exploring and Reward (Negative and Positive). Here we force machine to learn by providing it Reward or Punishment. The practicality is to let A.I to play game

- Transfer Learning:
    Transferring one model to do another task by fine-tuning. Like in image detection model, other than the subject, every thing is trained. So we can fine tune (build new on top of the old) the model to detect another subject rather/along with the original/old 
### Machine Learning Project Implementation Steeps:
* Data Collection
* Data Modeling:
 1. Problem Definition: What problem to solve. Is the problem SuperVised or UnSupervised or Classification or Regression
 2. Data: What kind of data we have? Structured (rows and column data or database entries) or Unstructured (images, audio)
    - Static Data: Non changing data, like csv data
    - Streaming Data: Changing Data, like Stock 
 3. Evaluation: What defines success of us? Find out the best model to get at least 90% accuracy to do prediction.
 4. Features: What do we already know about the data? Features help us find pattern to get a prediction
    - Numerical Features (numbers)
    - Categorical features (like person's sex, race, etc)
    - Derived Features || Feature Engineering : Creating new features from the data
    - Feature Coverage : the prediction works best when a feature is consistent through the data. or over 10% coverage.
 5. Modeling: Based on problem data, what model should we use?
    - Choosing Model -> Tuning a model (hyperparameter) -> Model Comparison
    - 3 sets : Trining (70/80%), validation (15/10%), test(15/10%) for a data collection
    - Model selection:
        - For Structured Data : CatBoost, dmlc XGBoost, Random Forest work best
        - For Unstructured Data : Deep Learning / Neural Network, Transfer Learning Work best.
    - testing: 
        - Balanced (Gold-i-locks zone) : Training prediction 98% vs Test Prediction 96%. It's an iterative steeps to reach this stage
        - Under-fitting: 64% (training data prediction) vs 47% test data prediction. When model is trained less or loosely
        - Over-fitting: Training 93% vs Test 99%. This means the model is over trained (the model is behaving explicitly with the dataset)
        - To solve : Use more advanced model, increase model hyperparameter, Reduce amount of features, Train longer, collect more data or use less advanced model
    NB: 
    - Over-fitting and Under-fitting: this happens then there is some data leakage (Overlap between Training Data and Test Data). Training data, validation data and test data should be well separated.. 
    - Data Mismatch: When different kind of data are used in Training/Test Stage. The data should be same kind
 6.  Experimentation: How could we improve
* Deployment

### Machine Learning Models (Algorithm)