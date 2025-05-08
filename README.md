# PREDICTIVE-ANALYSIS-USING-MACHINE-LEARNING

COMPANY : CODTECH IT SOLUTIONS 

NAME : ABIRAMI S 

INTERN ID : CT04DL633 

DOMAIN : DATA ANALYTICS DURATION : 4 WEEKS

# DISCRIPTION :

This repository uses PySpark to predict user engagement (likes) on AI-generated images based on features like generation time, GPU usage, file size, style accuracy, and platform. The analysis was performed on Google Colab, leveraging PySpark's distributed computing capabilities.

Key Steps
Data Loading and Preprocessing:

Loaded the dataset into a Spark DataFrame.

Cleaned the data by removing rows with missing values.

Encoded the platform column into numerical values for modeling.

Feature Engineering:

Selected relevant features and assembled them into a single feature vector using PySpark's VectorAssembler.

Model Training:

Trained a Random Forest Regressor to predict the number of likes.

Used 80% of the data for training and 20% for testing.

Evaluation:

Evaluated the model using Root Mean Squared Error (RMSE) to measure prediction accuracy.

#OUTPUT : 

![Image](https://github.com/user-attachments/assets/da6a5467-7532-45a2-a18b-2b090cbe560f)

Feature Importance:

Analyzed the importance of each feature in the model.

Visualized feature importance using a bar chart.
