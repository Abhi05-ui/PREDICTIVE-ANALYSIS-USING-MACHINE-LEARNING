# Import necessary libraries
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import matplotlib.pyplot as plt

# Start a Spark session with resource limits to avoid memory issues
spark = SparkSession.builder \
    .appName("AI_Image_Engagement_Prediction") \
    .config("spark.executor.memory", "2g") \
    .config("spark.cores.max", "2") \
    .getOrCreate()

# Load dataset
file_path = "/content/ai_ghibli_trend_dataset_v2.csv"  
df = spark.read.csv(file_path, header=True, inferSchema=True)

# Handle missing values 
df = df.na.drop()  # Drop rows with missing values (or use imputation)

# Select relevant features and target variable
features = ["generation_time", "gpu_usage", "file_size_kb", "style_accuracy_score"]
target = "likes"

# Convert categorical variables to numerical format 
indexer = StringIndexer(inputCol="platform", outputCol="platform_index")
df = indexer.fit(df).transform(df)

# Add encoded categorical variable to features
features.append("platform_index")

# Assemble feature vector
assembler = VectorAssembler(inputCols=features, outputCol="features")
df = assembler.transform(df).select("features", target)

# Split data into train and test sets
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# Train Random Forest Regressor
rf = RandomForestRegressor(featuresCol="features", labelCol=target, numTrees=100)
model = rf.fit(train_data)

# Make predictions
predictions = model.transform(test_data)

# Evaluate model
evaluator = RegressionEvaluator(labelCol=target, predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)

print(f"Root Mean Squared Error: {rmse}")

# Feature Importance
feature_importances = model.featureImportances.toArray()
print(f"Feature Importances: {feature_importances}")

# Plot feature importance
plt.bar(range(len(feature_importances)), feature_importances)
plt.xticks(range(len(features)), features, rotation=90)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Feature Importance")
plt.show()
