from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col

# Start Spark session
spark = SparkSession.builder.appName("TaxiFarePrediction").getOrCreate()

# Load CSV from GCS
df = spark.read.option("header", True).csv("gs://taxi-bda-bucket/taxi_data/Taxidataset.csv", inferSchema=True)

# Drop nulls and filter invalid data
df = df.dropna(subset=["fare", "trip_miles", "tips"])
df = df.filter((col("fare") > 0) & (col("fare") < 200) & (col("trip_miles") > 0))

# Assemble features
features = ["trip_miles", "tips"]
assembler = VectorAssembler(inputCols=features, outputCol="features")
df_features = assembler.transform(df).select("features", df["fare"].alias("label"))

# Train/test split
train_data, test_data = df_features.randomSplit([0.7, 0.3], seed=42)

# Linear Regression model
lr = LinearRegression()
model = lr.fit(train_data)

# Predict
predictions = model.transform(test_data)

# Evaluate
evaluator = RegressionEvaluator(metricName="rmse")
rmse = evaluator.evaluate(predictions)
r2 = RegressionEvaluator(metricName="r2").evaluate(predictions)

print("=== Model Performance ===")
print(f"RMSE: {rmse:.2f}")
print(f"R2: {r2:.2f}")

# Save predictions to GCS
predictions.select("label", "prediction")\
    .write.mode("overwrite").csv("gs://taxi-bda-bucket/ML_output/fare_predictions")

spark.stop()
