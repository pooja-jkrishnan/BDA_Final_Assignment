from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Initialize Spark
spark = SparkSession.builder.appName("TaxiData_Analysis").getOrCreate()

# Load the dataset from GCS
df = spark.read.option("header", True).csv("gs://taxi-bda-bucket/Taxidataset.csv", inferSchema=True)

# Prepare output folder
output_path = "gs://taxi-bda-bucket/output/"

# 1. Summary Statistics
summary = df.describe(["fare", "trip_miles", "tips"])
summary.write.mode("overwrite").csv(output_path + "summary_stats")

# 2. Total Trips by Payment Type
payment_count = df.groupBy("payment_type").count()
payment_count.write.mode("overwrite").csv(output_path + "trips_by_payment")

# 3. Average Fare per Company
avg_fare = df.groupBy("company").avg("fare").orderBy("avg(fare)", ascending=False)
avg_fare.write.mode("overwrite").csv(output_path + "avg_fare_per_company")

# 4. High Tip Trips (tips > 20)
high_tips = df.filter(df["tips"] > 20).select("company", "fare", "tips")
high_tips.write.mode("overwrite").csv(output_path + "high_tip_trips")

# 5. Trip Miles Distribution (< 50 miles)
miles_dist = df.select("trip_miles").filter(col("trip_miles") < 50).groupBy("trip_miles").count().orderBy("trip_miles")
miles_dist.write.mode("overwrite").csv(output_path + "miles_distribution")

spark.stop()
