# Spark-Recommender
Spark-Recommender is a library of parallel algorithms for recommender systems based on Spark. Now it includes following algorithms:
+ UserCF: one of most famous memory-based(or neighborhood-based) collaborative filter algorithms. The implementation of this algorithm is inspired from [Scalable Similarity-Based Neighborhood Methods with MapReduce](https://dl.acm.org/citation.cfm?id=2365984). Invert indexing and down-sampling tricks have been used to optimize its efficiency and scalability.

# Examples
## Scala API
```scala
val spark = SparkSession
  .builder()
  .appName("UserCFExample")
  .master("local[*]")
  .getOrCreate()
import spark.implicits._

val training = spark.read.textFile("data/training.csv")
  .map(parseRating)
  .toDF()
val test = spark.read.textFile("data/test.csv")
  .map(parseRating)
  .toDF()
// Build the recommendation model using UserCF on the training data
val userCF = new UserCF()
  .setSimilarityMeasure("cosineSimilarity")
val model = userCF.fit(training)

// Evaluate the model by computing the RMSE on the test data
// Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
model.setColdStartStrategy("drop")
val predictions = model.transform(test)

val evaluator = new RegressionEvaluator()
  .setMetricName("rmse")
  .setLabelCol("rating")
  .setPredictionCol("prediction")
val rmse = evaluator.evaluate(predictions)
println(s"Root-mean-square error = $rmse")
spark.stop()
```

# Requirements
Spark-Recommender is built against Spark 2.2.0.

# Build From Source
```scala
sbt package
```

# Licenses
Spark-Recommender is available under Apache Licenses 2.0.

# Contact & Feedback
If you encounter bugs, feel free to submit an issue or pull request. Also you can mail to:
+ hibayesian (hibayesian@gmail.com).
