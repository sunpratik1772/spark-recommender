/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.ml.recommender.cf

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql.SparkSession

object ItemCFSuite {

  case class Rating(user: Int, item: Int, rating: Float)
  def parseRating(str: String): Rating = {
    val fields = str.split(", ")
    assert(fields.size == 3)
    Rating(fields(0).toInt, fields(1).toInt, fields(2).toFloat)
  }

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .appName("ItemCFExample")
      .master("local[*]")
      .getOrCreate()
    import spark.implicits._

    val training = spark.read.textFile("data/training.csv")
      .map(parseRating)
      .toDF()
    val test = spark.read.textFile("data/test.csv")
      .map(parseRating)
      .toDF()
    // Build the recommendation model using ItemCF on the training data
    val itemCF = new ItemCF()
      .setSimilarityMeasure("cosineSimilarity")
    val model = itemCF.fit(training)

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
  }
}
