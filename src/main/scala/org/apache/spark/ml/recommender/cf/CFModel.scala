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

import java.util.Locale
import org.apache.spark.ml.param.shared.HasPredictionCol
import org.apache.spark.ml.param.{Param, Params}
import org.apache.spark.sql.functions.udf

/**
 * Params for CFModel.
 */
private[recommender] trait CFModelParams extends Params with HasPredictionCol {
  /**
   * Param for the column name for user ids. Ids must be integers. Other
   * numeric types are supported for this column, but will be cast to integers as long as they
   * fall within the integer value range.
   * Default: "user"
   * @group param
   */
  val userCol = new Param[String](this, "userCol", "column name for user ids. Ids must be " +
    "within the integer value range.")

  /** @group getParam */
  def getUserCol: String = $(userCol)

  /**
   * Param for the column name for item ids. Ids must be integers. Other
   * numeric types are supported for this column, but will be cast to integers as long as they
   * fall within the integer value range.
   * Default: "item"
   * @group param
   */
  val itemCol = new Param[String](this, "itemCol", "column name for item ids. Ids must be " +
    "within the integer value range.")

  /** @group getParam */
  def getItemCol: String = $(itemCol)

  /**
   * Param for strategy for dealing with unknown or new users/items at prediction time.
   * This may be useful in cross-validation or production scenarios, for handling user/item ids
   * the model has not seen in the training data.
   * Supported values:
   * - "nan":  predicted value for unknown ids will be NaN.
   * - "drop": rows in the input DataFrame containing unknown ids will be dropped from
   *           the output DataFrame containing predictions.
   * Default: "nan".
   * @group expertParam
   */
  val coldStartStrategy = new Param[String](this, "coldStartStrategy",
    "strategy for dealing with unknown or new users/items at prediction time. This may be " +
      "useful in cross-validation or production scenarios, for handling user/item ids the model " +
      "has not seen in the training data. Supported values: " +
      s"${CFModel.supportedColdStartStrategies.mkString(",")}.",
    (s: String) =>
      CFModel.supportedColdStartStrategies.contains(s.toLowerCase(Locale.ROOT)))

  /** @group expertGetParam */
  def getColdStartStrategy: String = $(coldStartStrategy).toLowerCase(Locale.ROOT)

  /**
   * Attempts to safely cast a user/item id to an Int. Throws an exception if the value is
   * out of integer range or contains a fractional part.
   */
  protected[recommender] val checkedCast = udf { (n: Any) =>
    n match {
      case v: Int => v // Avoid unnecessary casting
      case v: Number =>
        val intV = v.intValue
        // Checks if number within Int range and has no fractional part.
        if (v.doubleValue == intV) {
          intV
        } else {
          throw new IllegalArgumentException(s"UserCF/ItemCF only supports values in Integer " +
            s"range and without fractional part for columns ${$(userCol)} and ${$(itemCol)}. " +
            s"Value $n was either out of Integer range or contained a fractional part that " +
            s"could not be converted.")
        }
      case _ => throw new IllegalArgumentException(s"UserCF/ItemCF only supports values in " +
        s"Integer range for columns ${$(userCol)} and ${$(itemCol)}. Value $n was not numeric.")
    }
  }
}

object CFModel {
  private[recommender] val NaN = "nan"
  private[recommender] val Drop = "drop"
  private[recommender] final val supportedColdStartStrategies = Array(NaN, Drop)

  /** String name for "cosineSimilarity"(cosine similarity) */
  private[recommender] val cosineSimilarity = "cosineSimilarity"

  /** String name for "pearsonCorrelation"(pearson correlation coefficient) */
  private[recommender] val pearsonCorrelation = "pearsonCorrelation"

  /** Set of type of similarity measures that UserCF supports */
  private[recommender] val supportedSimilarityMeasures = Array(
    cosineSimilarity,
    pearsonCorrelation
  )
}
