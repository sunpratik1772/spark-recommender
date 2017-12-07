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

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.{Param, ParamMap, ParamValidators}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable, Instrumentation, SchemaUtils}
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{FloatType, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.SizeEstimator
import scala.util.{Random, Try}

private[recommender] trait UserCFParams extends CFModelParams {

  /**
   * Param for the column name for ratings.
   * Default: "rating"
   * @group param
   */
  val ratingCol = new Param[String](this, "ratingCol", "column name for ratings")

  /** @group getParam */
  def getRatingCol: String = $(ratingCol)

  /**
   * Param for the number of similar users to the active user.
   * Default: 100
   * @group param
   */
  val numSimilarUsers = new Param[Int](this, "numSimilarUsers", "the number of similar users to the active user",
    ParamValidators.gt(0))

  /** @group getParam */
  def getNumSimilarUsers: Int = $(numSimilarUsers)

  /**
   * Param for the top K items to recommend to the active user.
   * Default: 10
   * @group param
   */
  val topKItems = new Param[Int](this, "topKItems", "top K items to recommend to the active user",
    ParamValidators.gt(0))

  /** @group getParam */
  def getTopKItems: Int = $(topKItems)

  /**
   * Param for the type of similarity measure.
   * Default: "cosineSimilarity"
   * @group param
   */
  val similarityMeasure = new Param[String](this, "similarityMeasure", "the similarity measure of user-pairs",
    isValid = ParamValidators.inArray(CFModel.supportedSimilarityMeasures))

  /** @group getParam */
  def getSimilarityMeasure: String = $(similarityMeasure)

  /**
   * Param for the interaction cut.
   * Default: 0(disable)
   * @group param
   */
  val interactionCut = new Param[Int](this, "interactionCut", "randomly down sample the interaction " +
    "histories of the `power users`.", ParamValidators.gtEq(0))

  /** @group getParam */
  def getInteractionCut: Int = $(interactionCut)

  /**
   * Param for StorageLevel for intermediate datasets. Pass in a string representation of
   * `StorageLevel`. Cannot be "NONE".
   * Default: "MEMORY_AND_DISK".
   *
   * @group expertParam
   */
  val intermediateStorageLevel = new Param[String](this, "intermediateStorageLevel",
    "StorageLevel for intermediate datasets. Cannot be 'NONE'.",
    (s: String) => Try(StorageLevel.fromString(s)).isSuccess && s != "NONE")

  /** @group expertGetParam */
  def getIntermediateStorageLevel: String = $(intermediateStorageLevel)

  /**
   * Param for StorageLevel for UserCF model. Pass in a string representation of
   * `StorageLevel`.
   * Default: "MEMORY_AND_DISK".
   *
   * @group expertParam
   */
  val finalStorageLevel = new Param[String](this, "finalStorageLevel",
    "StorageLevel for UserCF model.",
    (s: String) => Try(StorageLevel.fromString(s)).isSuccess)

  /** @group expertGetParam */
  def getFinalStorageLevel: String = $(finalStorageLevel)

  /**
   * Validates and transforms the input schema with the provided param map.
   *
   * @param schema input schema
   * @return output schema
   */
  protected def validateAndTransformSchema(schema: StructType): StructType = {
    // user and item will be cast to Int
    SchemaUtils.checkNumericType(schema, $(userCol))
    SchemaUtils.checkNumericType(schema, $(itemCol))
    // rating will be cast to Float
    SchemaUtils.checkNumericType(schema, $(ratingCol))
    SchemaUtils.appendColumn(schema, $(predictionCol), FloatType)
  }

  setDefault(userCol -> "user", itemCol -> "item", ratingCol -> "rating", predictionCol -> "prediction",
    numSimilarUsers -> 100, topKItems -> 10, similarityMeasure -> "cosineSimilarity",
    interactionCut -> 0, intermediateStorageLevel -> "MEMORY_AND_DISK",
    finalStorageLevel -> "MEMORY_AND_DISK", coldStartStrategy -> "nan")
}

/**
 * User-based Collaborative Filtering(UserCF).
 *
 * UserCF is one of most famous memory-based(or neighborhood-based) collaborative filter algorithms.
 * The procedure of UserCF can be concluded as two steps:
 * 1. Look for users who share the same rating patterns with the active user (the user whom the prediction is for).
 * 2. Use the ratings from those like-minded users found in step 1 to calculate a prediction for the active user
 *
 * Reference:
 * The implementation of this algorithm is inspired from
 * "Scalable Similarity-Based Neighborhood Methods with MapReduce", available at
 * https://dl.acm.org/citation.cfm?id=2365984
 */
class UserCF(override val uid: String) extends Estimator[UserCFModel] with UserCFParams
    with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("usercf"))

  /** @group setParam */
  def setUserCol(value: String): this.type = set(userCol, value)

  /** @group setParam */
  def setItemCol(value: String): this.type = set(itemCol, value)

  /** @group setParam */
  def setRatingCol(value: String): this.type = set(ratingCol, value)

  /** @group setParam */
  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  /** @group setParam */
  def setNumSimilarUsers(value: Int): this.type = set(numSimilarUsers, value)

  /** @group setParam */
  def setTopKItems(value: Int): this.type = set(topKItems, value)

  /** @group setParam */
  def setSimilarityMeasure(value: String): this.type = set(similarityMeasure, value)

  /** @group setParam */
  def setInteractionCut(value: Int): this.type = set(interactionCut, value)

  /** @group expertSetParam */
  def setIntermediateStorageLevel(value: String): this.type = set(intermediateStorageLevel, value)

  /** @group expertSetParam */
  def setFinalStorageLevel(value: String): this.type = set(finalStorageLevel, value)

  /** @group expertSetParam */
  def setColdStartStrategy(value: String): this.type = set(coldStartStrategy, value)

  override def fit(dataset: Dataset[_]): UserCFModel = {
    transformSchema(dataset.schema, logging = true)
    val spark = dataset.sparkSession
    import dataset.sparkSession.implicits._

    val rating = if ($(ratingCol) != "") col($(ratingCol)).cast(FloatType) else lit(1.0)
    val rawCols = dataset
      .select(checkedCast(col($(userCol))), checkedCast(col($(itemCol))), rating)
      .rdd

    require(!rawCols.isEmpty(), s"No ratings available from $rawCols")

    val instr = Instrumentation.create(this, rawCols)
    instr.logParams(userCol, itemCol, ratingCol, numSimilarUsers, topKItems, similarityMeasure,
      interactionCut, intermediateStorageLevel, finalStorageLevel)

    val handlePersistence = rawCols.getStorageLevel == StorageLevel.NONE
    if (handlePersistence) rawCols.persist(StorageLevel.fromString($(intermediateStorageLevel)))
    // materialize raw columns
    rawCols.count()

    val norm = computeNorm(rawCols, $(similarityMeasure))
    val bcNorm = spark.sparkContext.broadcast(norm.collect())

    val itemUserRating = rawCols.map(row => (row.getInt(1), (row.getInt(0), row.getFloat(2))))
      .groupByKey()
      .filter(_._2.size > 1)

    val userPairs = computeUserPairs(itemUserRating, $(interactionCut))
    val similarity = computeUser2UserSimilarity(userPairs, $(similarityMeasure), bcNorm)

    val userItemRating = rawCols.map(row => (row.getInt(0), (row.getInt(1), row.getFloat(2))))
      .groupByKey()

    val bcUserItemRating = spark.sparkContext.broadcast(userItemRating.collect())
    val recsRDD = similarity.mapPartitions { part =>
      val localNorm = bcNorm.value.toMap[Int, (Float, Float, Int)]
      val userItemRatingMap = bcUserItemRating.value.toMap
      part.map { case (k: Int, v: Iterable[(Int, Float)]) =>
        val activeUser = k
        val userSimilarity = v.toArray
        val similaritySum = userSimilarity.map(_._2).sum
        val boughtItems = userItemRatingMap(activeUser).toMap
        val avgUserRating = localNorm(activeUser)._2 / localNorm(activeUser)._3
        val similarityMap = scala.collection.mutable.Map.empty[Int, Float]
        userSimilarity.foreach { case (neighbor: Int, similarity: Float) =>
          val avgNeighborRating = localNorm(neighbor)._2 / localNorm(neighbor)._3
          userItemRatingMap(neighbor)
            .filterNot { case (item: Int, _: Float) => boughtItems.contains(item) }
            .foreach { case (item: Int, rating: Float) =>
              similarityMap.update(
                item, similarityMap.getOrElse(item, 0.toFloat) + similarity * (rating - avgNeighborRating))
            }
        }
        val scoreMap = similarityMap.map { case (item: Int, similarity: Float) =>
          (item, avgUserRating + similarity / similaritySum)
        }
        (activeUser, scoreMap.toArray.sortBy(-_._2).take($(topKItems)).toIterable)
      }.flatMap { case (k: Int, v: Iterable[(Int, Float)]) =>
        v.map { case (u: Int, r: Float) => (k, u, r) }
      }
    }

    val recsDF = recsRDD.toDF($(userCol), $(itemCol), $(predictionCol))
      .persist(StorageLevel.fromString($(finalStorageLevel)))

    if (StorageLevel.fromString($(finalStorageLevel)) != StorageLevel.NONE) {
      recsDF.count()
      rawCols.unpersist()
      bcNorm.unpersist()
      bcUserItemRating.unpersist()
    }

    val model = new UserCFModel(uid, recsDF, $(userCol), $(itemCol)).setParent(this)
    instr.logSuccess(model)
    copyValues(model)
  }

  override def copy(extra: ParamMap): UserCF = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  private def computeNorm(rawCols: RDD[Row], similarityMeasure: String):
      RDD[(Int, (Float, Float, Int))] = {

    if (CFModel.supportedSimilarityMeasures.contains(similarityMeasure)) {
      rawCols.map(row => (row.getInt(0), row.getFloat(2)))
        .aggregateByKey((0.toFloat, 0.toFloat, 0))(
          (u, v) => (u._1 + v * v, u._2 + v, u._3 + 1),
          (u1, u2) => (u1._1 + u2._1, u1._2 + u2._2, u1._3 + u2._3))
    } else {
      throw new RuntimeException("The provided similarity measure is not supported.")
    }
  }

  private def computeUserPairs(
      itemUserRating: RDD[(Int, Iterable[(Int, Float)])],
      interactionCut: Int): RDD[((Int, Int), Iterable[(Float, Float)])] = {
    /**
     * A moderately sized interactionCut is sufficient to achieve prediction quality close to
     * that of un-sampled data while it can handle scaling issues introduced by the heavy tailed
     * distribution of user interactions commonly encountered in recommendation mining scenarios.
     */
    val sampledItemUserRating = if (interactionCut == 0) {
      itemUserRating
    } else {
      itemUserRating.map { case (item: Int, userRating: Iterable[(Int, Float)]) =>
        val userRatingSeq = userRating.toSeq
        if (userRatingSeq.length > interactionCut) {
          val sampledUserRatingSeq = Random.shuffle(userRatingSeq).take(interactionCut)
          (item, sampledUserRatingSeq)
        } else {
          (item, userRating)
        }
      }
    }

    sampledItemUserRating.flatMap { case (_: Int, userRating: Iterable[(Int, Float)]) =>
      val userPairsArray = userRating.toArray
      for (p1 <- userPairsArray; p2 <- userPairsArray if p1._1 != p2._1)
        yield ((p1._1, p2._1), (p1._2, p2._2))
    }.groupByKey()
  }

  private def computeUser2UserSimilarity(
      userPairs: RDD[((Int, Int), Iterable[(Float, Float)])],
      similarityMeasure: String,
      bcNorm: Broadcast[Array[(Int, (Float, Float, Int))]]): RDD[(Int, Iterable[(Int, Float)])] = {

    userPairs.mapPartitions { part =>
      val localNorm = bcNorm.value.toMap[Int, (Float, Float, Int)]
      part.map { case ((userPair: (Int, Int), ratingPair: Iterable[(Float, Float)])) =>
        similarityMeasure match {
          case CFModel.cosineSimilarity =>
            val numerator = ratingPair.foldLeft(0.toFloat) { case (s: Float, r: (Float, Float)) =>
              s + r._1 * r._2 }
            val u1Norm = localNorm(userPair._1)._1
            val u2Norm = localNorm(userPair._2)._1
            val denominator = math.sqrt(u1Norm) * math.sqrt(u2Norm).toFloat
            val similarity = if (denominator == 0) {
              0.toFloat
            } else {
              (numerator / denominator).toFloat
            }
            (userPair._1, (userPair._2, similarity))
          case CFModel.pearsonCorrelation =>
            val avgRating1 = localNorm(userPair._1)._2 / localNorm(userPair._1)._3
            val avgRating2 = localNorm(userPair._2)._2 / localNorm(userPair._2)._3
            val (factor1, numerator, factor2) = ratingPair
              .foldLeft(0.toFloat, 0.toFloat, 0.toFloat){
                case (s: (Float, Float, Float), r: (Float, Float)) =>
                  (s._1 + (r._1 - avgRating1) * (r._1 - avgRating1),
                  s._2 + (r._1 -avgRating1) * (r._2 - avgRating2),
                  s._3 + (r._2 - avgRating2) * (r._2 - avgRating2))
              }
            val denominator = math.sqrt(factor1) * math.sqrt(factor2).toFloat
            val similarity = if (denominator == 0) {
              0.toFloat
            } else {
              (0.5 + 0.5 * (numerator / denominator)).toFloat
            }
            (userPair._1, (userPair._2, similarity))
        }
      }
    }.groupByKey().map { case (k: Int, v: Iterable[(Int, Float)]) =>
      (k, v.toArray.sortBy(-_._2).take($(numSimilarUsers)).toIterable)
    }
  }

  private def computeIsCollectable(spark: SparkSession, rdd: RDD[_]): Boolean = {

    def parseConfig(config: String): Long = {
      if (config.equals("0")) {
        0
      } else if (config.contains("g") || config.contains("G")) {
        config.substring(0, config.length - 1).toLong * 1024 * 1024
      } else if (config.contains("m") || config.contains("M")) {
        config.substring(0, config.length - 1).toLong * 1024
      } else {
        config.toLong
      }
    }

    val maxResultSize = parseConfig(spark.conf.get(UserCF.maxResultSize, "1g"))
    if (maxResultSize == 0) {
      true
    } else {
      val size = rdd.count()
      if (size < UserCF.sampleSize) {
        /**
         * For simplicity, we return true if the size of rdd is smaller than
         * sampleSize though it may exceed maxResultSize.
         */
        true
      } else {
        val fraction = UserCF.sampleSize.toDouble / size
        val sample = rdd.sample(withReplacement = false, fraction)
        val estimatedSize = SizeEstimator.estimate(sample)
        (estimatedSize / fraction) < (0.9 * maxResultSize)
      }
    }
  }
}

object UserCF {
  val maxResultSize = "spark.driver.maxResultSize"
  val sampleSize = 100
}

class UserCFModel private[ml](
    override val uid: String,
    recsDF: DataFrame,
    userColOfRecsDF: String,
    itemColOfRecsDF: String)
    extends Model[UserCFModel] with CFModelParams {

  /** @group setParam */
  def setUserCol(value: String): this.type = set(userCol, value)

  /** @group setParam */
  def setItemCol(value: String): this.type = set(itemCol, value)

  /** @group setParam */
  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  /** @group expertSetParam */
  def setColdStartStrategy(value: String): this.type = set(coldStartStrategy, value)

  override def copy(extra: ParamMap): UserCFModel = {
    val copied = new UserCFModel(uid, recsDF, userColOfRecsDF, itemColOfRecsDF)
    copyValues(copied, extra).setParent(parent)
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema)
    // create a new column named map(predictionCol) by joining on dataset and recsDF
    val predictions = dataset
      .join(recsDF,
        checkedCast(dataset($(userCol))) === recsDF(userColOfRecsDF) and
        checkedCast(dataset($(itemCol))) === recsDF(itemColOfRecsDF), "left")
      .select(dataset("*"), recsDF($(predictionCol)))

    getColdStartStrategy match {
      case CFModel.Drop =>
        predictions.na.drop("all", Seq($(predictionCol)))
      case CFModel.NaN =>
        predictions
    }
  }

  override def transformSchema(schema: StructType): StructType = {
    // user and item will be cast to Int
    SchemaUtils.checkNumericType(schema, $(userCol))
    SchemaUtils.checkNumericType(schema, $(itemCol))
    SchemaUtils.appendColumn(schema, $(predictionCol), FloatType)
  }

  def recommendForUsers(dataset: Dataset[_]): DataFrame = {
    // create a new column named map(predictionCol) by joining on dataset and recsDF
    val predictions = dataset
      .join(recsDF,
        checkedCast(dataset($(userCol))) === recsDF(userColOfRecsDF), "left")
      .select(dataset("*"), recsDF($(predictionCol)))

    getColdStartStrategy match {
      case CFModel.Drop =>
        predictions.na.drop("all", Seq($(predictionCol)))
      case CFModel.NaN =>
        predictions
    }
  }

  def recommendForItems(dataset: Dataset[_]): DataFrame = {
    // create a new column named map(predictionCol) by joining on dataset and recsDF
    val predictions = dataset
      .join(recsDF,
        checkedCast(dataset($(itemCol))) === recsDF(itemColOfRecsDF), "left")
      .select(dataset("*"), recsDF($(predictionCol)))

    getColdStartStrategy match {
      case CFModel.Drop =>
        predictions.na.drop("all", Seq($(predictionCol)))
      case CFModel.NaN =>
        predictions
    }
  }

  def recommendForAll(): DataFrame = recsDF
}
