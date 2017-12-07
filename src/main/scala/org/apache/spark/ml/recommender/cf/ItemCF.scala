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

import org.apache.spark.ml.linalg.BLAS
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.ml.param.{Param, ParamMap, ParamValidators}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable, Instrumentation, SchemaUtils}
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.functions.{col, lit, udf}
import org.apache.spark.sql.types.{FloatType, StructType}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.storage.StorageLevel
import scala.util.{Random, Try}

private[recommender] trait ItemCFParams extends CFModelParams {
  
  /**
   * Param for the column name for ratings.
   * Default: "rating"
   * @group param
   */
  val ratingCol = new Param[String](this, "ratingCol", "column name for ratings")

  /** @group getParam */
  def getRatingCol: String = $(ratingCol)

  /**
   * Param for the number of similar items to the active item.
   * Default: 10
   * @group param
   */
  val numSimilarItems = new Param[Int](this, "numSimilarItems",
    "the number of similar items to the active item", ParamValidators.gt(0))

  /** @group getParam */
  def getNumSimilarItems: Int = $(numSimilarItems)

  /**
   * Param for the type of similarity measure.
   * Default: "cosineSimilarity"
   * @group param
   */
  val similarityMeasure = new Param[String](this, "similarityMeasure", "the similarity measure " +
    "of user-pairs", isValid = ParamValidators.inArray(CFModel.supportedSimilarityMeasures))

  /** @group getParam */
  def getSimilarityMeasure: String = $(similarityMeasure)

  /**
   * Param for the interaction cut.
   * Default: 0(disable)
   * @group param
   */
  val interactionCut = new Param[Int](this, "interactionCut", "randomly down sample the " +
    "interaction histories of the `power users`.", ParamValidators.gtEq(0))

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
    "StorageLevel for UserCF model.", (s: String) => Try(StorageLevel.fromString(s)).isSuccess)

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

  setDefault(userCol -> "user", itemCol -> "item", ratingCol -> "rating",
    numSimilarItems -> 10, similarityMeasure -> "cosineSimilarity",
    interactionCut -> 0, intermediateStorageLevel -> "MEMORY_AND_DISK",
    finalStorageLevel -> "MEMORY_AND_DISK", coldStartStrategy -> "nan")
}

/**
 * Item-based Collaborative Filtering(ItemCF).
 *
 * ItemCF is one of most famous memory-based(or neighborhood-based) collaborative filter algorithms.
 * The procedure of ItemCF can be concluded as two steps:
 * 1. Build an item-item matrix determining relationships between pairs of items
 * 2. Infer the tastes of the current user by examining the matrix and matching that user's data
 *
 * Reference:
 * The implementation of this algorithm is inspired from
 * "Scalable Similarity-Based Neighborhood Methods with MapReduce", available at
 * https://dl.acm.org/citation.cfm?id=2365984
 */
class ItemCF(override val uid: String) extends Estimator[ItemCFModel] with ItemCFParams
    with DefaultParamsWritable{

  def this() = this(Identifiable.randomUID("itemcf"))

  /** @group setParam */
  def setUserCol(value: String): this.type = set(userCol, value)

  /** @group setParam */
  def setItemCol(value: String): this.type = set(itemCol, value)

  /** @group setParam */
  def setRatingCol(value: String): this.type = set(ratingCol, value)

  /** @group setParam */
  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  /** @group setParam */
  def setNumSimilarItems(value: Int): this.type = set(numSimilarItems, value)

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

  override def fit(dataset: Dataset[_]): ItemCFModel = {
    transformSchema(dataset.schema, logging = true)
    val spark = dataset.sparkSession
    import dataset.sparkSession.implicits._

    val rating = if ($(ratingCol) != "") col($(ratingCol)).cast(FloatType) else lit(1.0)
    val rawCols = dataset
      .select(checkedCast(col($(userCol))), checkedCast(col($(itemCol))), rating)
      .rdd

    require(!rawCols.isEmpty(), s"No ratings available from $rawCols")

    val instr = Instrumentation.create(this, rawCols)
    instr.logParams(userCol, itemCol, ratingCol, numSimilarItems, similarityMeasure,
      interactionCut, intermediateStorageLevel, finalStorageLevel)

    val userItemRating = rawCols.map(row => (row.getInt(0), (row.getInt(1), row.getFloat(2))))
      .groupByKey()

    /**
     * A moderately sized interactionCut is sufficient to achieve prediction quality close to
     * that of un-sampled data while it can handle scaling issues introduced by the heavy tailed
     * distribution of user interactions commonly encountered in recommendation mining scenarios.
     */
    val sampledUserItemRating = if ($(interactionCut) == 0) {
      userItemRating
    } else {
      userItemRating.mapPartitions { part =>
        part.map { case (user: Int, itemRating: Iterable[(Int, Float)]) =>
          val itemRatingSeq = itemRating.toSeq
          if (itemRatingSeq.length > $(interactionCut)) {
            val sampledItemRatingSeq = Random.shuffle(itemRatingSeq).take($(interactionCut))
            (user, sampledItemRatingSeq)
          } else {
            (user, itemRating)
          }
        }
      }
    }

    val handlePersistence = sampledUserItemRating.getStorageLevel == StorageLevel.NONE
    if (handlePersistence) {
      sampledUserItemRating.persist(StorageLevel.fromString($(intermediateStorageLevel)))
    }
    // Materialize raw columns.
    sampledUserItemRating.count()

    /**
     * High efficiency of computing similarity between item pairs is guaranteed by
     * using a broadcast variable here.
     * The maximum number of items that held by a broadcast variable can be
     * estimated by using the following equation:
     *   maxNumItems = maxResultSize * 1000 * 1000 * 1000 / 112,
     * where 112 (bytes) is the fixed size of each record in the below norm rdd.
     * For instance, given a default maxResultSize which is 1G bytes,
     * the maximum number of items is approximately 9 million.
     * As the equation says, the maximum number of items supported by this implementation
     * can be enlarged by specifying the size of "spark.driver.maxResultSize" in Spark conf.
     */
    val norm = sampledUserItemRating.mapPartitions(_.flatMap(_._2))
      .aggregateByKey((0.toFloat, 0.toFloat, 0))(
        (u, v) => (u._1 + v * v, u._2 + v, u._3 + 1),
        (u1, u2) => (u1._1 + u2._1, u1._2 + u2._2, u1._3 + u2._3))
      .collect()
    val numItems = norm.length
    val bcNorm = spark.sparkContext.broadcast(norm)

    val itemPairs = sampledUserItemRating.filter(_._2.size > 1)
      .flatMap { case (_: Int, itemRating: Iterable[(Int, Float)]) =>
        val itemPairsArray = itemRating.toArray
        for (p1 <- itemPairsArray; p2 <- itemPairsArray if p1._1 != p2._1)
          yield ((p1._1, p2._1), (p1._2, p2._2))
      }.groupByKey()

    val similarityDF = itemPairs.mapPartitions { part =>
      val localNorm = bcNorm.value.toMap[Int, (Float, Float, Int)]
      part.map { case ((itemPair: (Int, Int), ratingPair: Iterable[(Float, Float)])) =>
        val numerator = ratingPair.foldLeft(0.toFloat) { case (s: Float, r: (Float, Float)) =>
          s + r._1 * r._2 }
        val u1Norm = localNorm(itemPair._1)._1
        val u2Norm = localNorm(itemPair._2)._1
        val denominator = math.sqrt(u1Norm) * math.sqrt(u2Norm).toFloat
        val similarity = if (denominator == 0) {
          0.toFloat
        } else {
          (numerator / denominator).toFloat
        }
        (itemPair._1, (itemPair._2, similarity))
      }
    }.groupByKey().map { case (k: Int, v: Iterable[(Int, Float)]) =>
      val (indices, values) = v.toArray.sortBy(-_._2).take($(numSimilarItems)).sortBy(_._1).unzip
      (k, new SparseVector(numItems, indices, values.map(_.toDouble)))
    }.toDF("item", "item_sim").persist(StorageLevel.fromString($(finalStorageLevel)))

    val userItemRatingDF = sampledUserItemRating.map { case (k: Int, v: Iterable[(Int, Float)]) =>
      val (indices, values) = v.toArray.sortBy(_._1).unzip
      (k, new SparseVector(numItems, indices, values.map(_.toDouble)))
    }.toDF("user", "item_rating").persist(StorageLevel.fromString($(finalStorageLevel)))

    if (StorageLevel.fromString($(finalStorageLevel)) != StorageLevel.NONE) {
      similarityDF.count()
      userItemRatingDF.count()
      rawCols.unpersist()
      bcNorm.unpersist()
      sampledUserItemRating.unpersist()
    }

    val model = new ItemCFModel(uid, numItems, userItemRatingDF, similarityDF).setParent(this)
    instr.logSuccess(model)
    copyValues(model)
  }

  override def copy(extra: ParamMap): ItemCF = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }
}

class ItemCFModel private[ml](
    override val uid: String,
    numItems: Int,
    userItemRating: DataFrame,
    similarity: DataFrame)
    extends Model[ItemCFModel] with CFModelParams {

  /** @group setParam */
  def setUserCol(value: String): this.type = set(userCol, value)

  /** @group setParam */
  def setItemCol(value: String): this.type = set(itemCol, value)

  /** @group setParam */
  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  /** @group expertSetParam */
  def setColdStartStrategy(value: String): this.type = set(coldStartStrategy, value)

  override def copy(extra: ParamMap): ItemCFModel = {
    val copied = new ItemCFModel(uid, numItems, userItemRating, similarity)
    copyValues(copied, extra).setParent(parent)
  }

  private val predict = udf { (first: SparseVector, second: SparseVector) =>
    if (first != null && second != null) {
      val numerator = BLAS.dot(first, second)
      val denominator = BLAS.dot(
        new SparseVector(numItems, first.indices, first.indices.map(_ => 1.0)), second)
      numerator / denominator
    } else {
      Float.NaN
    }
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    val predictions = dataset
      .join(userItemRating, dataset($(userCol)) === userItemRating("user"))
      .join(similarity, dataset($(itemCol)) === similarity("item"))
      .select(dataset("*"),
        predict(userItemRating("item_rating"), similarity("item_sim")).as($(predictionCol)))

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
}
