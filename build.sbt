name := "spark-recommender"

version := "0.1"

scalaVersion := "2.11.8"

spName := "hibayesian/spark-recommender"

sparkVersion := "2.2.0"

sparkComponents += "mllib"

resolvers += Resolver.sonatypeRepo("public")

spShortDescription := "spark-recommender"

spDescription := """A library of parallel algorithms for recommender systems based on Spark"""
  .stripMargin

credentials += Credentials(Path.userHome / ".ivy2" / ".sbtcredentials")

licenses += "Apache-2.0" -> url("http://opensource.org/licenses/Apache-2.0")