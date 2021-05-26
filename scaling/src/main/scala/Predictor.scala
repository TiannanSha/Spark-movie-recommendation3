import org.rogach.scallop._
import org.json4s.jackson.Serialization
import org.apache.log4j.Logger
import org.apache.log4j.Level
import breeze.linalg._
import breeze.numerics._

import scala.io.Source
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.SparkContext

class Conf(arguments: Seq[String]) extends ScallopConf(arguments) {
  val train = opt[String](required = true)
  val test = opt[String](required = true)
  val k = opt[Int]()
  val json = opt[String]()
  val users = opt[Int]()
  val movies = opt[Int]()
  val separator = opt[String]()
  verify()
}

object Predictor {
  def main(args: Array[String]) {
    var conf = new Conf(args)

    // Remove these lines if encountering/debugging Spark
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    val spark = SparkSession.builder()
      .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    val sc = spark.sparkContext

    println("Loading training data from: " + conf.train())
    val read_start = System.nanoTime
    val trainFile = Source.fromFile(conf.train())
    val trainBuilder = new CSCMatrix.Builder[Double](rows=conf.users(), cols=conf.movies()) 
    for (line <- trainFile.getLines) {
        val cols = line.split(conf.separator()).map(_.trim)
        trainBuilder.add(cols(0).toInt-1, cols(1).toInt-1, cols(2).toDouble)
    }
    val train = trainBuilder.result()
    trainFile.close
    val read_duration = System.nanoTime - read_start
    println("Read data in " + (read_duration/pow(10.0,9)) + "s")

    // conf object is not serializable, extract values that
    // will be serialized with the parallelize implementations
    val conf_users = conf.users()
    val conf_movies = conf.movies()
    val conf_k = conf.k()
    //println("Compute kNN on train data...")

    // my code starts here
    val num_u = train.rows.toInt
    val num_i = train.cols.toInt

    // helper functions
    def scale(x:Double, ru:Double): Double = {
      if (x>ru) {
        5-ru
      } else if (x<ru) {
        ru-1
      } else {
        1
      }
    }


    // *** calculate ru_， average rating for each user ***
    def avg_per_user(): Array[Double] = {
      var ru_arr = new Array[Double](num_u) // user rating array
      for (i <- ru_arr.indices) {
        ru_arr(i) = 0.0
      }
      var counts = new Array[Int](num_u) // count number of ratings per user
      for (i <- counts.indices) {
        counts(i) = 0
      }
      for ((k, rui) <- train.activeIterator) {
        val u = k._1
        val i = k._2
        ru_arr(u) += rui
        counts(u) += 1
      }
      for (i <- ru_arr.indices) {
        ru_arr(i) = ru_arr(i) / counts(i)
      }
      ru_arr
    }


    //*** rhat_ui, each rating's normalized deviation ***
    def normDev(ru_arr: Array[Double]) = {
      val rhatui_Builder = new CSCMatrix.Builder[Double](rows=num_u, cols=num_i)
      for ((k,rui) <- train.activeIterator) {
        val u = k._1
        val i = k._2
        val rhat_ui = (rui - ru_arr(u)) / scale(rui, ru_arr(u))
        rhatui_Builder.add(u, i, rhat_ui)
      }
      rhatui_Builder.result()
    }


    // **** rcapui, the preprocessed rating ***
    def preprocessRatings(rhatui_m: CSCMatrix[Double]) = {
      var sumOfSqrs = new Array[Double](num_u)
      for (i <- sumOfSqrs.indices) {sumOfSqrs(i) = 0.0}
      for ((k,rhatui) <- rhatui_m.activeIterator) {
        val u = k._1
        val i = k._2
        sumOfSqrs(u) += rhatui * rhatui
      }
      val rcapui_Builder = new CSCMatrix.Builder[Double](rows=num_u, cols=num_i)
      for ((k,rhatui) <- rhatui_m.activeIterator) {
        val u = k._1
        val i = k._2
        val rcap_ui = rhatui / math.sqrt(sumOfSqrs(u))
        rcapui_Builder.add(u, i, rcap_ui)
      }
      rcapui_Builder.result()
    }


    // *** calculate similarities ***
    def knnSims(rcapui_m: CSCMatrix[Double], k:Int, sc:SparkContext) = {
    //TODO
      val br = sc.broadcast(rcapui_m)
      val topk = (u:Int)=> {
        val rcapui_m_bc = br.value
        val vec_for_u = rcapui_m_bc(u, 0 until rcapui_m_bc.cols).t.toDenseVector
        val simVec_for_u = rcapui_m_bc * vec_for_u
        simVec_for_u(u) = 0
        (   u, argtopk(simVec_for_u, k).map( v=>(v,simVec_for_u(v)) )   )
      }
      val topks = sc.parallelize(0 until num_u).map(topk).collect()

      val knnBuilder = new CSCMatrix.Builder[Double](rows=num_u,cols=num_u)
      for (t <- topks) {
        val u = t._1
        val v_suv_seq = t._2
        for ((v,suv) <- v_suv_seq) {
          if (suv!=0) {
            knnBuilder.add(u,v,suv)  // maybe try all suvs
          }
        }
      }
      knnBuilder.result
    }

    // *** compute all sims from train ***
    def knnSims_from_train(k:Int) = {
      val ru_arr = avg_per_user()
      val rhatui_m = normDev(ru_arr)
      val rcapui_m = preprocessRatings(rhatui_m)
      val sims_m = knnSims(rcapui_m, k, sc)
      (ru_arr, rhatui_m, rcapui_m, sims_m)
    }

    
    println("Loading test data from: " + conf.test())
    val testFile = Source.fromFile(conf.test())
    val testBuilder = new CSCMatrix.Builder[Double](rows=conf.users(), cols=conf.movies()) 
    for (line <- testFile.getLines) {
        val cols = line.split(conf.separator()).map(_.trim)
        testBuilder.add(cols(0).toInt-1, cols(1).toInt-1, cols(2).toDouble)
    }
    val test = testBuilder.result()
    testFile.close

    // *** compute rbarhat, the user-specific weighted-sum deviation, eq.2 in project2
    // *** then compute pui
    def predict_for_test(ru_arr: Array[Double], rhatui_m: CSCMatrix[Double],  sims_m: CSCMatrix[Double]) = {
      val pui_m = CSCMatrix.zeros[Double](test.rows, test.cols)  // maybe change to builder
      for ((k, _) <- test.activeIterator) {
        val u = k._1
        val i = k._2
        val sims_for_u = sims_m(u, 0 until sims_m.cols).t.toDenseVector.t
        val rhats_for_i = rhatui_m(0 until rhatui_m.rows, i).toDenseVector
        val numerator = sims_for_u * rhats_for_i   // maybe to dense vec
        val denom = sims_for_u * rhats_for_i.mapValues(rhat => if (rhat!=0.0) 1.0 else 0.0)

        if (denom != 0.0) {
          val rbarhat_ui = numerator / denom
          pui_m(u,i) = ru_arr(u) + rbarhat_ui * scale((ru_arr(u) + rbarhat_ui), ru_arr(u))
        } else {
          pui_m(u,i) = ru_arr(u)
        }
      }
      pui_m
    }

    println("Compute kNN on train data...")
    val (ru_arr, rhatui_m, _, sims_m) = knnSims_from_train(k=200)
    println("Compute predictions on test data...")
    val pui_m = predict_for_test(ru_arr, rhatui_m, sims_m)
    // *** calculate MAE ***
    val mae_k200 = sum(abs(pui_m - test))/test.activeSize

    // Save answers as JSON
    def printToFile(content: String,
                    location: String = "./answers.json") =
      Some(new java.io.PrintWriter(location)).foreach{
        f => try{
          f.write(content)
        } finally{ f.close }
    }
    conf.json.toOption match {
      case None => ;
      case Some(jsonFile) => {
        var json = "";
        {
          // Limiting the scope of implicit formats with {}
          implicit val formats = org.json4s.DefaultFormats

          val answers: Map[String, Any] = Map(
            "Q4.1.1" -> Map(
              "MaeForK=200" -> mae_k200  // Datatype of answer: Double
            ),
            // Both Q4.1.2 and Q4.1.3 should provide measurement only for a single run
            "Q4.1.2" ->  Map(
              "DurationInMicrosecForComputingKNN" -> 0.0  // Datatype of answer: Double
            ),
            "Q4.1.3" ->  Map(
              "DurationInMicrosecForComputingPredictions" -> 0.0 // Datatype of answer: Double  
            )
            // Answer the other questions of 4.1.2 and 4.1.3 in your report
           )
          json = Serialization.writePretty(answers)
        }

        println(json)
        println("Saving answers in: " + jsonFile)
        printToFile(json, jsonFile)
      }
    }

    println("")
    spark.stop()
  } 
}
