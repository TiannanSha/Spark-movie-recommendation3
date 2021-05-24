import org.rogach.scallop._
import org.json4s.jackson.Serialization
import breeze.linalg._
import breeze.numerics._
import scala.io.Source
import scala.collection.mutable.ArrayBuffer

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
    println("")
    println("******************************************************")

    var conf = new Conf(args)

    println("Loading training data from: " + conf.train())
    val read_start = System.nanoTime
    val trainFile = Source.fromFile(conf.train())
    val trainBuilder = new CSCMatrix.Builder[Double](rows=conf.users(), cols=conf.movies()) 
    for (line <- trainFile.getLines) {
        val cols = line.split(conf.separator()).map(_.trim)
        trainBuilder.add(cols(0).toInt-1, cols(1).toInt-1, cols(2).toDouble)
//      println(f"cols(0).toInt-1 = ${cols(0).toInt-1}")
//      println(f"cols(1).toInt-1 = ${cols(1).toInt-1}")
//      println(f"cols(2).toDouble = ${cols(2).toDouble}")

    }
    val train = trainBuilder.result()
    trainFile.close
    val read_duration = System.nanoTime - read_start
    println("Read data in " + (read_duration/pow(10.0,9)) + "s")


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
    def knnSims(rcapui_m: CSCMatrix[Double], k:Int) = {
      val sims_Builder = new CSCMatrix.Builder[Double](rows=num_u, cols=num_u)
      for (u <- 0 until rcapui_m.rows) {
        val vec_for_u = rcapui_m(u, 0 until rcapui_m.cols).t.toDenseVector
        val simVec_for_u = rcapui_m * vec_for_u
        for (v <- argtopk(simVec_for_u, k+1)) {
          if (u!=v) {
            sims_Builder.add(u, v, simVec_for_u(v))
          }
        }
      }
      sims_Builder.result()
    }

    // *** compute all sims from train ***
    def knnSims_from_train(k:Int) = {
      val ru_arr = avg_per_user()
      val rhatui_m = normDev(ru_arr)
      val rcapui_m = preprocessRatings(rhatui_m)
      val sims_m = knnSims(rcapui_m, k)
      (ru_arr, rhatui_m, rcapui_m, sims_m)
    }

    // *** load test data ***
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

    // *** Q3.2.1 ***
    println("Compute kNN on train data...")
    val (ru_arr_k100, rhatui_m_k100, _, sims_m_k100) = knnSims_from_train(k=100)
    println("Compute predictions on test data...")
    val pui_m_k100 = predict_for_test(ru_arr_k100, rhatui_m_k100, sims_m_k100)
    // *** calculate MAE ***
    val mae_k100 = sum(abs(pui_m_k100 - test))/test.activeSize


    println("Compute kNN on train data...")
    val (ru_arr, rhatui_m, _, sims_m) = knnSims_from_train(k=200)
    println("Compute predictions on test data...")
    val pui_m = predict_for_test(ru_arr, rhatui_m, sims_m)
    // *** calculate MAE ***
    val mae_k200 = sum(abs(pui_m - test))/test.activeSize



    // *** Q3.2.2, measure the time for computing all knn sims ***
    var durations_knnSims = Array[Double]()
    for (i <- 1 to 5) {
      val t = System.nanoTime()
      knnSims_from_train(k=200)
      val du = (System.nanoTime() - t) / 1e3
      durations_knnSims = durations_knnSims :+ du
    }
    // calculate duration stats
    val knnTimeMin = durations_knnSims.min
    val knnTimeMax = durations_knnSims.max
    val knnTimeMean = durations_knnSims.sum/durations_knnSims.length
    val knnTimeStd = math.sqrt(
      durations_knnSims.map( t=>(t-knnTimeMean)*(t-knnTimeMean) ).sum / durations_knnSims.size
    )

    // *** Q3.2.3, measure the time for predictions ***
    var durations_predict = Array[Double]()
    for (i <- 1 to 5) {
      val t = System.nanoTime()
      predict_for_test(ru_arr, rhatui_m, sims_m)
      val du = (System.nanoTime() - t) / 1e3
      durations_predict = durations_predict :+ du
    }
    // calculate duration stats
    val predTimeMin = durations_predict.min
    val predTimeMax = durations_predict.max
    val predTimeMean = durations_predict.sum/durations_predict.length
    val predTimeStd = math.sqrt(
      durations_predict.map( t=>(t-predTimeMean)*(t-predTimeMean) ).sum / durations_predict.size
    )



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
            "Q3.3.1" -> Map(
              "MaeForK=100" -> mae_k100, // Datatype of answer: Double
              "MaeForK=200" -> mae_k200 // Datatype of answer: Double
            ),
            "Q3.3.2" ->  Map(
              "DurationInMicrosecForComputingKNN" -> Map(
                "min" -> knnTimeMin,  // Datatype of answer: Double
                "max" -> knnTimeMax, // Datatype of answer: Double
                "average" -> knnTimeMean, // Datatype of answer: Double
                "stddev" -> knnTimeStd // Datatype of answer: Double
              )
            ),
            "Q3.3.3" ->  Map(
              "DurationInMicrosecForComputingPredictions" -> Map(
                "min" -> predTimeMin,  // Datatype of answer: Double
                "max" -> predTimeMax, // Datatype of answer: Double
                "average" -> predTimeMean, // Datatype of answer: Double
                "stddev" -> predTimeStd // Datatype of answer: Double
              )
            )
            // Answer the Question 3.3.4 exclusively on the report.
           )
          json = Serialization.writePretty(answers)
        }

        println(json)
        println("Saving answers in: " + jsonFile)
        printToFile(json, jsonFile)
      }
    }

    println("")
  } 
}
