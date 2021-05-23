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

    println("Compute kNN on train data...")

    // my code starts here
    val num_u = train.rows.toInt
    val num_i = train.cols.toInt
    println(f"num_u = ${num_u}")

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

    val t = System.nanoTime()

    // *** calculate ru_， average rating for each user ***
    var ru_arr = new Array[Double](num_u)   // user rating array
    for (i <- ru_arr.indices) {ru_arr(i) = 0.0}
    var counts = new Array[Int](num_u) // count number of ratings per user
    for (i <- counts.indices) {counts(i) = 0}
    for ((k,rui) <- train.activeIterator) {
      val u = k._1
      val i = k._2
      ru_arr(u) += rui
      counts(u) += 1
    }
    for (i <- ru_arr.indices) {ru_arr(i) = ru_arr(i)/counts(i)}
    //val ru_ = new DenseVector(ru_arr)
    //println(s"ru_ = ${ru_}")

    //*** rhat_ui, each rating's normalized deviation ***
    val rhatui_Builder = new CSCMatrix.Builder[Double](rows=num_u, cols=num_i)
    for ((k,rui) <- train.activeIterator) {
      val u = k._1
      val i = k._2
      val rhat_ui = (rui - ru_arr(u)) / scale(rui, ru_arr(u))
      rhatui_Builder.add(u, i, rhat_ui)
    }
    val rhatui_m = rhatui_Builder.result()


    // **** rcapui, the preprocessed rating ***
    // calculate denom
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
    val rcapui_m = rcapui_Builder.result()
    //println(rcapui_m)

    // *** calculate similarities ***
    val k = 200
    val sims_Builder = new CSCMatrix.Builder[Double](rows=num_u, cols=num_u)
    for (u <- 0 until rcapui_m.rows) {
      val vec_for_u = rcapui_m(u, 0 until rcapui_m.cols).t.toDenseVector
      val simVec_for_u = rcapui_m * vec_for_u
      for (v <- argtopk(simVec_for_u, k+1)) {
        //println(u,v, simVec_for_u(v))
        if (u!=v) {
          sims_Builder.add(u, v, simVec_for_u(v))
        }
      }
    }
    val sims_m = sims_Builder.result()

    val duration_knnSims = (System.nanoTime() - t) / 1e9
    println(f"duration_knnSims = $duration_knnSims")

    // *** compute rbarhat, the user-specific weighted-sum deviation, eq.2 in project2 ***
//    val rbarhat_Builder = new CSCMatrix.Builder[Double](rows=num_u, cols=num_i)
//    val eq2num = sims_m * rhatui_m
//    val eq2denom = abs(sims_m) * rhatui_m.mapValues(rhatui => if (rhatui!=0) 1.0 else 0.0)
//    //val rbarhat_m = eq2num /:/ eq2denom
//    for ((k, numerator) <- eq2num.activeIterator) {
//      val u = k._1
//      val i = k._2
//      // only need to add rbarhat if rbarhat is not zero
//      if (eq2denom(u,i)!=0) {
//        val rbarhat = numerator/eq2denom(u,i)
//        rbarhat_Builder.add(u,i, rbarhat)
//      }
//    }
//    val rbarhat_m = rbarhat_Builder.result()

//    val rhat_ui_Builder = new CSCMatrix.Builder[Double](rows=train.rows, cols=train.cols)
//    for ((k,rui) <- train.activeIterator) {
//      val u = k._1
//      val i = k._2
//      val rhat_ui = (rui - ru_(u))/scale(rui, ru_(u))
//      rhat_ui_Builder.add(u, i, rhat_ui)
//    }
//    val rhat_ui_s = rhat_ui_Builder.result()

//    for ((k, rhat_ui) <- rhat_ui_s.activeIterator) {
//      println((k._1, k._2, rhat_ui))
//    }



    println("Loading test data from: " + conf.test())
    val testFile = Source.fromFile(conf.test())
    val testBuilder = new CSCMatrix.Builder[Double](rows=conf.users(), cols=conf.movies())
    for (line <- testFile.getLines) {
        val cols = line.split(conf.separator()).map(_.trim)
        testBuilder.add(cols(0).toInt-1, cols(1).toInt-1, cols(2).toDouble)
    }
    val test = testBuilder.result()
    testFile.close

    println("Compute predictions on test data...")
    val t2 = System.nanoTime()
    // *** compute rbarhat, the user-specific weighted-sum deviation, eq.2 in project2
    // *** then compute pui
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
    val duration_predict = (System.nanoTime() - t2)/1e9
    println(f"duration_predict = $duration_predict")

    // *** calculate MAE ***
    val mae = sum(abs(pui_m - test))/test.activeSize
    println(f"mae = ${mae}")

    // *** compute pui ***
//    val rui_builder = new CSCMatrix.Builder[Double](rows=num_u, cols=num_i)
//    for ((k, _) <- test.activeIterator) {
//      val u = k._1
//      val i = k._2
//      val pui = ru_arr(u) + rbarhat_ui * scale((ru_arr(u)+rbarhat_ui), ru_arr(u))
//
//    }


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
              "MaeForK=100" -> 0.0, // Datatype of answer: Double
              "MaeForK=200" -> 0.0  // Datatype of answer: Double
            ),
            "Q3.3.2" ->  Map(
              "DurationInMicrosecForComputingKNN" -> Map(
                "min" -> 0.0,  // Datatype of answer: Double
                "max" -> 0.0, // Datatype of answer: Double
                "average" -> 0.0, // Datatype of answer: Double
                "stddev" -> 0.0 // Datatype of answer: Double
              )
            ),
            "Q3.3.3" ->  Map(
              "DurationInMicrosecForComputingPredictions" -> Map(
                "min" -> 0.0,  // Datatype of answer: Double
                "max" -> 0.0, // Datatype of answer: Double
                "average" -> 0.0, // Datatype of answer: Double
                "stddev" -> 0.0 // Datatype of answer: Double
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
