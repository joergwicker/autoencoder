//package nz.wicker.autoencoder.demo
//
//import nz.wicker.autoencoder.math.optimization.DifferentiableErrorFunctionFactory
//import nz.wicker.autoencoder.math.optimization.Minimizer
//import nz.wicker.autoencoder.neuralnet.rbm.DefaultRbmTrainingConfiguration
//import nz.wicker.autoencoder.neuralnet.rbm.ConstantConfigurationTrainingStrategy
//import nz.wicker.autoencoder.math.optimization.SquareErrorFunctionFactory
//import nz.wicker.autoencoder.math.optimization.CG_Rasmussen2
//import nz.wicker.autoencoder.math.optimization.CrossEntropyErrorFunctionFactory
//import nz.wicker.autoencoder.neuralnet.rbm.RbmTrainingStrategy
//
//object Arff_yeast extends ManuallyFittedExperimentMain {
//  
//  val fileName = "yeast.arff"
//  val isSparse = false
//  val compressionDimension = 5
//  val numLayers = 2
//  val layerAlpha = 0.8
//  val rbmTrainingStrategies: List[RbmTrainingStrategy] = List.fill(numLayers){
//    new ConstantConfigurationTrainingStrategy(
//      new DefaultRbmTrainingConfiguration(
//        epochs = 100,
//        minibatchSize = 10,
//        learningRate = 0.02,
//        initialMomentum = 0.5,
//        finalMomentum = 0.875,
//        initialGibbsSamplingSteps = 1,
//        finalGibbsSamplingSteps = 2,
//        sampleVisibleUnitsDeterministically = true,
//        weightPenaltyFactor = 0.00001 // 
//      ),
//      0.33
//    )
//  }
//  val errorFunctionFactory = CrossEntropyErrorFunctionFactory
//  val maximumNumberOfFunctionEvaluations = 100000
//}
//
///* !!! THIS IS BULLSHIT !!!
// * The algorithm used here was flawed, just look how terrible it was:
//A run with only 100 epochs
//Fitness: -291.4869750959224
//...
//Fitness: -20.627378225795006
//Early stopping: terminating
//optimization finished!
//TIME FOR TRAINING: 5 min
//L2 Error: 42.39682183459549
//Total number of errors: 2228.0
//0 -> 1 errors: 1315.0
//1 -> 0 errors: 913.0
//*/ 
//
///* That's what it looks like after the algorithms has been fixed:
//
//dimensions: 2417 x 14
//number of ones: 10241.0
//optimization finished!
//TIME FOR TRAINING: 3 min
//L2 Error: 12.204199277534242
//Total number of errors: 133.0
//0 -> 1 errors: 52.0
//1 -> 0 errors: 81.0
//*/