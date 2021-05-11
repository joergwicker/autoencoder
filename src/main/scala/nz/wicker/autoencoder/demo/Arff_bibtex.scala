//package nz.wicker.autoencoder.demo
//import nz.wicker.autoencoder.math.optimization.DifferentiableErrorFunctionFactory
//import nz.wicker.autoencoder.math.optimization.Minimizer
//import nz.wicker.autoencoder.neuralnet.rbm.DefaultRbmTrainingConfiguration
//import nz.wicker.autoencoder.math.optimization.SquareErrorFunctionFactory
//import nz.wicker.autoencoder.math.optimization.CG_Rasmussen2
//import nz.wicker.autoencoder.neuralnet.rbm.ConstantConfigurationTrainingStrategy
//import nz.wicker.autoencoder.neuralnet.rbm.RbmTrainingStrategy
//
//object Arff_bibtex extends ManuallyFittedExperimentMain {
//
//  val fileName = "bibtex.arff"
//  val isSparse = true
//  val compressionDimension = 10
//  override val numberOfHiddenLayers = 2
//  val layerAlpha = 0.5
//  val rbmTrainingStrategies: List[RbmTrainingStrategy] = 
//    List.fill(numberOfHiddenLayers + 1){
//    new ConstantConfigurationTrainingStrategy(
//      new DefaultRbmTrainingConfiguration(
//        epochs = 256,
//        minibatchSize = 10,
//        learningRate = 0.01,
//        initialMomentum = 0.5,
//        finalMomentum = 0.875,
//        initialGibbsSamplingSteps = 1,
//        finalGibbsSamplingSteps = 2,
//        sampleVisibleUnitsDeterministically = true,
//        weightPenaltyFactor = 0.00001 
//      ), 
//      0.33
//    )
//  }
//  val errorFunctionFactory = SquareErrorFunctionFactory
//  val maximumNumberOfFunctionEvaluations = 5000
//}
//
//// 512 epochs, 10k CG iters, learnRate = 0.02:                       2200 errors
//// 50 epochs, 5k CG iters:                                           2353 errors
//// 500 epochs, 5k CG iters:
///*
//TIME FOR TRAINING: 14 min
//L2 Error: 37.504204780550566
//Total number of errors: 1617.0
//0 -> 1 errors: 1167.0
//1 -> 0 errors: 450.0
//*/
