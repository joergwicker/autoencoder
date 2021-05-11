//package nz.wicker.autoencoder.demo
//
//import nz.wicker.autoencoder.math.optimization.DifferentiableErrorFunctionFactory
//import nz.wicker.autoencoder.math.optimization.Minimizer
//import nz.wicker.autoencoder.neuralnet.rbm.DefaultRbmTrainingConfiguration
//import nz.wicker.autoencoder.math.optimization.SquareErrorFunctionFactory
//import nz.wicker.autoencoder.math.optimization.CG_Rasmussen2
//import nz.wicker.autoencoder.neuralnet.rbm.ConstantConfigurationTrainingStrategy
//import nz.wicker.autoencoder.neuralnet.rbm.RbmTrainingStrategy
//
//object Arff_mediamill extends ManuallyFittedExperimentMain {
//
//  val fileName = "mediamill.arff"
//  val isSparse = false
//  val compressionDimension = 8
//  val numLayers = 3
//  val layerAlpha = 0.7
//  val rbmTrainingStrategies: List[RbmTrainingStrategy] = List.fill(numLayers){
//    new ConstantConfigurationTrainingStrategy(
//      new DefaultRbmTrainingConfiguration(
//        epochs = 32,// good value: 32,
//        minibatchSize = 10,
//        learningRate = 0.02,
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
//  
//}