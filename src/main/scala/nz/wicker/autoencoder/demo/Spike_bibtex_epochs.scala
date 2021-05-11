//package nz.wicker.autoencoder.demo
//
//import nz.wicker.autoencoder.neuralnet.rbm.DefaultRbmTrainingConfiguration
//import nz.wicker.autoencoder.math.optimization.SquareErrorFunctionFactory
//import nz.wicker.autoencoder.math.optimization.CG_Rasmussen2
//
//object Spike_bibtex_epochs extends SpikePlotMain {
//
//  val fileName = "bibtex.arff"
//  val isSparse = true
//  
//  val params = List(1, 10, 50, 100, 200, 500, 2000).map { e => 
//    Params(
//      compressionDimension = 10,
//      numLayers = 2,
//      layerAlpha = 0.5,
//      rbmConfiguration = new DefaultRbmTrainingConfiguration(
//        epochs = e,
//        minibatchSize = 10,
//        learningRate = 0.01,
//        initialMomentum = 0.5,
//        finalMomentum = 0.875,
//        initialGibbsSamplingSteps = 1,
//        finalGibbsSamplingSteps = 2,
//        sampleVisibleUnitsDeterministically = true,
//        weightPenaltyFactor = 0.00001 
//      ),
//      errorFunctionFactory = SquareErrorFunctionFactory,
//      minimizer = new CG_Rasmussen2(
//        maxIters = 4000
//      )
//    )
//  } 
//}