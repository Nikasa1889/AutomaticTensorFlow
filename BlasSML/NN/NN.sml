structure NN :> NN =
struct
type matrix = Matrix.matrix
type vector = Matrix.vector
exception InputError
exception NotSupported
(* nnlayer: neural network layer type
            a neural network is a list of nnlayer variables. 
            each layer contains neccesary information for training and predicting process
*)
(* costType: supported cost function type 
    - Negative log likelyhood: only use when you're using oneHot labels.
    - Mean square error
	- Cross entropy
    - Error percentage - for benchmark only
*)
datatype costType = NLL|MSE|CE|PER 
(* initType: supported initialization methods 
    - Sparse
    - Normal
    - Normalized
*)
datatype initType = SPARSE|NORMAL|NORMALISED
(* actType: supported activation function 
    - Sigmoid
    - Tanh
*)
datatype actType = SIGM|TANH|LINEAR
(* wdType: supported weight decay type
    - L0: no weight decay
    - L1: L1 weight decay
    - L2: L2 weight decay
*)
datatype wdType = L0|L1|L2
type nnlayer = {input: matrix, (* ROWMATRIX *)
		output: matrix,(* ROWMATRIX *)
		GradW: matrix, (* COLMATRIX *)
		GradB: vector, 
		B: vector,     
		W: matrix,      (* COLMATRIX *)
		actType:actType
	       }
type params = {batchsize: int,   (* number of training cases per batch *)
	      nBatches: int,    (* number of batches *)
	      testsize: int,    (* number of testing cases *)
		  nTestBatches: int, (* because of memory issue, we should split big test into smaller batches *)
	      lambda: real,   (* momentum coefficient *)
	      momentumSchedule: bool,
	      maxLambda: real,
	      lr: real,        (* learning rate *)
	      costType: costType,   (* cost function type *)
	      initType: initType,(* initialization type *)
	      actType: actType,   (* activation function type *)
	      layerSizes: int list, (* structure of network *)
	      initWs: matrix option list, (* preinitialized Ws matrices *)
	      initBs: vector option list, (* preinitialized Bs matrices *)
	      nItrs: int,         (* number of iterations/epoches *)
	      wdType: wdType,
	      wdValue: real,
	      verbose: bool		  
	      }
val params:params ref = 
    ref{batchsize = 10,   (* number of training cases per batch *)
	nBatches = 12,    (* number of batches *)
	testsize = 30,    (* number of testing cases *)
	nTestBatches = 1, (* because of memory issue, we should split big test into smaller batches *)
	lambda = 0.0,   (* momentum coefficient *)
	momentumSchedule = false,
	maxLambda = 0.0,
	lr = 0.005,        (* learning rate *)
	costType = MSE,   (* cost function type *)
	initType = NORMAL,(* initialization type *)
	actType = SIGM,   (* activation function type *)
	layerSizes = [4, 8, 3], (* structure of network *)
	nItrs = 40,         (* number of iterations/epoches *)
	initWs = [],
	initBs = [],
	wdType = L0,
	wdValue = 0.0,
	verbose = false
       };
type fileNames = { data_train: string,
		   labels_train: string,
		   data_test: string,
		   labels_test: string
		 }	
(*  all input file names *)
val fileNames:fileNames = 
    { data_train   = "data_train.csv",
      labels_train = "labels_train.csv",
      data_test    = "data_test.csv",
      labels_test  = "labels_test.csv"
    }

val isDebug:bool ref = ref (false);
fun debugOn () = isDebug:=true;
fun debugOff () = isDebug:=false;
(* -------------------- Functions for reading inputs --------------*)
(* parseLine: parse line of string
   input
    - line: string of numbers seperated by commas 
   output
    - list of real numbers
*)
fun setParams(p: params) = 
    params:=p
fun parseLine(line) =
let 
    fun getFirstNum nil = nil
    |   getFirstNum (x::xs) = if x = #"," then [] else x::getFirstNum(xs);
in
    case line of
	[] => []
      | #","::line' => parseLine(line')
      | #" "::line' => parseLine(line')
      | _ => 
	let
	    val numStr = implode(getFirstNum(line))
	    val num = valOf(Real.fromString(numStr))
	in num::parseLine(List.drop(line, size(numStr)))
	end
end
(* readData: read input data
   input
     - nAtts: Each line has nAtts numbers seperated by commas
     - batchsize: number of lines to create a matrix - a batch of training cases
     - nBatches: total number of batches
   output
     - list of batches (each batch is a ROWMATRIX)
 *)
fun readData (fileName, nAtts, batchsize, nBatches) =
    let
	fun readLines (fh, nlines) = 
	    case (TextIO.endOfStream fh, nlines = 0) of
		( _   , true)  => []
	      | (false, false) => 
		(parseLine(explode(valOf(TextIO.inputLine fh))))
		::readLines (fh, nlines-1)
	      | (true,  false)  => raise InputError
	fun readBatches (fh, nBatches) =
	    if nBatches > 0 
	    then Matrix.fromLists(readLines(fh, batchsize), 
					 (batchsize, nAtts))
		 ::readBatches(fh, nBatches-1)
	    else (TextIO.closeIn fh; [])
    in
	readBatches(TextIO.openIn fileName, nBatches)
    end
(* ------------------- Functions for training process --------------*)
	
fun initSparseW RandState  (nInputs, nOutputs) =
    let
	val nconn = 4;
	fun initaNode (nInputs) =
	    let
		fun initSparseList (ids, i, n) =
		    case (ids, i <= n) of
			(_, false) => []
		       |( [], true)  => 0.0::initSparseList (ids, i+1, n)
		       |( idx::ids', true) => 
			if (idx = i) then
			    Randomext.rand_normal RandState (0.0, 1.0)
			    ::initSparseList (ids', i+1, n)
			else
			    0.0::initSparseList (ids, i+1, n)
	    in
		initSparseList (Randomext.rand_perm  RandState (nconn, nInputs)
			       , 1, nInputs)
	    end
	fun initSparseVects (nInputs, nOutputs) =
	    if (nOutputs > 0) then
		initaNode (nInputs)::initSparseVects (nInputs, nOutputs-1)
	    else []
    in
	Matrix.fromLists(initSparseVects (nInputs, nOutputs), 
				(nInputs, nOutputs))
    end
	
(* initLayer: initialize a neurral network layer
   input:
     - nInputs: number of incomming nodes
     - nOutputs: number of outgoing nodes
   output:
     - a initialized layer
*)
fun initLayer RandState  (nInputs, nOutputs, actType, initW, initB):nnlayer = 
    let
	val input = Matrix.newMat (1, 1)
	val _ = Matrix.freeMat (input)
	val output = Matrix.newMat (1, 1)
	val GradW = Matrix.newMat (nInputs, nOutputs)
	val _ = Matrix.mdset (GradW, 0.0)
	val GradB = Matrix.newVec (nOutputs)
	val _ = Matrix.dset (GradB, 0.0)
	val B = case initB of
		    NONE => Matrix.newVec(nOutputs)
		  | SOME v => v
	val _ = case initB of 
		NONE => Matrix.dset(B, 0.0)
		|SOME v => ()
	val W = case initW of
		    NONE => (case #initType(!params) of
				 SPARSE => initSparseW RandState  (nInputs, nOutputs)
			       (*| NORMAL => 
				 Matrix.uniRandCols RandState (
				     1.0/Math.sqrt(Real.fromInt(nInputs)),
				     (nInputs, nOutputs)) *)
			       |_ => raise NotSupported)
		  | SOME m => m
    in
	{input = input, output = output, GradW = GradW, 
	 GradB = GradB, B = B, W = W, actType = actType}
    end

(* initLayers: init the whole neural network
   input
     - layerSizes: structure of neural network
   output
     - a list of initialized layer (e.g. a initlaized network)
   NOTE:
     - if cost function is NLL or CE, 
       last layer should not use any activation function 
*)
fun initLayers RandState  (layerSizes, initWs, initBs) =
    let
	val (initW, initWs') = case initWs of
				   [] => (NONE, [])
				 | W::initWs' => (W, initWs')
	val (initB, initBs') = case initBs of
				   [] => (NONE, [])
				 | B::initBs' => (B, initBs')
    in
	case layerSizes of
	    [] => []
	  | last::[] => []
	  | nInputs::nOutputs::[] => 
	    if (#costType(!params) = NLL) orelse (#costType(!params) = CE) then
		initLayer RandState (nInputs, nOutputs, LINEAR, initW, initB) :: []
	    else
		initLayer RandState (nInputs, nOutputs, #actType(!params), initW, initB) 
		:: []
	  | nInputs::nOutputs::layerSizes' => 
	    initLayer RandState (nInputs, nOutputs, #actType(!params), initW, initB)
	    ::initLayers RandState (nOutputs::layerSizes', initWs', initBs')
    end

fun freeLayer (layer:nnlayer) =
let
	val _ = Matrix.freeMat (#output(layer))
	val _ = Matrix.freeVec (#GradB(layer))
	val _ = Matrix.freeMat (#GradW(layer))
	val _ = Matrix.freeMat (#W(layer))
	val _ = Matrix.freeVec (#B(layer))
in
	()
end
fun freeNN (layers) = List.foldl (fn (layer, acc) => freeLayer(layer)) () layers

fun sigm x = 
	if x > 13.0 then 1.0 
	else if x < ~13.0 then 0.0 
	else 1.0/(1.0+Math.exp(~x))
(* fprop1layer: forward propagate 1 layer
   input:
    - layer: layer to be propagated
    - input: input data to be propagated
   output:
    - (propagated layer, propagated input)
*)
fun fprop1layer (layer:nnlayer, input) =
    let
	val nSamples = #1(Matrix.size(input))
	val _ = if (!isDebug ) then print ((Int.toString (nSamples)) ^ "fprop1") else ()
	val _ = Matrix.freeMat (#output(layer))
	val output = Matrix.dreprows (#B(layer), nSamples)
	val _ = Matrix.dgemm(Matrix.NOTRANS, Matrix.NOTRANS, 1.0, input, #W(layer), 1.0, output)
	val _ =
	    case #actType(layer) of
		SIGM => Matrix.dappSigm (Matrix.toVector(output))
		| TANH => Matrix.dappTanh (Matrix.toVector(output))
		| LINEAR => ()	
	
    in
	({input = input, output = output, GradW = #GradW(layer), 
	  GradB = #GradB(layer), B = #B(layer), W = #W(layer), 
	  actType = #actType(layer)}, output)
    end
(* fprop: fpropagate the whole network
   input:
     - layers: neural network to be forward propagated
     - input:  input to the first layer (training data)
   output:
     - (propagated network, final propagated input)
   ** Note that returing network has the inversed order of layers
      This will be inversed again when using bprop function
*)
fun fprop (layers, input) = 
    List.foldl(fn(layer, (layers', input)) =>
		  case fprop1layer(layer, input) of
		      (fpropedLayer, output) => (fpropedLayer::layers', output))
	      ([], input) layers
(* fprop put the layers into backward order, we need to reorder the layers when we don't use bprop*)
fun reorder (layers) = 
	let
		val (newLayers, _) = List.foldl ( fn(layer, (layers', input)) =>
						     (layer::layers', input)) ([], []) layers
	in
		newLayers
	end

(* bprop1layer: backward propagate 1 layer
   input:
    - layer: layer to be propagated
    - gradInput: input gradient to be propagated
   output:
    - (propagated layer, propagated gradient)
*)
fun bprop1layer (layer:nnlayer, gradInput) =
    let
	val lambda = #lambda(!params)
	val wdValue = #wdValue(!params)
	(*fun dervSigm x = x * (1.0 - x)
	fun dervTanh x = 1.0-x*x*)
	fun sign x = if x > 0.0 then 1.0 else ~1.0
	val _ = if (!isDebug) then print "bprop1" else ()
	val _ = case #actType(layer) of
			   SIGM => Matrix.dappDSigm (Matrix.toVector(#output(layer)))
			 | TANH => Matrix.dappDTanh (Matrix.toVector(#output(layer)))
			 | LINEAR =>  Matrix.mdset(#output(layer), 1.0)
	val gradOutput = Matrix.mdotmul(gradInput, #output(layer))
	val oldGradW = Matrix.mdcopy(#GradW(layer))
	val oldGradB = Matrix.dcopy(#GradB(layer))
	
	val _ = Matrix.dgemm(Matrix.TRANS, Matrix.NOTRANS, 1.0, #input(layer), gradOutput, 0.0, #GradW(layer))
	val _ = case #wdType(!params) of
			L0 => ()
			| L2 => Matrix.mdaxpy (wdValue, #W(layer), #GradW(layer))
		    | L1 => raise NotSupported
		    
	val _ = Matrix.sumRows(gradOutput, #GradB(layer))
	
	val _ = Matrix.mdscal ((1.0 - lambda), #GradW(layer))
	val _ = Matrix.dscal ((1.0 - lambda), #GradB(layer))
	val _ = Matrix.mdaxpy (lambda, oldGradW, #GradW(layer))
	val _ = Matrix.daxpy (lambda, oldGradB, #GradB(layer))
	
	val propedGrad = Matrix.newMat (Matrix.size(#input(layer)))
	(*val _ = print ("gradOutput")
	val _ = print ("("^Int.toString(#1(Matrix.size(gradOutput))) ^ "," ^ Int.toString(#2(Matrix.size(gradOutput))) ^ ")")
	val _ = print ("("^Int.toString(#1(Matrix.size(#W(layer)))) ^ "," ^ Int.toString(#2(Matrix.size(#W(layer)))) ^ ")") *)
	val _  = Matrix.dgemm(Matrix.NOTRANS, Matrix.TRANS, 1.0, gradOutput, #W(layer), 0.0, propedGrad)
	val _ = Matrix.freeVec (oldGradB)
	val _ = Matrix.freeMat (oldGradW)
	val _ = Matrix.freeMat (gradOutput)
    in
	({input = #input(layer), output = #output(layer), GradW = #GradW(layer), 
	  GradB = #GradB(layer), B = #B(layer), W = #W(layer), 
	  actType = #actType(layer)}, propedGrad)
    end

(* bprop: back propagate the whole network
   input:
     - layers: neural network with inversed order of layers created by fprop
     - gradInput:  gradient to the last layer
   output:
     - (propagated network, final propagated gradient)
   ** Note that the input layers has to be in inversed layer order
      Which is the order in the network returned from fprop function
*)
fun bprop (layers, gradInput) = 
    List.foldl (fn (layer, (layers', gradInput)) =>
		   case bprop1layer(layer, gradInput) of
		       (bpropedLayer, gradOutput) => 
		       (Matrix.freeMat (gradInput); (bpropedLayer::layers', gradOutput)))
	       ([], gradInput) layers
(* computeCost: compute training/validating cost
   input:
     - output: predicted result from the neural network
     - target: the desired target
	 - costType: type of cost - MSE/NLL/PER
   output:
     - (errors, gradient)
*)
val i:int ref = ref 0;
fun computeCost(output, target, costType) = 
    let
	val nSamples = #1(Matrix.size(output))
	val rSamples = Real.fromInt(nSamples)
	fun calMSE() =
	    let 
		val diff = Matrix.mdcopy (output)
		val _ = Matrix.mdaxpy (~1.0, target, diff)
		val gradient = Matrix.mdcopy(diff)
		val _ = Matrix.mdscal (1.0/rSamples, gradient)
		val _ = Matrix.dappHSquare(Matrix.toVector(diff))
		val meanErr = (Matrix.sumAll (diff))/rSamples
		val _ = Matrix.freeMat(diff)
	    in
		(meanErr, gradient)
	    end
	fun softmax(input) =
	    let
		val nCols = #2 (Matrix.size(input))
		val maxInputVec = Matrix.newVec (nSamples)
		val _ = Matrix.maxCols (input, maxInputVec)
		val _ = Matrix.dscal (~1.0, maxInputVec)
		val maxInput = Matrix.drepcols (maxInputVec, nCols)
		val _ = Matrix.mdaxpy (1.0, input, maxInput)
		val _ = Matrix.dappExp (Matrix.toVector(maxInput))
		
		val expInput = maxInput
		
		val sumExp = Matrix.newVec (nSamples)
		val _ = Matrix.sumCols (expInput, sumExp)
		val _ = Matrix.dappInvr (sumExp)
		val _ = Matrix.mdscalRows (expInput, sumExp)
		val _ = Matrix.freeVec (maxInputVec)
		val _ = Matrix.freeVec (sumExp)
	    in 
			expInput
	    end
	fun calNLL() =
	    let
		val _ = i:= (!i+1)
		val _ = if (!isDebug) then print ("calNN") else ()
		val softmaxOutput = softmax(output)
		(* val _ = Matrix.printfMatrixReal("SoftmaxOutput" ^ Int.toString(!i)^".csv", softmaxOutput) *)
		val gradient = Matrix.mdcopy (softmaxOutput)
		val _ = Matrix.mdaxpy (~1.0, target, gradient)
		val _ = Matrix.mdscal (1.0/rSamples, gradient)
		val _ = Matrix.dappMinusLn (Matrix.toVector(softmaxOutput))
		val errors = Matrix.mdotmul (softmaxOutput, target)
		val meanErr = Matrix.sumAll (errors) / rSamples
		val _ = Matrix.freeMat (errors)
		val _ = Matrix.freeMat (softmaxOutput)
		(* val _ = Matrix.printfMatrixReal("errors" ^ Int.toString(!i)^".csv", softmaxOutput) *)
		(* val _ = Matrix.printfMatrixReal("target" ^ Int.toString(!i)^".csv", target) *)
		(* val _ = Matrix.printfMatrixReal("gradOut" ^ Int.toString(!i)^".csv", gradient) *)
	    in
		(meanErr, gradient)
	    end
	fun calCE() =
	    let
		val errors = Matrix.mdcopy (output)
		val _ = Matrix.dappCe (Matrix.toVector(errors))
		val temp = Matrix.mdotmul (output, target)
		val _ = Matrix.mdaxpy (~1.0, temp, errors)
		val meanErr = Matrix.sumAll (errors) / rSamples
		
		val gradient = Matrix.mdcopy (output)
		val _ = Matrix.dappSigm (Matrix.toVector gradient)
		val _ = Matrix.mdaxpy (~1.0, target, gradient)
		val _ = Matrix.mdscal (1.0/rSamples, gradient)
		val _ = Matrix.freeMat (temp)
	    in
		(meanErr, gradient)		
	    end
	fun calPER() =
	    let
		val idxOutput = Matrix.newVec (nSamples)
		val idxTarget = Matrix.newVec (nSamples)
		val _ = Matrix.maxColsIdx (output, idxOutput)
		val _ = Matrix.maxColsIdx (target, idxTarget)
		val nRights = Matrix.eqcount (idxOutput, idxTarget)
		val _ = Matrix.freeVec (idxOutput)
		val _ = Matrix.freeVec (idxTarget)
		val dummy = Matrix.newMat (1, 1)
		val _ = Matrix.freeMat (dummy)
	    in
		(1.0 - (Real.fromInt(nRights) / rSamples), 
		 dummy)
	    end
	    
    in
	case costType of
	    MSE => calMSE ()
	  | NLL => calNLL ()
	  | CE  => calCE ()
	  | PER => calPER ()
    end
(* update1layer: update 1 layer
   input
     - layer: a layer to be updated
   output
     - updated layer
*)
fun update1layer (layer:nnlayer) =
    let
	val lr = #lr(!params)
	(* val _ = print "update1" *)
	val _ = Matrix.mdaxpy (~lr, #GradW(layer), #W(layer))
	val _ = Matrix.daxpy  (~lr, #GradB(layer), #B(layer))
	
    in
	{input = #input(layer), output = #output(layer), GradW = #GradW(layer),
	GradB = #GradB(layer), actType = #actType(layer), W = #W(layer), B = #B(layer)}
    end

(* update: update the whole network
*)
fun update (layers)  = List.foldr (fn (a, acc) => update1layer(a)::acc) 
				  [] layers
				  
(* 
	Support method for training neural network
*)
val i:int ref = ref 0;
fun train1Batch (layers, input, target) =
	let
	val (fpropedLayers, output) = fprop (layers, input)
	val _ = i:= (!i+1)
(* 	val _ = Matrix.printfMatrixReal("output" ^ Int.toString(!i)^".csv", output)  *)	
	val (errors, grad) = computeCost(output, target, 
					 #costType(!params))
	val (bpropedLayers, temp) = bprop (fpropedLayers, grad)
	val _ = Matrix.freeMat (temp) 
	(* val _ = Matrix.printfMatrixReal("gradback" ^ Int.toString(!i)^".csv", grad) *)
	val _ = if #verbose(!params) then print (Real.toString(errors) ^ "\n")
								else ()
	in
	update(bpropedLayers)
	end
fun trainBatches (layers, inputs, targets) =
	case (inputs, targets) of
	([], []) => layers
	  | (input::inputs', target::targets') => 
	trainBatches(train1Batch(layers, input, target), 
			 inputs', targets')
	| _  => raise Matrix.UnmatchedDimension

fun test1Batch (layers, input, target) =
	let

	val (layers, output) = fprop(layers, input)
	val layers = reorder (layers)
	val (errors, _) = computeCost(output, target, PER)
	in
		(layers, errors)
	end
fun testBatches (layers, inputs, targets) =
	let
		fun sumErr (layers, inputs, targets, err) =
			case (inputs, targets) of
			([], []) => (layers, err)
			| (input::inputs', target::targets') =>
				let val (layers', errors) = test1Batch( layers, input, target)
				in
					sumErr(layers', inputs', targets', err + errors)
				end
			| _  => raise Matrix.UnmatchedDimension
		val (layers', err) = sumErr (layers, inputs, targets, 0.0)
	in
		(layers', err/Real.fromInt(length (inputs)))
	end
(* trainNN: train neural network
   input:
     - data_train: list of batches of training cases
     - target_train: list of batches of desired target
   output:
     - trained neural network
*)
fun trainNN(startLayers, data_train, target_train) =
	let
		fun trainEpoches (layers, nItrs) =
			let val _ = if #verbose(!params) then print 
		("****** epochs: "^Int.toString(nItrs) ^ " *******\n") else ()
			in
			if nItrs = 0 then layers
			else
				trainEpoches(
				trainBatches (layers, data_train, target_train)
				   ,nItrs - 1)
			end
	in
		trainEpoches(startLayers, #nItrs(!params))
	end

(*
	TrainNN on training set and pick the best result on validation set
*)
fun trainBest(startLayers, data_train, target_train, data_validation, target_validation) = 
	let
		fun trainEpoches (layers, nItrs, bestErr) =
			let val _ = if #verbose(!params) then print 
		("****** epochs: "^Int.toString(nItrs) ^ " *******\n") else ()
				(*val (layers, output) = fprop(layers, hd(data_validation))
				val layers = reorder (layers)
				val (errors, _) = computeCost(output, hd(target_validation), PER)*)
				val (layers, errors) = testBatches (layers, data_validation, target_validation)
				val bestErr = if bestErr < errors then bestErr else errors
				val _ = if #verbose(!params) then print 
				("Best Validation Err =  "^ Real.toString(bestErr) ^ "\n") else ()
			in
			if nItrs = 0 then (bestErr, layers)
			else
				trainEpoches(
					trainBatches (layers, data_train, target_train)
				   ,nItrs - 1, bestErr)
			end
	in
		trainEpoches(trainBatches (startLayers, data_train, target_train), #nItrs(!params), 1.0)
	end
(* run: read input data and train a neural network
*)
fun run RandState (p:params, fs:fileNames) =
    let
	val _ = params := p
	val _ = if #verbose(!params) then print 
			("********* Reading data ********\n") else ()
	val data_train = readData(#data_train(fs), 
				  hd(#layerSizes(!params)),
				  #batchsize(!params), #nBatches(!params))
	val labels_train = readData(#labels_train(fs), 
				    List.last(#layerSizes(!params)),
				  #batchsize(!params), #nBatches(!params))
	val data_test = readData(#data_test(fs), 
				 hd(#layerSizes(!params)),
				 #testsize(!params), 1)
	val labels_test = readData(#labels_test(fs), 
				   List.last(#layerSizes(!params)),
				   #testsize(!params), 1)
	val _ = if #verbose(!params) then print 
		("****** Done reading data, start training *****\n") else ()   
	val startLayers = initLayers RandState (#layerSizes(!params), 
				     #initWs(!params), #initBs(!params))

	val trainedLayers = trainNN(startLayers, data_train, labels_train)
	val (_, output) = fprop(trainedLayers, hd(data_test))
	val (errors, _) = computeCost(output, hd(labels_test), 
				      PER)
    in
		(trainedLayers, errors)
    end
end
