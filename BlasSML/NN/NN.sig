signature NN =
sig
    type matrix = Matrix.matrix
	type vector = Matrix.vector
    datatype costType = NLL|MSE|CE|PER
    datatype initType = SPARSE|NORMAL|NORMALISED
    datatype actType = SIGM|TANH|LINEAR
	datatype wdType = L0|L1|L2
    type nnlayer = {input: Matrix.matrix, (* ROWMATRIX *)
		output: Matrix.matrix,(* ROWMATRIX *)
		GradW: Matrix.matrix, (* COLMATRIX *)
		GradB: Matrix.vector, 
		B: Matrix.vector,     
		W: Matrix.matrix,      (* COLMATRIX *)
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
		  initWs: Matrix.matrix option list, (* preinitialized Ws matrices *)
		  initBs: Matrix.vector option list, (* preinitialized Bs matrices *)
	      nItrs: int,         (* number of iterations/epoches *)
		  wdType: wdType,
		  wdValue: real,
		  verbose: bool
	     }
    type fileNames = { data_train: string,
			labels_train: string,
			data_test: string,
			labels_test: string
		}	
   
    exception InputError
    exception NotSupported
	
    val setParams:    params -> unit
    val run:   Random.rand ->     params * fileNames -> nnlayer list * real
	val readData:     string * int * int * int -> matrix list
    val trainNN:      nnlayer list * matrix list 
		       * matrix list -> nnlayer list
	val trainBest:    nnlayer list * matrix list * matrix list
					* matrix list * matrix list -> real * nnlayer list
	val freeNN:       nnlayer list -> unit
    val update:       nnlayer list -> nnlayer list
    val update1layer: nnlayer -> nnlayer
    val computeCost:  matrix * matrix * costType 
		      -> real * matrix
    val bprop:        nnlayer list * matrix 
		      -> nnlayer list * matrix
    val bprop1layer:  nnlayer * matrix -> nnlayer * matrix

    val fprop:        nnlayer list * matrix 
		      -> nnlayer list * matrix
    val fprop1layer:  nnlayer * matrix -> nnlayer * matrix

    val initLayers: Random.rand ->  int list * matrix option list * vector option list -> nnlayer list
    val initLayer:  Random.rand ->  int * int * actType * matrix option * vector option-> nnlayer
end

