(*CM.make ("utils/sources.cm");*)
(*ADATE built-in functions*)
fun realLess(a: real, b:real) = a<b;
fun realAdd(a:real, b:real) = a+b;
fun realSubtract(a:real, b:real) = a-b;
fun realMultiply(a:real, b:real) = a*b;
fun realDevide(a:real, b:real) = a/b;
fun fromInt(a:int) = Real.fromInt(a);
fun sqrt(a:real) = Math.sqrt(a);
fun ln (a:real) = Math.ln(a);
fun tanh (a: real) = Math.tanh(a);
exception D0
(* Datatype *)
datatype real_list = rnil | consr of real * real_list

datatype real_list_list = rlnil | consrl of real_list * real_list_list

datatype weightMatrix = weightMatrix of real * real_list_list

datatype weightMatrix_list = wnil | consw of weightMatrix * weightMatrix_list
datatype layerType = VisHid | HidHid1 | HidHid2 | HidOut
						      
datatype unary = zero | succ of unary

fun isHidHid1 (x) = (x = HidHid1)
val RandomSeed = Random.rand(1, 12)
			
(* Box muller method for generate normal distribution random number *)

fun rand_normal (Mean, Std) =
    let
	fun box_muller () =
	    case realSubtract(realMultiply(2.0, 
			(Random.randReal RandomSeed)), 1.0) of X1
	    => case realSubtract(realMultiply(2.0, 
			(Random.randReal RandomSeed)), 1.0) of X2
	    => case realAdd(realMultiply(X1, X1), realMultiply( X2, X2)) of W
	    => case realLess(W, 1.0) of
		   true   => realAdd(Mean, 
			     realMultiply(Std,
			     realMultiply(X1, 
			     sqrt(realDevide(realMultiply(~2.0, ln(W)), W)))))
		  |false => box_muller()
    in
	box_muller()
    end
fun rand_perm RandState (m, n) =
                let
                        fun fisher_yates (a, i, n, m) =
                                if (i = n orelse i = m) then a
                                else
                                        let val randj = i + Real.floor((Random.randReal RandState)*Real.fromInt(n-i))
                                                val j = if (randj = n) then n-1 else randj
                                                val tmp = Array.sub(a, j)
                                        in
                                                (Array.update(a, j, Array.sub(a, i));
                                                Array.update(a, i, tmp);
                                                fisher_yates (a, i+1, n, m))
                                        end
                        fun seqArray(a, i, n) =
                                if i = n then a
                                else (Array.update(a, i, i+1);
                                        seqArray(a, i+1, n))
                        fun arrayToList (a, i, l) =
                                if ((i = Array.length(a)) orelse (l = 0)) then []
                                else Array.sub(a, i)::arrayToList(a, i+1, l-1)
                        val array_perm = fisher_yates(seqArray(Array.array(n, 0), 0, n), 0, n, m)
                        val list_perm = arrayToList(array_perm, 0, m)
                in
                        ListMergeSort.sort (fn (a, b) => a>b) list_perm
                end

fun rand_2 () =
	2.0*(Random.randReal RandomSeed) - 1.0;

fun f( NInputs, NOutputs, LayerType ) = 
 case realLess( NInputs, NOutputs ) of 
 false => 
 consr( tanh( NInputs ), consr( tanh( NOutputs ), rnil ) ) 
 | true => rnil 

fun initW ( (NInputs, NOutputs, LayerType) : real * real * layerType ) : weightMatrix =
let
  fun initSparseVects( I : real ) : real_list_list =
      case realLess (I, NOutputs) of
        false => rlnil
      | true =>
          consrl( f( NInputs, NOutputs, LayerType ), initSparseVects( I + 1.0 ) )
in
  weightMatrix( NInputs, initSparseVects 0.0 )
end
fun main ((Layer1size, Layer2size, Layer3size, Layer4size, Layer5size):
	  (real * real * real * real * real)) : weightMatrix_list =
    consw(initW(Layer1size, Layer2size, VisHid),
    consw(initW(Layer2size, Layer3size, HidHid1),
    consw(initW(Layer3size, Layer4size, HidHid2),
    consw(initW(Layer4size, Layer5size, HidOut),
	  wnil))));

val params:NN.params =  
    {batchsize = 20,   (* number of training cases per batch *)
     nBatches = 400,    (* number of batches *)
     testsize = 1298,  (* number of testing cases *)
	 nTestBatches = 1,
     lambda = 0.99,      (* momentum coefficient *)
     momentumSchedule = false, (* momentum schedule *)
     maxLambda = 0.0,    (* max momentum *)
     lr = 0.05,        (* learning rate *)
     costType = NN.NLL,   (* cost function type *)
     initType = NN.SPARSE,(* initialization type *)
     actType = NN.TANH,   (* activation function type *)
     layerSizes = [256, 200, 200, 800, 10], (* structure of network *)
     nItrs = 10,         (* number of iterations/epoches *)
     initWs = [],        (* init weight matrices *)
     initBs = [],
     wdType = NN.L2,     (* weight decay type *)
     wdValue = 0.00001,
     verbose = true
    };
(* all training and validation use the same Network structure *)
val nInputCases = 1;
val nValidationCases = 1;
val Inputs = [(256, 200, 200, 800, 10)]
val Validations = [(256, 200, 200, 800, 10)]
val input_files = [("../datasets/USPS/USPS_data.csv", "../datasets/USPS/USPS_labels.csv", 
        "../datasets/USPS/USPS_datavalid.csv", "../datasets/USPS/USPS_labelsvalid.csv")];
fun readData (data, target, data_valid, target_valid) =
    (NN.readData(data, 
		 hd(#layerSizes(params)),
		 #batchsize(params), #nBatches(params)),
     NN.readData(target, 
		 List.last(#layerSizes(params)),
		 #batchsize(params), #nBatches(params)),
     NN.readData(data_valid, 
		 hd(#layerSizes(params)),
		 #testsize(params), #nTestBatches(params)),
     NN.readData(target_valid, 
		 List.last(#layerSizes(params)),
		 #testsize(params), #nTestBatches(params)))
(* real input data for training process *)
val input_data = Array.fromList (List.map readData input_files)
				
(* convert ADATE list type to ML list *)
(* convert ADATE list type to ML list *)
fun toRealListListList (Ws: weightMatrix_list):(real * real list list) list =
    let
  fun toRealList (rs): real list =
      case rs of
    rnil => []
         |consr(r, rs') => r::toRealList(rs')
  fun toRealListList (rls: real_list_list): real list list =
      case rls of
    rlnil => []
         |consrl(rl, rls') => toRealList(rl)::toRealListList(rls')
    in
  case Ws of
      wnil => []
    | consw( weightMatrix(nInputs, rls),Ws') => 
      (nInputs, toRealListList(rls)) :: toRealListListList(Ws')
    end
	
fun toWeightList RandState (Ws: (real * real list list) list) 
    : Matrix.matrix option list =
    let
  fun initWeightMatrix(arg as (n, W):(real * real list list))
      : Matrix.matrix option =
      let
    val nInputs = Real.floor (n)
    fun initSparseList ((ids, initializedWeights, nInputs, I):
            (int list * real list * int * int))
        : real list =
        case (ids, I <= nInputs) of
      (_, false) => []
           |( [], true)  => 
    0.0::initSparseList (ids, initializedWeights, nInputs, I+1)
           |( idx::ids', true) => 
      if (idx = I) then
          hd(initializedWeights)
          ::initSparseList (ids', 
                tl(initializedWeights), 
                nInputs, I+1)
      else
          0.0::initSparseList (ids, 
             initializedWeights, 
             nInputs, I+1)
    fun initSparseVects (nInputs, vs) =
        case vs of
      [] => []
           |v::vs' => initSparseList(
             (rand_perm RandState (length(v), nInputs)),
             v, nInputs, 1)::
          (initSparseVects(nInputs, vs'))
      in
    SOME (Matrix.fromLists(initSparseVects(nInputs, W), 
                (nInputs, length(W))))
      end
    in
  case Ws of
      [] => []
     |W::Ws' => initWeightMatrix(W) :: toWeightList RandState (Ws')
    end


fun output_eval_fun( I: int, WeightList : weightMatrix_list) =
    let 
	val  RandState = Random.rand( 10, I )
	val (data_train, labels_train, data_test, labels_test) = 
	    Array.sub(input_data, I)
		
	val Ws = toWeightList RandState (toRealListListList(WeightList))
	val _ = NN.setParams (params)
	val startLayers = NN.initLayers RandState (#layerSizes(params), Ws, [])
	val (errors, trainedLayers) = NN.trainBest(startLayers, data_train, labels_train, data_test, labels_test)
	val _ = NN.freeNN(trainedLayers)
    in
		errors
	end
fun test (I)=
	output_eval_fun(I, main(256.0, 200.0, 200.0, 800.0, 10.0))
val _ = test(0)