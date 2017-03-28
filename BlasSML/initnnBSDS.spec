
datatype aUnit = aUnit

datatype real_list = rnil | consr of real * real_list

datatype real_list_list = rlnil | consrl of real_list * real_list_list

datatype weightMatrix = weightMatrix of real * real_list_list

datatype weightMatrix_list = wnil | consw of weightMatrix * weightMatrix_list
datatype layerType = visHid | hidHid1 | hidHid2 | hidOut
                  
fun rconstLess( ( X, C ) : real * rconst ) : bool =
  case C of rconst( Compl, StepSize, Current ) => realLess( X, Current )


fun rand_normal( ( Mean, Std) : real * real ) : real =
    let
  fun box_muller( Dummy : aUnit ) : real  =
      case realSubtract(realMultiply(2.0, 
      (aRand 0)), 1.0) of X1
      => case realSubtract(realMultiply(2.0, 
      (aRand 0)), 1.0 ) of X2
      => case realAdd(realMultiply(X1, X1), realMultiply( X2, X2)) of W
      => case realLess(W, 1.0) of
       true   => realAdd(Mean, 
           realMultiply(Std,
           realMultiply(X1, 
           sqrt(realDivide(realMultiply(~2.0, ln(W)), W)))))
      | false => box_muller aUnit
    in
  box_muller aUnit
    end
  
fun f( ( NInputs, NOutputs, LayerType) : real * real * layerType ) : real_list =
let
        fun h( N: real) : real_list =
                case 0.0 < N of
                        false => rnil
                |       true =>
             consr( realDivide (realMultiply(
                sqrt( tor( rconst( 0, 3.0, 6.0 ) ) ),
                tor( rconst( 0, 1.0, 2.0 ) ) * (aRand 0) - tor( rconst( 0, 0.5, 1.0 ) ) ),
                                sqrt(realAdd(NInputs, NOutputs)) ),
                        h(realSubtract( N, 1.0 )) )
in
        h NInputs
end

      
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
  
fun main ((Layer1size, Layer2size, Layer3size, Layer4size, Layer5size) : 
    real * real * real * real * real ) : weightMatrix_list =
    consw(initW(Layer1size, Layer2size, visHid),
    consw(initW(Layer2size, Layer3size, hidHid1),
    consw(initW(Layer3size, Layer4size, hidHid2),
    consw(initW(Layer4size, Layer5size, hidOut),
    wnil))))


%% 

val NumInputs = 
  case getCommandOption "--numInputs" of SOME S => 
  case Int.fromString S of SOME N => N


val NumIterations = 
  case getCommandOption "--numIterations" of SOME S => 
  case Int.fromString S of SOME N => N


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




val TrainParams : NN.params =  
    {batchsize = 100,   (* number of training cases per batch *)
     nBatches = 100,    (* number of batches *)
     testsize = 30000,  (* number of testing cases *)
	 nTestBatches = 1, (* because of memory issue, we should split big test into smaller batches *)
     lambda = 0.99,      (* momentum coefficient *)
     momentumSchedule = false, (* momentum schedule *)
     maxLambda = 0.0,    (* max momentum *)
     lr = 0.05,        (* learning rate *)
     costType = NN.NLL,   (* cost function type *)
     initType = NN.SPARSE,(* initialization type *)
     actType = NN.TANH,   (* activation function type *)
     layerSizes = [25, 50, 50, 20, 2], (* structure of network *)
     nItrs = NumIterations,         (* number of iterations/epoches *)
     initWs = [],        (* init weight matrices *)
     initBs = [],
     wdType = NN.L2,     (* weight decay type *)
     wdValue = 0.00001,
     verbose = false
    };

val ValidationParams : NN.params =  
    {batchsize = 100,   (* number of training cases per batch *)
     nBatches = 100,    (* Was: 300. number of batches *)
     testsize = 50000,  (* number of testing cases *)
	 nTestBatches = 1, (* because of memory issue, we should split big test into smaller batches *)
     lambda = 0.99,      (* momentum coefficient *)
     momentumSchedule = false, (* momentum schedule *)
     maxLambda = 0.0,    (* max momentum *)
     lr = 0.05,        (* learning rate *)
     costType = NN.NLL,   (* cost function type *)
     initType = NN.SPARSE,(* initialization type *)
     actType = NN.TANH,   (* activation function type *)
     layerSizes = [25, 50, 50, 20, 2], (* structure of network *)
     nItrs = 20,         (* Was: 80. number of iterations/epoches *)
     initWs = [],        (* init weight matrices *)
     initBs = [],
     wdType = NN.L2,     (* weight decay type *)
     wdValue = 0.00001,
     verbose = false
    };

(* all training and validation use the same Network structure *)

val Inputs = [(25.0, 50.0, 50.0, 20.0, 2.0),
        (25.0, 50.0, 50.0, 20.0, 2.0),
        (25.0, 50.0, 50.0, 20.0, 2.0),
        (25.0, 50.0, 50.0, 20.0, 2.0),
        (25.0, 50.0, 50.0, 20.0, 2.0),
		(25.0, 50.0, 50.0, 20.0, 2.0),
        (25.0, 50.0, 50.0, 20.0, 2.0),
        (25.0, 50.0, 50.0, 20.0, 2.0),
        (25.0, 50.0, 50.0, 20.0, 2.0),
        (25.0, 50.0, 50.0, 20.0, 2.0),
		(25.0, 50.0, 50.0, 20.0, 2.0),
        (25.0, 50.0, 50.0, 20.0, 2.0),
        (25.0, 50.0, 50.0, 20.0, 2.0),
        (25.0, 50.0, 50.0, 20.0, 2.0),
        (25.0, 50.0, 50.0, 20.0, 2.0),
		(25.0, 50.0, 50.0, 20.0, 2.0),
        (25.0, 50.0, 50.0, 20.0, 2.0),
        (25.0, 50.0, 50.0, 20.0, 2.0),
        (25.0, 50.0, 50.0, 20.0, 2.0),
        (25.0, 50.0, 50.0, 20.0, 2.0)]

val Test_inputs = 
     [ (25.0, 50.0, 50.0, 20.0, 2.0),
       (25.0, 50.0, 50.0, 20.0, 2.0),
       (25.0, 50.0, 50.0, 20.0, 2.0),
       (25.0, 50.0, 50.0, 20.0, 2.0),
       (25.0, 50.0, 50.0, 20.0, 2.0),
		(25.0, 50.0, 50.0, 20.0, 2.0),
        (25.0, 50.0, 50.0, 20.0, 2.0),
        (25.0, 50.0, 50.0, 20.0, 2.0),
        (25.0, 50.0, 50.0, 20.0, 2.0),
        (25.0, 50.0, 50.0, 20.0, 2.0),
		(25.0, 50.0, 50.0, 20.0, 2.0),
        (25.0, 50.0, 50.0, 20.0, 2.0),
        (25.0, 50.0, 50.0, 20.0, 2.0),
        (25.0, 50.0, 50.0, 20.0, 2.0),
        (25.0, 50.0, 50.0, 20.0, 2.0),
		(25.0, 50.0, 50.0, 20.0, 2.0),
        (25.0, 50.0, 50.0, 20.0, 2.0),
        (25.0, 50.0, 50.0, 20.0, 2.0),
        (25.0, 50.0, 50.0, 20.0, 2.0),
        (25.0, 50.0, 50.0, 20.0, 2.0)
        ]

val input_files = [("/local/BSDS/data1.csv", "/local/BSDS/labels1.csv", 
        "/local/BSDS/datavalid1.csv", "/local/BSDS/labelsvalid1.csv"), 
       ("/local/BSDS/data2.csv", "/local/BSDS/labels2.csv", 
        "/local/BSDS/datavalid2.csv", "/local/BSDS/labelsvalid2.csv"),
       ("/local/BSDS/data3.csv", "/local/BSDS/labels3.csv", 
        "/local/BSDS/datavalid3.csv", "/local/BSDS/labelsvalid3.csv"),
       ("/local/BSDS/data4.csv", "/local/BSDS/labels4.csv", 
        "/local/BSDS/datavalid4.csv", "/local/BSDS/labelsvalid4.csv"),
       ("/local/BSDS/data5.csv", "/local/BSDS/labels5.csv",
        "/local/BSDS/datavalid5.csv", "/local/BSDS/labelsvalid5.csv"),
       ("/local/BSDS/data6.csv", "/local/BSDS/labels6.csv",
        "/local/BSDS/datavalid6.csv", "/local/BSDS/labelsvalid6.csv"),
       ("/local/BSDS/data7.csv", "/local/BSDS/labels7.csv",
        "/local/BSDS/datavalid7.csv", "/local/BSDS/labelsvalid7.csv"),
       ("/local/BSDS/data8.csv", "/local/BSDS/labels8.csv", 
        "/local/BSDS/datavalid8.csv", "/local/BSDS/labelsvalid8.csv"),
       ("/local/BSDS/data9.csv", "/local/BSDS/labels9.csv",
        "/local/BSDS/datavalid9.csv", "/local/BSDS/labelsvalid9.csv"),
       ("/local/BSDS/data10.csv", "/local/BSDS/labels10.csv", 
        "/local/BSDS/datavalid10.csv", "/local/BSDS/labelsvalid10.csv"),
		
		("/local/BSDS/data11.csv", "/local/BSDS/labels11.csv", 
        "/local/BSDS/datavalid11.csv", "/local/BSDS/labelsvalid11.csv"), 
       ("/local/BSDS/data12.csv", "/local/BSDS/labels12.csv", 
        "/local/BSDS/datavalid12.csv", "/local/BSDS/labelsvalid12.csv"),
       ("/local/BSDS/data13.csv", "/local/BSDS/labels13.csv", 
        "/local/BSDS/datavalid13.csv", "/local/BSDS/labelsvalid13.csv"),
       ("/local/BSDS/data14.csv", "/local/BSDS/labels14.csv", 
        "/local/BSDS/datavalid14.csv", "/local/BSDS/labelsvalid14.csv"),
       ("/local/BSDS/data15.csv", "/local/BSDS/labels15.csv",
        "/local/BSDS/datavalid15.csv", "/local/BSDS/labelsvalid15.csv"),
       ("/local/BSDS/data16.csv", "/local/BSDS/labels16.csv",
        "/local/BSDS/datavalid16.csv", "/local/BSDS/labelsvalid16.csv"),
       ("/local/BSDS/data17.csv", "/local/BSDS/labels17.csv",
        "/local/BSDS/datavalid17.csv", "/local/BSDS/labelsvalid17.csv"),
       ("/local/BSDS/data18.csv", "/local/BSDS/labels18.csv", 
        "/local/BSDS/datavalid18.csv", "/local/BSDS/labelsvalid18.csv"),
       ("/local/BSDS/data19.csv", "/local/BSDS/labels19.csv",
        "/local/BSDS/datavalid19.csv", "/local/BSDS/labelsvalid19.csv"),
       ("/local/BSDS/data20.csv", "/local/BSDS/labels20.csv", 
        "/local/BSDS/datavalid20.csv", "/local/BSDS/labelsvalid20.csv"),
		
		("/local/BSDS/data21.csv", "/local/BSDS/labels21.csv", 
        "/local/BSDS/datavalid21.csv", "/local/BSDS/labelsvalid21.csv"), 
       ("/local/BSDS/data22.csv", "/local/BSDS/labels22.csv", 
        "/local/BSDS/datavalid22.csv", "/local/BSDS/labelsvalid22.csv"),
       ("/local/BSDS/data23.csv", "/local/BSDS/labels23.csv", 
        "/local/BSDS/datavalid23.csv", "/local/BSDS/labelsvalid23.csv"),
       ("/local/BSDS/data24.csv", "/local/BSDS/labels24.csv", 
        "/local/BSDS/datavalid24.csv", "/local/BSDS/labelsvalid24.csv"),
       ("/local/BSDS/data25.csv", "/local/BSDS/labels25.csv",
        "/local/BSDS/datavalid25.csv", "/local/BSDS/labelsvalid25.csv"),
       ("/local/BSDS/data26.csv", "/local/BSDS/labels26.csv",
        "/local/BSDS/datavalid26.csv", "/local/BSDS/labelsvalid26.csv"),
       ("/local/BSDS/data27.csv", "/local/BSDS/labels27.csv",
        "/local/BSDS/datavalid27.csv", "/local/BSDS/labelsvalid27.csv"),
       ("/local/BSDS/data28.csv", "/local/BSDS/labels28.csv", 
        "/local/BSDS/datavalid28.csv", "/local/BSDS/labelsvalid28.csv"),
       ("/local/BSDS/data29.csv", "/local/BSDS/labels29.csv",
        "/local/BSDS/datavalid29.csv", "/local/BSDS/labelsvalid29.csv"),
       ("/local/BSDS/data30.csv", "/local/BSDS/labels30.csv", 
        "/local/BSDS/datavalid30.csv", "/local/BSDS/labelsvalid30.csv"),
		
		("/local/BSDS/data31.csv", "/local/BSDS/labels31.csv", 
        "/local/BSDS/datavalid31.csv", "/local/BSDS/labelsvalid31.csv"), 
       ("/local/BSDS/data32.csv", "/local/BSDS/labels32.csv", 
        "/local/BSDS/datavalid32.csv", "/local/BSDS/labelsvalid32.csv"),
       ("/local/BSDS/data33.csv", "/local/BSDS/labels33.csv", 
        "/local/BSDS/datavalid33.csv", "/local/BSDS/labelsvalid33.csv"),
       ("/local/BSDS/data34.csv", "/local/BSDS/labels34.csv", 
        "/local/BSDS/datavalid34.csv", "/local/BSDS/labelsvalid34.csv"),
       ("/local/BSDS/data35.csv", "/local/BSDS/labels35.csv",
        "/local/BSDS/datavalid35.csv", "/local/BSDS/labelsvalid35.csv"),
       ("/local/BSDS/data36.csv", "/local/BSDS/labels36.csv",
        "/local/BSDS/datavalid36.csv", "/local/BSDS/labelsvalid36.csv"),
       ("/local/BSDS/data37.csv", "/local/BSDS/labels37.csv",
        "/local/BSDS/datavalid37.csv", "/local/BSDS/labelsvalid37.csv"),
       ("/local/BSDS/data38.csv", "/local/BSDS/labels38.csv", 
        "/local/BSDS/datavalid38.csv", "/local/BSDS/labelsvalid38.csv"),
       ("/local/BSDS/data39.csv", "/local/BSDS/labels39.csv",
        "/local/BSDS/datavalid39.csv", "/local/BSDS/labelsvalid39.csv"),
       ("/local/BSDS/data40.csv", "/local/BSDS/labels40.csv", 
        "/local/BSDS/datavalid40.csv", "/local/BSDS/labelsvalid40.csv")];

fun readData (data, target, data_valid, target_valid) =
    (NN.readData(data, 
     hd(#layerSizes(TrainParams)),
     #batchsize(TrainParams), #nBatches(TrainParams)),
     NN.readData(target, 
     List.last(#layerSizes(TrainParams)),
     #batchsize(TrainParams), #nBatches(TrainParams)),
     NN.readData(data_valid, 
     hd(#layerSizes(TrainParams)),
     #testsize(TrainParams), 1),
     NN.readData(target_valid, 
     List.last(#layerSizes(TrainParams)),
     #testsize(TrainParams), 1))
(* real input data for training process *)
val input_data = Array.fromList (List.map readData input_files)
        
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

val Abstract_types = []
val Reject_funs = []
fun restore_transform D = D
fun compile_transform D = D
val print_synted_program  = Print.print_dec'


       
val Funs_to_use = [
  "false", "true",
  "realLess", "realAdd", "realSubtract", "realMultiply",
  "tanh",
  "tor", "rconstLess",
  "rand_normal",
  "0",
  "aRand",
  "rnil", "consr"
  ]


fun to( G : real ) : LargeInt.int =
    Real.toLargeInt IEEEReal.TO_NEAREST ( G * 1.0e14 )

structure Grade : GRADE =
struct
type grade = LargeInt.int
val NONE = LargeInt.maxInt (* To check that LargeInt has infinite precision. *)
val zero = LargeInt.fromInt 0
val op+ = LargeInt.+
val comparisons = [ LargeInt.compare ]
val N = LargeInt.fromInt 1000000 * LargeInt.fromInt 1000000
val significantComparisons = [ fn( E1, E2 )  => LargeInt.compare( E1 div N, E2 div N )  ]

fun toString( G : grade ) : string =
  Real.toString( Real.fromLargeInt G / 1.0E14 )

val pack = LargeInt.toString

fun unpack( S : string ) : grade =
  case LargeInt.fromString S of SOME G => G

val post_process = fn X => X

val toRealOpt = NONE

end


val Inputs = take( NumInputs, Inputs )
val MaxInputs = 20

fun output_eval_fun( exactlyOne( I: int, _ : (real * real * real * real * real), 
         WeightList : weightMatrix_list ) ) = [
let 
  val _ = NN.setParams( if I < Int64.fromInt NumInputs then TrainParams else ValidationParams )
  val  RandState = Random.rand( 10, Int64.toInt I )
  val DataIdx = if Int64.toInt I <  NumInputs then Int64.toInt I else Int64.toInt I - NumInputs + MaxInputs
  val (data_train, labels_train, data_test, labels_test) = 
    Array.sub(input_data, DataIdx)
  val Ws = toWeightList RandState (toRealListListList(WeightList))
  val startLayers = NN.initLayers RandState (#layerSizes(TrainParams), Ws, [])
  val trainedLayers = NN.trainNN(startLayers, data_train, labels_train)
  val (tempLayers, errors) = NN.testBatches(trainedLayers, data_test, labels_test)
  val _ = NN.freeNN (tempLayers)
(*
  val () = (
    p"\noutput_eval_fun: I = "; print_int64 I;
    p"  errors = "; print_real errors;
    p"\n"
    )
*)
in
  if errors > 1.0E30 orelse not (  Real.isFinite errors  ) then
      { numCorrect = 0 : int, numWrong = 1 : int, grade = to 1.0E30 }
  else
      { numCorrect = 1, numWrong = 0, grade = to errors }
end
    ]
  

exception MaxSyntComplExn
val MaxSyntCompl = (
  case getCommandOption "--maxSyntacticComplexity" of
    NONE => 150.0
  | SOME S => case Real.fromString S of SOME N => N
  ) handle Ex => raise MaxSyntComplExn


fun rlEq( rnil,  rnil ) = true
  | rlEq( rnil,  consr( _, _ ) ) = false
  | rlEq( consr( _, _ ), _ ) = false
  | rlEq( consr( X1, Xs1 ), consr( Y1, Ys1 ) ) = real_eq( X1, Y1 ) andalso rlEq( Xs1, Ys1 )


fun rllEq( rlnil,  rlnil ) = true
  | rllEq( rlnil,  consrl( _, _ ) ) = false
  | rllEq( consrl( _, _ ), _ ) = false
  | rllEq( consrl( X1, Xs1 ), consrl( Y1, Ys1 ) ) = rlEq( X1, Y1 ) andalso rllEq( Xs1, Ys1 )

fun wmEq( weightMatrix( X, Xss ), weightMatrix( Y, Yss ) ) =
  real_eq( X, Y ) andalso rllEq( Xss, Yss )

fun wlEq( wnil,  wnil ) = true
  | wlEq( wnil,  consw( _, _ ) ) = false
  | wlEq( consw( _, _ ), _ ) = false
  | wlEq( consw( X1, Xs1 ), consw( Y1, Ys1 ) ) = wmEq( X1, Y1 ) andalso wlEq( Xs1, Ys1 )


val AllAtOnce = false
val OnlyCountCalls = false
val TimeLimit : Int.int = 10000000
val max_time_limit = fn () => Word64.fromInt TimeLimit : Word64.word
val max_test_time_limit = fn () => Word64.fromInt TimeLimit : Word64.word
val time_limit_base = fn () => real TimeLimit

fun max_syntactic_complexity() = MaxSyntCompl
fun min_syntactic_complexity() = 0.0
val Use_test_data_for_max_syntactic_complexity = false

val main_range_eq = wlEq
val File_name_extension = 
  "numIterations" ^ Int.toString NumIterations ^ 
  "numInputs" ^ Int.toString NumInputs


val Resolution = NONE
val StochasticMode = false

val Number_of_output_attributes : Int64.int = 4


fun terminate( Nc, G )  = false

