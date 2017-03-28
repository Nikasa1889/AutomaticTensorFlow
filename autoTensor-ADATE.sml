(* Author: Dang Ha The Hien 
   Date:   13/03/2017
   Last Modified: 27/03/2017
   Define a set of datatype that allows ADATE to generate 
            an arbitrary tensorflow computation graph.
   Note: do not take into account the batch-size dimension of a tensor
*)
exception NA1   (*NotImplemented*)
exception NA2   (*UnmatchedDimension*)
exception NA3   (*ArgumentOutOfRange*)
exception NA4   (*TensorNil*)
(* detail of tensor datatype is not exposed to the f function 
   it must use the helper functions to produce output_tensor from input_tensor*)
signature TENSOR = sig
    type tensor
    type tensor_list
    type input_tensor
    type output_tensor
    datatype output_tensor_list: nil | c of output_tensor * output_tensor_list
    
    val c:            output_tensor * output_tensor_list -> output_tensor_list
    
    val fromInput:    input_tensor      -> tensor
    val toOutput:     tensor            -> output_tensor
    val fullyConnect: tensor * real     -> tensor
    val split:        tensor * real     -> tensor * tensor
    val head:         tensor * real     -> tensor
    val tail:         tensor * real     -> tensor
    val concat:       tensor * tensor   -> tensor
    val relu:         tensor            -> tensor
    val tanh:         tensor            -> tensor
    val sigmoid:      tensor            -> tensor
    val sqrt:         tensor            -> tensor
    val dropout:      tensor * real     -> tensor
    val add:          tensor * tensor   -> tensor
    val multiply:     tensor * tensor   -> tensor
    val substract:    tensor * tensor   -> tensor
    (*allow averaging prediction of multiple path*)
    val averageOutput: output_tensor_list -> output_tensor 
end
val Funs_to_use = [ 
  "false", "true", 
  "numberLess", 
  "numberTimes", "numberAdd", "numberMinus",
  "positive", "negative",
  "nil", "c", (*able to construct list of output_tensor*)
  "fromInput", "toOutput", "averageOutput",
  "fullyConnect", "split", "head", "tail", "concat",
  "relu", "tanh", "sigmoid", "sqrt", "dropout",
  "dropout", "add", "multiply", "substract"]
  
datatype dim = dim_1 of int | dim_2 of int * int | dim_3 of int * int * int
datatype operation = placeholder_c
                | fullyConnect_c
                | softmax_c
                (*| conv2d_c of  (int * int) * int * int 
                        (* kernel_size * strides * padding *)
                | max_pool_c of (int * int) * int * int 
                        (* kernel_size * strides * padding *) *)
                | head_c
                | tail_c
                | splitR_c (* percentage of the first *)
                | splitL_c
                | concat_c
                | relu_c
                | tanh_c
                | sigmoid_c
                | sqrt_c 
                (*| square_c *)
                | dropout_c of real
                | add_c
                | multiply_c
                | substract_c
                | averageOutput_c
                (*| maximum_c*)
datatype tensor_list = nil | c of tensor * tensor_list
and tensor = tensor_nil | tensor_cons of int * tensor_list * operation * dim
type input_tensor = tensor
type output_tensor = tensor
type output_tensor_list = tensor_list

val tensor_id = ref 0
fun generateId () =
        (tensor_id := !tensor_id + 1;!tensor_id)

fun tensor_c (Tensors, Op, InputDim) = 
        tensor_cons (generateId(), Tensors, Op, InputDim)
    
fun fromInput (Input as tensor_nil) = raise NA4
  | fromInput (Input as tensor_cons(ID, Parents as nil, Oper as placeholder_c, InputDim)) = 
        tensor_c(nil, placeholder_c, InputDim)
  | fromInput (Input as tensor_cons(ID, Parents, Oper, InputDim)) = raise NA1 
   (* input tensor must have operation placeholder and has no dependency *)
    
fun toOutput (Input as tensor_nil) = raise NA4
  | toOutput (Input as tensor_cons (ID, nodes, oper, InputDim)) = 
        (* Specify the required tensor output dimension here, 
        only support 1D output now *)
        case dim_1(10) of OutDim =>
        (* only support classification now *)
            case InputDim of
                dim_1 (n) => tensor_c (c (Input, nil), 
                                        softmax_c, OutDim)
               |dim_2 (n1, n2) => raise NA1
               |dim_3 (n1, n2, n3) => raise NA1
(* Define helper functions for all of the allowed tensor operations here *)
fun fullyConnect (Input as tensor_nil, ScaleFactor) = raise NA4
  | fullyConnect (Input as tensor_cons (ID, Parents, Oper, InputDim as dim_2(n1, n2)), ScaleFactor) =  
        raise NA1
  | fullyConnect (Input as tensor_cons (ID, Parents, Oper, InputDim as dim_3(n1, n2, n3)), ScaleFactor) =  
        raise NA1
  | fullyConnect (Input as tensor_cons (ID, Parents, Oper, InputDim as dim_1(n)), ScaleFactor) = 
        case dim_1(floor(real(n) * ScaleFactor)) of OutDim =>
            tensor_c (c(Input, nil), fullyConnect_c, OutDim)

fun split (Input as tensor_nil, SplitFactor) = raise NA4
  | split (Input as tensor_cons (ID, Parents, Oper, InputDim as dim_2(n1, n2)), SplitFactor) = 
        raise NA1
  | split (Input as tensor_cons (ID, Parents, Oper, InputDim as dim_3(n1, n2, n3)), SplitFactor) = 
        raise NA1
  | split (Input as tensor_cons (ID, Parents, Oper, InputDim as dim_1(n)), SplitFactor) =
        case (SplitFactor > 0.0) of
            true => (case (SplitFactor < 1.0) of
                    true =>
                     (case dim_1( floor(real(n) * SplitFactor)) of OutDim1 =>
                      case dim_1( n - floor(real(n) * SplitFactor)) of OutDim2 =>
                      (tensor_c (c (Input, nil), splitL_c, OutDim1),
                       tensor_c (c (Input, nil), splitR_c, OutDim2)))
                   |false =>  raise NA3)
           |false => raise NA3
              


fun head (Input as tensor_nil, nElems) = raise NA4
  | head (Input as tensor_cons (ID, Parents, Oper, InputDim as dim_2(n1, n2)), nElems)=
        raise NA1
  | head (Input as tensor_cons (ID, Parents, Oper, InputDim as  dim_3(n1, n2, n3)), nElems) = 
        raise NA1
  | head (Input as tensor_cons (ID, Parents, Oper, InputDim as  dim_1(n)), nElems) =
        case (nElems > 0.0) of
            true => (case (floor(nElems) < n) of
                     true =>
                       (case dim_1( floor(nElems)) of OutDim =>
                        tensor_c (c (Input, nil), head_c, OutDim))
                    |false => raise NA3)
           |false => raise NA3

fun tail (Input as tensor_nil, nElems) = raise NA4
  | tail (Input as tensor_cons (ID, Parents, Oper, InputDim as dim_2(n1, n2)), nElems) = raise NA1
  | tail (Input as tensor_cons (ID, Parents, Oper, InputDim as dim_3(n1, n2, n3)), nElems) = raise NA1
  | tail (Input as tensor_cons (ID, Parents, Oper, InputDim as dim_1(n)), nElems) =
        case (nElems > 0.0) of 
            true => (case (floor(nElems) < n) of
                    true =>
                       (case dim_1( floor(nElems)) of OutDim =>
                        tensor_c (c (Input, nil), tail_c, OutDim))
                   |false => raise NA3)
           |false => raise NA3
                
   
   
fun concat (Input1 as tensor_nil, Input2) = raise NA4
   |concat (Input1, Input2 as tensor_nil) = raise NA4
   |concat (Input1 as tensor_cons (ID1, Parents1, Oper1, InputDim1 as dim_1(n1)), 
            Input2 as tensor_cons (ID2, Parents2, Oper2, InputDim2 as dim_1(n2))) = 
        (case dim_1(n1 + n2) of OutDim =>
            tensor_c (c(Input1, c(Input2, nil)), concat_c, OutDim))
   |concat (Input1, Input2) = raise NA4
                
fun relu (Input as tensor_nil) = raise NA4
   |relu (Input as tensor_cons (ID, Parents, Oper, InputDim)) = 
        tensor_c (c(Input, nil), relu_c, InputDim)
        
fun sigmoid (Input as tensor_nil) = raise NA4
   |sigmoid (Input as tensor_cons (ID, Parents, Oper, InputDim)) = 
        tensor_c (c(Input, nil), sigmoid_c, InputDim)
           
fun tanh (Input as tensor_nil) = raise NA4
   |tanh (Input as tensor_cons (ID, Parents, Oper, InputDim)) = 
        tensor_c (c(Input, nil), tanh_c, InputDim)
        
fun sqrt (Input as tensor_nil) = raise NA4
   |sqrt (Input as tensor_cons (ID, Parents, Oper, InputDim)) = 
        tensor_c (c(Input, nil), sqrt_c, InputDim)
        
fun dropout (Input as tensor_nil, KeepRate) = raise NA4
   |dropout (Input as tensor_cons (ID, Parents, Oper, InputDim), KeepRate) = 
        case (KeepRate > 0.0) of
            true => (case (KeepRate < 1.0) of
                     true => tensor_c (c(Input, nil), dropout_c (KeepRate), InputDim)
                    |false => raise NA3)
           |false => raise NA3
                    
fun add (Input1 as tensor_nil, Input2) = raise NA4
   |add (Input1, Input2 as tensor_nil) = raise NA4
   |add (Input1 as tensor_cons (ID1, Parents1, Oper1, InputDim1 as dim_1(n1)), 
         Input2 as tensor_cons (ID2, Parents2, Oper2, InputDim2 as dim_1(n2))) =
        (case (n1 <> n2) of
            true => raise NA2
           |false => tensor_c (c(Input1, c(Input2, nil)), add_c, dim_1(n1)))
   |add (Input1, Input2)  = raise NA1

fun substract (Input1 as tensor_nil, Input2) = raise NA4
   |substract (Input1, Input2 as tensor_nil) = raise NA4
   |substract (Input1 as tensor_cons (ID1, Parents1, Oper1, InputDim1 as dim_1(n1)), 
               Input2 as tensor_cons (ID2, Parents2, Oper2, InputDim2 as dim_1(n2))) =
        (case (n1 <> n2) of 
            true => raise NA2
           |false => tensor_c(c(Input1, c(Input2, nil)), substract_c, dim_1(n1)))
   |substract (Input1, Input2)  = raise NA1
       
fun multiply (Input1 as tensor_nil, Input2) = raise NA4
   |multiply (Input1, Input2 as tensor_nil) = raise NA4
   |multiply (Input1 as tensor_cons (ID1, Parents1, Oper1, InputDim1 as dim_1(n1)), 
              Input2 as tensor_cons (ID2, Parents2, Oper2, InputDim2 as dim_1(n2))) =
        (case (n1 <> n2) of
            true => raise NA2
           |false => tensor_c (c(Input1, c(Input2, nil)), multiply_c, dim_1(n1)))
   |multiply (Input1, Input2)  = raise NA1
fun averageOutput (nil) = raise NA4
   |averageOutput (TensorList as c(T as tensor_cons(ID, Parents, Oper, dim_1(n)), Ts)) 
                        = tensor_c(TensorList, averageOutput_c, dim_1(n))
   |averageOutput (TensorList) = raise NA1


fun f (a: input_tensor) = 
    let
        val (x1, x2)= split(a, 0.5);
        val x3 = head(x1, 2.0);
        val x4 = tail(x2, 2.0);
        val x5 = add(x3, x4);
        val x6 = multiply( x1, x2)
        val x7 = fullyConnect(x3, 2.0)
        val x8 = relu(x7);
        val x9 = concat(x5, x6);
        val x10 = dropout(x8, 0.5)
        val x11 = sigmoid(x10)
        val y1 = toOutput(x11);
        val y2 = toOutput(x9);
        val y3 = toOutput(x1);
        val y = averageOutput(c(y1, c(y2, c(y3, nil))));
    in
        y
    end

(*val InputTensor = tensor_c(nil, placeholder_c, dim_1(20))
  f(InputTensor)
  %%
  fun main () = 
  *)

(* Translate tensor datatype to string *)
open Array
                
fun convertTensorToTf ( FinalTensor as tensor_cons(ID, Parents, Oper, dim_1(n))) = 
    let
        val tfCommands = array(ID+1, "")
        fun println (idx, str) = (print (Int.toString(idx) ^ ": " ^ str); print "\n")
        fun tfCommandMock (ID, Oper, n, Parents as nil) = 
              (case Oper of
                 placeholder_c => update(tfCommands, ID, "placeholder_c dim_1 " ^ Int.toString(n))
               | _ => raise NotImplemented) (*Only placeholder has no parents*)
           |tfCommandMock (ID, Oper, n , Parents as c(tensor_cons(ParentID, _, _, _), nil)) =
              ( case Oper of
                  fullyConnect_c => update(tfCommands, ID, "fullyConnect_c dim_1: " ^ Int.toString(n) ^ "; ParentID: " ^ Int.toString(ParentID))
                 |softmax_c      => update (tfCommands, ID, "softmax_c dim_1: " ^ Int.toString(n) ^ "; ParentID: " ^ Int.toString(ParentID))
                 |head_c         => update (tfCommands, ID, "head_c dim_1: " ^ Int.toString(n) ^ "; ParentID: " ^ Int.toString(ParentID))
                 |tail_c         => update (tfCommands, ID, "tail_c dim_1: " ^ Int.toString(n) ^ "; ParentID: " ^ Int.toString(ParentID))
                 |splitR_c       => update (tfCommands, ID, "splitR_c dim_1: " ^ Int.toString(n) ^ "; ParentID: " ^ Int.toString(ParentID))
                 |splitL_c       => update (tfCommands, ID, "splitL_c dim_1: " ^ Int.toString(n) ^ "; ParentID: " ^ Int.toString(ParentID))
                 |relu_c         => update (tfCommands, ID, "relu_c dim_1: " ^ Int.toString(n) ^ "; ParentID: " ^ Int.toString(ParentID))
                 |tanh_c         => update (tfCommands, ID, "tanh_c dim_1: " ^ Int.toString(n) ^ "; ParentID: " ^ Int.toString(ParentID))
                 |sigmoid_c      => update (tfCommands, ID, "sigmoid_c dim_1: " ^ Int.toString(n) ^ "; ParentID: " ^ Int.toString(ParentID))
                 |sqrt_c         => update (tfCommands, ID, "sqrt_c dim_1: " ^ Int.toString(n) ^ "; ParentID: " ^ Int.toString(ParentID))
                 |dropout_c(r)   => update (tfCommands, ID, "dropout_c rate: " ^ Real.toString(r) ^ " dim_1" ^ Int.toString(n) ^ "; ParentID: " ^ Int.toString(ParentID))
                 |_              => raise NotImplemented (*Only these operators have 1 parent tensor *)
                )
           |tfCommandMock (ID, Oper, n, Parents as c(tensor_cons(ParentID1, _, _, _), c(tensor_cons(ParentID2, _, _, _), nil))) = 
              (case Oper of
                   concat_c      => update (tfCommands, ID, "concat_c dim_1: " ^ Int.toString(n) ^ "; ParentID: " ^ Int.toString(ParentID1) ^ ", " ^ Int.toString(ParentID2))
                  |add_c         => update (tfCommands, ID, "add_c dim_1: " ^ Int.toString(n) ^ "; ParentID: " ^ Int.toString(ParentID1) ^ ", " ^ Int.toString(ParentID2))
                  |multiply_c    => update (tfCommands, ID, "multiply_c dim_1: " ^ Int.toString(n) ^ "; ParentID: " ^ Int.toString(ParentID1) ^ ", " ^ Int.toString(ParentID2))
                  |substract_c   => update (tfCommands, ID, "substract_c dim_1: " ^ Int.toString(n) ^ "; ParentID: " ^ Int.toString(ParentID1) ^ ", " ^ Int.toString(ParentID2))
                  | _            => raise NotImplemented (*Only these operators have 2 parent tensors*)   
              )
           |tfCommandMock (ID, Oper, n, Parents) =
              (case Oper of
                  averageOutput_c => update (tfCommands, ID, "averageOutput_c dim_1: " ^ Int.toString(n) )
                 | _              => raise NotImplemented) (*Only these operators have a list of parents*)
        fun tensorName (ID) = "tensor"^Int.toString(ID)
        fun weightName (ID) = "W" ^ Int.toString(ID)
        fun biasName (ID) = "B" ^ Int.toString(ID)
        fun str(n) = Int.toString(n)
        fun len(nil) = 0
           |len(c(Tensor, TensorList)) = 1+len(TensorList)
          
        fun listTensor(TensorList) =
            let
                fun listT(nil) = "]"
                   |listT(c(Tensor as tensor_cons(ID, _, _, _), TensorList)) = tensorName(ID) ^ ","^listT(TensorList)
            in
                "[" ^ listT(TensorList)
            end
        fun tfCommand (ID, Oper, n, Parents as nil) = 
              (case Oper of
                 placeholder_c => update(tfCommands, ID, tensorName(ID) ^ "= tf.placeholder(tf.float32, shape=(None, "^str(n)^"))")
               | _ => raise NotImplemented) (*Only placeholder has no parents*)
           |tfCommand (ID, Oper, n , Parents as c(tensor_cons(ParentID, _, _, dim_1(nParent)), nil)) =
              ( case Oper of
                  fullyConnect_c => update(tfCommands, ID, 
                    weightName(ID) ^ "= tf.Variable (tf.truncated_normal([" ^ str(nParent) ^ ", " ^ str(n) ^"]), stddev = 1.0/math.sqrt(float(" ^ str(nParent)^ "))) \n" ^ 
                    biasName(ID) ^ "= tf.Variable(tf.zeros(["^str(n)^"])) \n" ^
                    tensorName(ID) ^ "= tf.matmul(" ^ tensorName(ParentID)^", "^weightName(ID)^") + "^ biasName(ID))
                 |softmax_c      => update (tfCommands, ID, 
                    tensorName(ID) ^ "= tf.nn.softmax(" ^ tensorName(ParentID) ^")")
                 |head_c         => update (tfCommands, ID, 
                    tensorName(ID) ^ ", _= tf.split("^tensorName(ParentID)^", ["^str(n)^", "^str(nParent-n)^"], axis=1)")
                 |tail_c         => update (tfCommands, ID, 
                    "_, "^tensorName(ID)^ "= tf.split("^tensorName(ParentID)^", ["^str(nParent-n)^", "^str(n)^"], axis=1)")
                 |splitR_c       => update (tfCommands, ID, 
                    tensorName(ID-1) ^ ", " ^ tensorName(ID) ^ "= tf.split("^tensorName(ParentID)^", ["^str(nParent-n)^", "^str(n)^"], axis=1)")
                 |splitL_c       => update (tfCommands, ID, " ")
                 |relu_c         => update (tfCommands, ID, 
                    tensorName(ID) ^ "= tf.nn.relu("^tensorName(ParentID)^")")
                 |tanh_c         => update (tfCommands, ID, 
                    tensorName(ID) ^ "= tf.tanh("^tensorName(ParentID)^")")
                 |sigmoid_c      => update (tfCommands, ID, 
                    tensorName(ID) ^ "= tf.sigmoid("^tensorName(ParentID)^")")
                 |sqrt_c         => update (tfCommands, ID, 
                    tensorName(ID) ^ "= tf.sqrt("^tensorName(ParentID)^")")
                 |dropout_c(r)   => update (tfCommands, ID, 
                    tensorName(ID) ^ "= tf.nn.dropout("^tensorName(ParentID)^", keep_prob = "^Real.toString(r)^")")
                 |_              => raise NotImplemented (*Only these operators have 1 parent tensor *)
                )
           |tfCommand (ID, Oper, n , Parents as c(tensor_cons(ParentID, _, _, OutDim), nil)) = (raise NotImplemented)
           |tfCommand (ID, Oper, n, Parents as c(tensor_cons(ParentID1, _, _, _), c(tensor_cons(ParentID2, _, _, _), nil))) = 
              (case Oper of
                   concat_c      => update (tfCommands, ID, 
                   tensorName(ID) ^ " = tf.concat ([" ^tensorName(ParentID1)^ ", " ^ tensorName(ParentID2) ^"], axis = 0)")
                  |add_c         => update (tfCommands, ID, 
                   tensorName(ID) ^ " = tf.add("^tensorName(ParentID1)^", "^tensorName(ParentID2)^")")
                  |multiply_c    => update (tfCommands, ID, 
                   tensorName(ID) ^ " = tf.multiply("^tensorName(ParentID1)^", "^tensorName(ParentID2)^")")
                  |substract_c   => update (tfCommands, ID, 
                   tensorName(ID) ^ " = tf.substract("^tensorName(ParentID1)^", "^tensorName(ParentID2)^")")
                  | _            => raise NotImplemented (*Only these operators have 2 parent tensors*)   
              )
           | tfCommand (ID, Oper, n, Parents) =
              (case Oper of
                  averageOutput_c => update (tfCommands, ID, 
                  tensorName(ID) ^ " = tf.divide(tf.add_n("^listTensor(Parents)^"), " ^ str(len(Parents)) ^ ")")
                 | _              => raise NotImplemented) (*Only these operators have a list of parents*)
                 
        
        fun tensorListToTfCommands (TensorList as nil) = ()
           |tensorListToTfCommands (TensorList as c(T, Ts)) = 
                (tensorToTfCommand(T); tensorListToTfCommands(Ts))
        and tensorToTfCommand (Tensor as tensor_cons(ID, Parents, Oper, dim_1(n))) =
                ( case String.compare(sub(tfCommands, ID), "") of
                      EQUAL => (tfCommand(ID, Oper, n, Parents); tensorListToTfCommands(Parents))
                     |_  => () )
           |tensorToTfCommand (_) =  raise NotImplemented (*Do not support dim > 1 *)
            
    in
        (tensorToTfCommand(FinalTensor); appi println tfCommands)
    end
 | convertTensorToTf ( _ ) = raise NotImplemented; (* Do not support dim > 1 now *)

resetId();
val inputTensor = tensor_c(nil, placeholder_c, dim_1(20));
val outTensor = f(inputTensor);
convertTensorToTf(outTensor);