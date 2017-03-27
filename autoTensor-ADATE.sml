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
end

datatype dim = dim_1 of int | dim_2 of int * int | dim_3 of int * int * int
datatype operation = placeholder_c
                | fullyConnect_c of dim
                | softmax_c of dim
                (*| conv2d_c of  (int * int) * int * int 
                        (* kernel_size * strides * padding *)
                | max_pool_c of (int * int) * int * int 
                        (* kernel_size * strides * padding *) *)
                | head_c of dim
                | tail_c of dim
                | splitR_c of dim (* percentage of the first *)
                | splitL_c of dim 
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
                (*| maximum_c*)
datatype tensor_list = nil | c of tensor * tensor_list
and tensor = tensor_nil | tensor_cons of int * tensor_list * operation * dim
type input_tensor = tensor
type output_tensor = tensor

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
                                        softmax_c (OutDim), OutDim)
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
            tensor_c (c(Input, nil), fullyConnect_c( OutDim ), OutDim)

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
                      (tensor_c (c (Input, nil), splitR_c ( OutDim1 ), OutDim1),
                       tensor_c (c (Input, nil), splitL_c ( OutDim2 ), OutDim2)))
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
                        tensor_c (c (Input, nil), head_c ( OutDim ), OutDim))
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
                        tensor_c (c (Input, nil), tail_c ( OutDim ), OutDim))
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

fun f (a: input_tensor) = 
    toOutput(fromInput(a))

(*val InputTensor = tensor_c(nil, placeholder_c, dim_1(20))
  f(InputTensor)
  %%
  fun main () = 
  *)

