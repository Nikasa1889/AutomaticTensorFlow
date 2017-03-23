(* Author: Dang Ha The Hien 
   Date:   13/03/2017
   Last Modified: 13/03/2017
   Define a set of datatype that allows ADATE to generate an arbitrary tensorflow computation graph.
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
    
fun fromInput tensor_nil = raise NA4
  | fromInput (tensor_cons(ID, nil, placeholder_c, InputDim)) = 
        tensor_c(nil, placeholder_c, InputDim)
  | fromInput _ = raise NA1 
   (* input tensor must have operation placeholder and has no dependency *)
    
fun toOutput tensor_nil = raise NA4
  | toOutput (InputTensor as tensor_cons (_, nodes, oper, InputDim)) = 
        (* Specify the required tensor output dimension here, 
        only support 1D output now *)
        case dim_1(10) of OutDim =>
        (* only support classification now *)
            case InputDim of
                dim_1 (n) => tensor_c (c (InputTensor, nil), 
                                        softmax_c (OutDim), OutDim)
               |dim_2 (_) => raise NA1
               |dim_3 (_) => raise NA1
(* Define helper functions for all of the allowed tensor operations here *)
fun fullyConnect (tensor_nil, _) = raise NA4
  | fullyConnect (InputTensor as tensor_cons (_, _, _, dim_2(_)), ScaleFactor) =  
        raise NA1
  | fullyConnect (InputTensor as tensor_cons (_, _, _, dim_3(_)), ScaleFactor) =  
        raise NA1
  | fullyConnect (InputTensor as tensor_cons (_, _, _, dim_1(n)), ScaleFactor) = 
        case dim_1(floor(real(n) * ScaleFactor)) of OutDim =>
            tensor_c (c(InputTensor, nil), fullyConnect_c( OutDim ), OutDim)

fun split (tensor_nil, _) = raise NA4
  | split (InputTensor as tensor_cons (_, _, _, dim_2(_)), SplitFactor) = 
        raise NA1
  | split (InputTensor as tensor_cons (_, _, _, dim_3(_)), SplitFactor) = 
        raise NA1
  | split (InputTensor as tensor_cons (_, _, _, dim_1(n)), SplitFactor) =
        case (SplitFactor <= 0.0 orelse SplitFactor >= 1.0) of
            true => raise NA3
           |false =>
              case dim_1( floor(real(n) * SplitFactor)) of OutDim1 =>
              case dim_1( n - floor(real(n) * SplitFactor)) of OutDim2 =>
               ( tensor_c (c (InputTensor, nil), splitR_c ( OutDim1 ), OutDim1),
                 tensor_c (c (InputTensor, nil), splitL_c ( OutDim2 ), OutDim2))


fun head (tensor_nil, _) = raise NA4
  | head (InputTensor as tensor_cons (_, _, _, dim_2(_)), nElems)=
        raise NA1
  | head (InputTensor as tensor_cons (_, _, _, dim_3(_)), nElems) = 
        raise NA1
  | head (InputTensor as tensor_cons (_, _, _, dim_1(n)), nElems) =
        case (nElems <= 0.0 orelse floor(nElems) >= n) of
            true => raise NA3
           |false => 
                case dim_1( floor(nElems)) of OutDim =>
                    tensor_c (c (InputTensor, nil), head_c ( OutDim ), OutDim)


fun tail (tensor_nil, _) = raise NA4
  | tail (InputTensor as tensor_cons (_, _, _, dim_2(_)), nElems) = raise NA1
  | tail (InputTensor as tensor_cons (_, _, _, dim_3(_)), nElems) = raise NA1
  | tail (InputTensor as tensor_cons (_, _, _, dim_1(n)), nElems) =
        case (nElems <= 0.0 orelse floor(nElems) >= n) of
            true => raise NA3
           |false =>
                case dim_1( floor(nElems)) of OutDim =>
                    tensor_c (c (InputTensor, nil), tail_c ( OutDim ), OutDim)
   
   
fun concat (tensor_nil, _) = raise NA4
   |concat (_, tensor_nil) = raise NA4
   |concat (Input1 as tensor_cons (_, _, _, dim_1(n1)), 
            Input2 as tensor_cons (_, _, _, dim_1(n2))) = 
        (case dim_1(n1 + n2) of OutDim =>
            tensor_c (c(Input1, c(Input2, nil)), concat_c, OutDim))
   |concat (_, _) = raise NA4
                
fun relu (tensor_nil) = raise NA4
   |relu (InputTensor as tensor_cons (_, _, _, InputDim)) = 
        tensor_c (c(InputTensor, nil), relu_c, InputDim)
        
fun sigmoid (tensor_nil) = raise NA4
   |sigmoid (InputTensor as tensor_cons (_, _, _, InputDim)) = 
        tensor_c (c(InputTensor, nil), sigmoid_c, InputDim)
           
fun tanh (tensor_nil) = raise NA4
   |tanh (InputTensor as tensor_cons (_, _, _, InputDim)) = 
        tensor_c (c(InputTensor, nil), tanh_c, InputDim)
        
fun sqrt (tensor_nil) = raise NA4
   |sqrt (InputTensor as tensor_cons (_, _, _, InputDim)) = 
        tensor_c (c(InputTensor, nil), sqrt_c, InputDim)
        
fun dropout (tensor_nil, _) = raise NA4
   |dropout (InputTensor as tensor_cons (_, _, _, InputDim), keepRate) = 
        case ((keepRate <=0.0) orelse (keepRate > 1.0)) of
            true => raise NA3
           |false => tensor_c (c(InputTensor, nil), dropout_c (keepRate), InputDim)
                    
fun add (tensor_nil, _) = raise NA4
   |add (_, tensor_nil) = raise NA4
   |add (Input1 as tensor_cons (_, _, _, dim_1(n1)), 
         Input2 as tensor_cons (_, _, _, dim_1(n2))) =
        (case (n1 <> n2) of
            true => raise NA2
           |false => tensor_c (c(Input1, c(Input2, nil)), add_c, dim_1(n1)))
   |add (_, _)  = raise NA1

fun substract (tensor_nil, _) = raise NA4
   |substract (_, tensor_nil) = raise NA4
   |substract (Input1 as tensor_cons (_, _, _, dim_1(n1)), 
               Input2 as tensor_cons (_, _, _, dim_1(n2))) =
        (case (n1 <> n2) of 
            true => raise NA2
           |false => tensor_c(c(Input1, c(Input2, nil)), substract_c, dim_1(n1)))
   |substract (_, _)  = raise NA1
       
fun multiply (tensor_nil, _) = raise NA4
   |multiply (_, tensor_nil) = raise NA4
   |multiply (Input1 as tensor_cons (_, _, _, dim_1(n1)), 
              Input2 as tensor_cons (_, _, _, dim_1(n2))) =
        (case (n1 <> n2) of
            true => raise NA2
           |false => tensor_c (c(Input1, c(Input2, nil)), multiply_c, dim_1(n1)))
   |multiply (_, _)  = raise NA1

fun f (a: input_tensor) = 
    toOutput(fromInput(a))

(*val InputTensor = tensor_c(nil, placeholder_c, dim_1(20))
  f(InputTensor)
  %%
  fun main () = 
  *)

