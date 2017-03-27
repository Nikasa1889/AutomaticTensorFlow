(* Author: Dang Ha The Hien 
   Date:   13/03/2017
   Last Modified: 13/03/2017
   Define a set of datatype that allows ADATE to generate an arbitrary tensorflow computation graph.
   Note: do not take into account the batch-size dimension of a tensor
*)
exception NotImplemented
exception UnmatchedDimension
exception ArgumentOutOfRange
exception TensorNil
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

datatype dim = dim_1 of int | dim_2 of int * int | dim_3 of int * int * int
datatype operation = placeholder_c
                | fullyConnect_c of dim
                | softmax_c of dim
                (*| conv2d_c of  (int * int) * int * int (* kernel_size * strides * padding *)
                | max_pool_c of (int * int) * int * int (* kernel_size * strides * padding *) *)
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

fun tensor_c (tensors, oper, inputDim) : tensor = tensor_cons (generateId(), tensors, oper, inputDim)
    
fun fromInput tensor_nil = raise TensorNil
   |fromInput (tensor_cons(id, nil, placeholder_c, inputDim)) = tensor_c(nil, placeholder_c, inputDim)
   |fromInput _ = raise NotImplemented (* input tensor must have operation placeholder and has no dependency *)
    
fun toOutput tensor_nil = raise TensorNil
   |toOutput (inputTensor as tensor_cons (_, nodes, oper, inputDim)) = 
   let 
        val outputDim = dim_1(10) (* Specify the required tensor output dimension here, only support 1D output now *)
   in
        case inputDim of
            dim_1 (n) => tensor_c (c (inputTensor, nil), softmax_c (outputDim), outputDim) (* only for classification now *)
           |dim_2 (n) => raise NotImplemented
           |dim_3 (n) => raise NotImplemented
   end
(* Define helper functions for all of the allowed tensor operations here *)
fun fullyConnect (tensor_nil, _) = raise TensorNil
   |fullyConnect (inputTensor as tensor_cons (_, _, _, dim_1(n)), scaleFactor) = 
                    let 
                        val outDim = dim_1(floor(real(n) * scaleFactor))
                    in
                        tensor_c (c(inputTensor, nil), fullyConnect_c( outDim ), outDim)
                    end
   |fullyConnect (inputTensor as tensor_cons (_, _, _, dim_2(nrows, ncolumns)), scaleFactor) = 
                    raise NotImplemented
   |fullyConnect (inputTensor as tensor_cons (_, _, _, dim_3(nrows, ncolumns, ndepths)), scaleFactor) =
                    raise NotImplemented

fun split (tensor_nil, _) = raise TensorNil
   |split (inputTensor as tensor_cons (_, _, _, dim_1(n)), splitFactor) =
                if (splitFactor <= 0.0 orelse splitFactor >= 1.0) then raise ArgumentOutOfRange
                else 
                    let
                        val outDim1 = dim_1( floor(real(n) * splitFactor))
                        val outDim2 = dim_1( n - floor(real(n) * splitFactor))
                    in
                       ( tensor_c (c (inputTensor, nil), splitR_c ( outDim1 ), outDim1),
                         tensor_c (c (inputTensor, nil), splitL_c ( outDim2 ), outDim2))
                    end
   |split (inputTensor as tensor_cons (_, _, _, dim_2(nrows, ncolumns)), splitFactor) = raise NotImplemented
   |split (inputTensor as tensor_cons (_, _, _, dim_3(nrows, ncolumns, ndepths)), splitFactor) = raise NotImplemented


fun head (tensor_nil, _) = raise TensorNil
   |head (inputTensor as tensor_cons (_, _, _, dim_1(n)), nElems) =
                if (nElems <= 0.0 orelse floor(nElems) >= n) then raise ArgumentOutOfRange
                else 
                    let
                        val outDim = dim_1( floor(nElems))
                    in
                        tensor_c (c (inputTensor, nil), head_c ( outDim ), outDim)
                    end
   |head (inputTensor as tensor_cons (_, _, _, dim_2(nrows, ncolumns)), nElems) = raise NotImplemented
   |head (inputTensor as tensor_cons (_, _, _, dim_3(nrows, ncolumns, ndepths)), nElems) = raise NotImplemented

fun tail (tensor_nil, _) = raise TensorNil
   |tail (inputTensor as tensor_cons (_, _, _, dim_1(n)), nElems) =
                if (nElems <= 0.0 orelse floor(nElems) >= n) then raise ArgumentOutOfRange
                else 
                    let
                        val outDim = dim_1( floor(nElems))
                    in
                        tensor_c (c (inputTensor, nil), tail_c ( outDim ), outDim)
                    end
   |tail (inputTensor as tensor_cons (_, _, _, dim_2(nrows, ncolumns)), nElems) = raise NotImplemented
   |tail (inputTensor as tensor_cons (_, _, _, dim_3(nrows, ncolumns, ndepths)), nElems) = raise NotImplemented
   
fun concat (tensor_nil, _) = raise TensorNil
   |concat (_, tensor_nil) = raise TensorNil
   |concat (input1 as tensor_cons (_, _, _, dim_1(n1)), input2 as tensor_cons (_, _, _, dim_1(n2))) = 
                let
                    val outDim = dim_1(n1 + n2)
                in
                    tensor_c (c(input1, c(input2, nil)), concat_c, outDim)
                end
   |concat (_, _) = raise TensorNil
                
fun relu (tensor_nil) = raise TensorNil
   |relu (inputTensor as tensor_cons (_, _, _, inputDim)) = tensor_c (c(inputTensor, nil), relu_c, inputDim)
        
fun sigmoid (tensor_nil) = raise TensorNil
   |sigmoid (inputTensor as tensor_cons (_, _, _, inputDim)) = tensor_c (c(inputTensor, nil), sigmoid_c, inputDim)
           
fun tanh (tensor_nil) = raise TensorNil
   |tanh (inputTensor as tensor_cons (_, _, _, inputDim)) = tensor_c (c(inputTensor, nil), tanh_c, inputDim)
        
fun sqrt (tensor_nil) = raise TensorNil
   |sqrt (inputTensor as tensor_cons (_, _, _, inputDim)) = tensor_c (c(inputTensor, nil), sqrt_c, inputDim)
        
        
fun dropout (tensor_nil, _) = raise TensorNil
   |dropout (inputTensor as tensor_cons (_, _, _, inputDim), keepRate) = 
                if ((keepRate <=0.0) orelse (keepRate > 1.0))
                then 
                    raise ArgumentOutOfRange
                else
                    tensor_c (c(inputTensor, nil), dropout_c (keepRate), inputDim)
                    
fun add (tensor_nil, _) = raise TensorNil
   |add (_, tensor_nil) = raise TensorNil
   |add (input1 as tensor_cons (_, _, _, dim_1(n1)), input2 as tensor_cons (_, _, _, dim_1(n2))) =
            if (n1 <> n2) 
            then raise UnmatchedDimension
            else tensor_c (c(input1, c(input2, nil)), add_c, dim_1(n1))
   |add (_, _)  = raise NotImplemented

fun substract (tensor_nil, _) = raise TensorNil
   |substract (_, tensor_nil) = raise TensorNil
   |substract (input1 as tensor_cons (_, _, _, dim_1(n1)), input2 as tensor_cons (_, _, _, dim_1(n2))) =
                if (n1 <> n2) 
                then raise UnmatchedDimension
                else tensor_c(c(input1, c(input2, nil)), substract_c, dim_1(n1))
   |substract (_, _)  = raise NotImplemented
       
fun multiply (tensor_nil, _) = raise TensorNil
   |multiply (_, tensor_nil) = raise TensorNil
   |multiply (input1 as tensor_cons (_, _, _, dim_1(n1)), input2 as tensor_cons (_, _, _, dim_1(n2))) =
            if (n1 <> n2) 
            then raise UnmatchedDimension
            else tensor_c (c(input1, c(input2, nil)), multiply_c, dim_1(n1))
   |multiply (_, _)  = raise NotImplemented

fun averageOutput (nil) = raise TensorNil
   |averageOutput (TensorList as c(T as tensor_cons(ID, Parents, Oper, dim_1(n)), Ts)) 
                        = tensor_c(TensorList, averageOutput_c, dim_1(n))
   |averageOutput (TensorList) = raise NotImplemented
   
fun f (a: input_tensor) = 
    toOutput(fromInput(a))

val inputTensor = tensor_c(nil, placeholder_c, dim_1(20))
f(inputTensor)
(* Translate tensor datatype to string *)
open Array
fun convertTensorToTf ( FinalTensor as tensor_c(ID, Parents, Oper, dim_1(n)) = 
    let
        
    in
    end
    
