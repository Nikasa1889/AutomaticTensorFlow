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
datatype tensor_list = nil | c of tensor * tensor_list
type input_tensor = tensor
type output_tensor = tensor

fun f (a: input_tensor) = 
    toOutput(fromInput(a))

datatype dim = dim_1 of int | dim_2 of int * int | dim_3 of int * int * int
datatype operation = placeholder_c
    `               | fullyConnect_c of dim
                    | softmax_c of dim
                    (*| conv2d_c of  (int * int) * int * int (* kernel_size * strides * padding *)
                    | max_pool_c of (int * int) * int * int (* kernel_size * strides * padding *) *)
                    | split of dim (* percentage of the first *)
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

datatype tensor = tensor_nil | tensor_cons of int * tensor_list * operation * dim

val tensor_id = ref 0
fun generateId () =
    (tensor_id := !tensor_id + 1;!tensor_id);

fun tensor_c(tensor_list, op, inputDim) = 
    tensor_cons(generateId(), tensor_list, op, inputDim)
    
fun fromInput tensor_nil = raise TensorNil
   |fromInput (tensor_nil, placeholder, input_dim) = tensor_c(tensor_nil, placeholder, input_dim)
   |fromInput _ = raise NotImplemented (* input tensor must have operation placeholder and has no dependency *)
    
fun toOutput tensor_nil = raise TensorNil
   |toOutput inputTensor as tensor_cons (nodes, op, inputDim) = 
   let 
        val outputDim = 10 (* Specify the required tensor output dimension here, only support 1D output now *)
   in
        case inputDim of
            dim_1 (n) = tensor_c (c (inputTensor, nil), softmax_c (outputDim), dim_1 (outputDim)) (* only for classification now *)
           |dim_2 (n) = raise NotImplemented
           |dim_3 (n) = raise NotImplemented
   end
(* Define helper functions for all of the allowed tensor operations here *)
fun fullyConnect (tensor_nil, _) = raise TensorNil
                |(inputTensor as tensor_cons (_, _, dim_1(n)), scaleFactor) = 
                    let 
                        val outDim = dim_1(floor(n * scaleFactor))
                    in
                        tensor_c (c(inputTensor, nil), fully_connect( outDim ), outDim)
                    end
                | (inputTensor as tensor_cons (_, _, dim_2(nrows, ncolumns)), scaleFactor) = 
                    raise NotImplemented
                | (inputTensor as tensor_cons (_, _, dim_3(nrows, ncolumns, ndepths), scaleFactor) =
                    raise NotImplemented

fun split (tensor_nil, _) = raise TensorNil
          |(inputTensor as tensor_cons (_, _, _, dim_1(n)), splitFactor) =
                if (splitFactor <=0) or (splitFactor >= 1) then raise ArgumentOutOfRange
                else 
                    let
                        val outDim1 = dim_1( floor(n * splitFactor))
                        val outDim2 = dim_1( n - floor(n * splitFactor))
                    in
                       ( tensor_c (c (inputTensor, nil), split_c ( outDim1 ), outDim1),
                         tensor_c (c (inputTensor, nil), split_c ( outDim2 ), outDim2))
                    end
          | (inputTensor as tensor_cons (_, _, dim_2(nrows, ncolumns)), splitFactor): raise NotImplemented
          | (inputTensor as tensor_cons (_, _, dim_3(nrows, ncolumns, ndepths)), splitFactor): raise NotImplemented
          
fun concat (tensor_nil, _) = raise TensorNil
            |(_, tensor_nil) = raise TensorNil
            |(input1 as tensor_cons (_, _, dim_1(n1)), input2 as (_, _, dim_2(n2))) = 
                let
                    val outDim = dim_1(n1 + n2)
                in
                    tensor_c (c(input1, c(input2, nil)), concat_c, outDim)

fun relu (tensor_nil) = raise TensorNil
        |(inputTensor as tensor_cons (_, _, inputDim)) = tensor_c (c(inputTensor, c), relu_c, inputDim)
        
fun sigmoid (tensor_nil) = raise TensorNil
           |(inputTensor as tensor_cons (_, _, inputDim)) = tensor_c (c(inputTensor, c), sigmoid_c, inputDim)
           
fun tanh (tensor_nil) = raise TensorNil
        |(inputTensor as tensor_cons (_, _, inputDim)) = tensor_c (c(inputTensor, c), tanh_c, inputDim)
        
fun sqrt (tensor_nil) = raise TensorNil
        |(inputTensor as tensor_cons (_, _, inputDim)) = tensor_c (c(inputTensor, c), sqrt_c, inputDim)
        
        
fun dropout (tensor_nil) = raise TensorNil
            |(inputTensor as tensor_cons (_, _, inputDim), keepRate) = 
                if (keepRate <=0) or (keepRate > 1) then raise ArgumentOutOfRange
                else
                    tensor_c (c(inputTensor, c), dropout_c (keepRate), inputDim)
                    
fun add (tensor_nil, _) = raise TensorNil
       |(_, tensor_nil) = raise TensorNil
       |(input1 as tensor_cons (_, _, dim_1(n1)), input2 as tensor_cons (_, _, dim_1(n2))) =
            if (n1 <> n2) raise UnmatchedDimension
            else tensor_cons (c(input1, c(input2, nil)), add_c, dim_1(n1))
       |(_, _)  = raise NotImplemented

fun substract (tensor_nil, _) = raise TensorNil
             |(_, tensor_nil) = raise TensorNil
             |(input1 as tensor_cons (_, _, dim_1(n1)), input2 as tensor_cons (_, _, dim_1(n2))) =
                if (n1 <> n2) raise UnmatchedDimension
                else tensor_cons (c(input1, c(input2, nil)), substract_c, dim_1(n1))
             |(_, _)  = raise NotImplemented
       
fun multiply_c (tensor_nil, _) = raise TensorNil
               |(_, tensor_nil) = raise TensorNil
               |(input1 as tensor_cons (_, _, dim_1(n1)), input2 as tensor_cons (_, _, dim_1(n2))) =
            if (n1 <> n2) raise UnmatchedDimension
            else tensor_cons (c(input1, c(input2, nil)), multiply_c, dim_1(n1))
       |(_, _)  = raise NotImplemented