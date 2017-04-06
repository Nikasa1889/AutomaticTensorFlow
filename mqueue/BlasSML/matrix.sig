(*
   File: matrix.sml
   Content: matrix library using C Array and Blas library
   Author: Dang Ha The Hien, Hiof, hdthe@hiof.no
   Convention for variable names:
       matrix:          m,  m1,  m2  ...
       vector:          v,  v1,  v2  ...
	   
*)
signature MATRIX =
sig
  type vector
  type matrix
	datatype isTrans = TRANS|NOTRANS
  exception UnmatchedDimension
  (* basic functions for datatype matrix *)
	val toVector:      matrix -> vector
  val size:          matrix -> int * int
  (*  functions to create new matrix *)
	val newVec:		int -> vector
  val newMat:     int * int -> matrix
	val freeVec:     vector -> unit
	val freeMat:     matrix -> unit
	
  val fromVec:    vector * (int * int) -> matrix
	val fromList2Vec: real list -> vector
  val fromLists: real list list * (int * int) -> matrix
  (*  functions for debuging *)
  val printMat:  matrix -> unit
	val printVec:  vector -> unit
	
	(*  functions for output *)
	(* val printfMatrix: string * matrix -> unit *)

  (*  supported operators on vector *)
	val dset: vector * real -> unit
	val ddot: vector * vector -> real
	val dotmul: vector * vector -> vector
	val dcopy: vector -> vector
	val dscal: real * vector -> unit
	val daxpy: real * vector * vector -> unit
	val dnrm2: vector -> real
	(*val dapp: (real -> real) * vector -> unit*)
	val dappSigm: vector -> unit
	val dappDSigm: vector -> unit
	val dappTanh: vector -> unit
	val dappDTanh: vector -> unit
	val dappExp: vector -> unit
	val dappHSquare: vector -> unit
	val dappMinusLn: vector -> unit
	val dappCe: vector -> unit
	val dappInvr: vector -> unit
	
	val dreprows: vector * int -> matrix
	val drepcols: vector * int -> matrix
	val dsum: vector -> real
	val eqcount: vector * vector -> int
	val eq:   vector * vector -> bool
	(* supported operators on matrix *)
	val dgemm: isTrans * isTrans * real * matrix * matrix * real * matrix -> unit
	val dgemv: isTrans * real * matrix * vector * real * vector -> unit
	val mdcopy: matrix -> matrix
	(*val mdapp: (real -> real) * matrix -> unit*)
	val mdset: matrix * real -> unit
	val mdotmul: matrix * matrix -> matrix
	val mdaxpy: real * matrix * matrix -> unit
	val mdscal: real * matrix -> unit
	val mdscalRows: matrix * vector -> unit
	val sumRows: matrix * vector -> unit
	val sumCols: matrix * vector -> unit
	val sumAll: matrix -> real
	val maxCols: matrix * vector -> unit
	val maxColsIdx: matrix * vector -> unit
	val meq: matrix * matrix -> bool
(*	
	
	

    val mergeVector:     (('a*'a)->'b) -> ('a vector * 'a vector) -> 'b vector
	val mergeVect2Matrix: (('a * 'a) -> 'b) -> ('a vector * 'a matrix) -> 'b matrix 
	val mergeVect'2Matrix: (('a * 'a) -> 'b) -> ('a vector * 'a matrix) -> 'b matrix
	val addVect2MatrixReal: real list * real matrix -> real matrix
    val addVect2MatrixInt: int list * int matrix -> int matrix
*)
(*
    (*  supported operators on matrix *)
    (* scalar operator *)
    val map:               ('a -> 'b) -> 'a matrix -> 'b matrix
    val addScalarInt:      int matrix * int -> int matrix
    val mulScalarInt:      int matrix * int -> int matrix
    val addScalarReal:     real matrix * real -> real matrix
    val mulScalarReal:     real matrix * real -> real matrix
    (* accumulate operator *)
    val foldl:             (('a*'b) -> 'b) -> 'b -> ('a matrix) -> 'b vector
    val sumInt:            (int matrix) -> int vector
    val sumReal:           (real matrix) -> real vector
    (* matrix operator *)
    val merge:             (('a*'b)->'c) -> ('a matrix * 'b matrix) -> 'c matrix
    val dotMulMatrixInt:   int matrix * int matrix -> int matrix
    val dotMulMatrixReal:  real matrix * real matrix -> real matrix
    val addMatrixInt:   int matrix * int matrix -> int matrix
    val addMatrixReal:  real matrix * real matrix -> real matrix
    (* real multiply operation *)
    val mulMatrixIntR:      int matrix * int matrix -> int matrix
    val mulMatrixIntC:      int matrix * int matrix -> int matrix
    val mulMatrixRealR:     real matrix * real matrix -> real matrix
    val mulMatrixRealC:     real matrix * real matrix -> real matrix
    (* sum collumn if it's COLMATRIX, sum row if it's ROWMATRIX *)
    (*val sum:           'a matrix -> 'a vector
    val repeatCol:     'a vector * int -> 'a matrix
    val repeatRow:     'a vector * int -> 'a matrix
    
   *)
*)
end

