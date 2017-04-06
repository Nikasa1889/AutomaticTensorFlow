(* This array is Row-major order to be consistent with C array*)
structure Matrix :> MATRIX =
struct
	type CArray = MLton.Pointer.t
    type vector = CArray * int       (* array * length *)
	type matrix = CArray * int * int (* array * nrows * ncols *)
	datatype isTrans = TRANS|NOTRANS
    exception UnmatchedDimension
	
	val createCArr = _import "createCArr" public: int -> CArray;
	val toCArr = _import "toCArr" public: real array * int -> CArray;
	val printMat = _import "printMat" public: CArray * int * int -> unit;
	val printVec = _import "printVec" public: CArray * int -> unit;
	
	val blas_ddot = _import "blas_ddot" public: int * CArray * CArray -> real;
	val blas_dotmul = _import "blas_dotmul" public: int * CArray * CArray * CArray-> unit;
	val blas_dcopy = _import "blas_dcopy" public: int * CArray * CArray -> unit;
	val blas_dscal = _import "blas_dscal" public: int * real * CArray -> unit;
	val blas_axpy = _import "blas_axpy" public: int * real * CArray * CArray -> unit;
	val blas_dnrm2 = _import "blas_dnrm2" public: int * CArray -> real;
	val blas_reprows = _import "blas_reprows" public: int * CArray * int * CArray -> unit;
	val blas_repcols = _import "blas_repcols" public: int * CArray * int * CArray -> unit;
	
	val blas_dgemm = _import "blas_dgemm" public: int * int * int * int * int * real * CArray * int * CArray * int * real * CArray * int -> unit;
	val blas_dgemv = _import "blas_dgemv" public: int * int * int * real * CArray * CArray * real * CArray -> unit;
	
	
	(*val c_dapp = _import "c_dapp" public: CArray * int -> unit;*)
	val c_dset = _import "c_dset" public: CArray * int * real -> unit;
	val c_free = _import "free" public: CArray -> unit;
	val c_sumRows = _import "c_sumRows" public: CArray * int * int * CArray -> unit;
	val c_sumCols = _import "c_sumCols" public: CArray * int * int * CArray -> unit;
	val c_sum = _import "c_sum" public: CArray * int -> real;
	val c_maxCols = _import "c_maxCols" public: CArray * int * int * CArray -> unit;
	val c_mdscalRows = _import "c_mdscalRows" public: CArray * int * int * CArray -> unit;
	val c_maxColsIdx = _import "c_maxColsIdx" public: CArray * int * int * CArray -> unit;
	val c_eqcount = _import "c_eqcount" public: int * CArray * CArray -> int;
	val c_eq      = _import "c_eq" public: int * CArray * CArray -> int;
	
	val dappSigm = _import "dappSigm" public: CArray * int -> unit;
	val dappTanh = _import "dappTanh" public: CArray * int -> unit;
	val dappDTanh = _import "dappDTanh" public: CArray * int -> unit;
	val dappDSigm = _import "dappDSigm" public: CArray * int -> unit;
	val dappExp = _import "dappExp" public: CArray * int -> unit;
	val dappDSigm = _import "dappDSigm" public: CArray * int -> unit;
	val dappHSquare = _import "dappHSquare" public: CArray * int -> unit;
	val dappMinusLn = _import "dappMinusLn" public: CArray * int -> unit;
	val dappCe = _import "dappCe" public: CArray * int -> unit;
	val dappInvr = _import "dappInvr" public: CArray * int -> unit;
	
	
	fun toVector (p, m, n) = (p, m*n)
	fun size (p, m, n) = (m, n)
	
	fun newVec l = (createCArr (l), ((*print ("NewVec-"^Int.toString(l));*) l))
	fun newMat (m, n) = (createCArr (m*n), m, ((*print ("NewMat-"^Int.toString(m*n));*)n))
	fun fromVec ((v, l), (m, n)) = if (l = m*n) then (v, m, n) else (print ("Error in fromVec()"); raise UnmatchedDimension)
	fun fromList2Vec ls = 
		let val v = Array.fromList (ls)
			val l = Array.length (v)
		in	(toCArr (v, l), l)
		end
	fun fromLists (lss, (m, n)) = fromVec( fromList2Vec( List.concat (lss) ), (m, n))
	
	(* Vector operations *)
	fun dset ((v1, l1), alpha) = 
		c_dset(v1, l1, alpha)
	fun ddot ((v1, l1), (v2, l2)) = 
		if l1 <> l2 then (print ("Error in ddot()"); raise UnmatchedDimension)
		else
			blas_ddot(l1, v1, v2)
	fun dotmul ((v1, l1), (v2, l2)) =
		if l1 <> l2 then (print ("Error in dotmul()"); raise UnmatchedDimension)
		else
		let
			val z = newVec(l1)
			val _ = blas_dotmul(l1, v1, v2, #1 z)
		in
			z
		end
	(*fun dapp (f, (v1, l1)) = 
	let
		val e = _export "applyVecf": (real -> real) -> unit;
		val _ = e (f);
	in
		c_dapp (v1, l1)
	end*)
	fun dcopy (v1, l1) =
	let
		val y = newVec (l1)
		val _ = blas_dcopy(l1, v1, #1 y)
	in
		y
	end
	fun dscal (alpha, (v1, l1)) = blas_dscal (l1, alpha, v1)
	fun daxpy (alpha, (v1, l1), (v2, l2)) = 
		if l1 <> l2 then (print ("Error in daxpy()"); raise UnmatchedDimension)
		else
			blas_axpy(l1, alpha, v1, v2)
	fun dnrm2 (v1, l1) =
		blas_dnrm2(l1, v1)
	fun dreprows ((v1, l1), m) =
	let
		val X = newMat (m, l1)
		val _ = blas_reprows(l1, v1, m, #1 X)
	in
		X
	end
	fun drepcols ((v1, l1), n) = 
	let
		val X = newMat (l1, n)
		val _ = blas_repcols (l1, v1, n, #1 X)
	in
		X
	end
	fun dsum (v1, l1) =
		c_sum (v1, l1)
	fun eqcount ((v1, l1), (v2, l2)) =
		if l1 <> l2 then (print ("Error in eqcount()"); raise UnmatchedDimension)
		else
			c_eqcount(l1, v1, v2)
	fun eq ((v1, l1), (v2, l2)) =
		if (l1 <> l2) then (print ("Error in eq()"); raise UnmatchedDimension)
		else
			if (c_eq(l1, v1, v2) = 1) then 
				true 
			else false
	
	fun freeVec (v, l) = ((*print ("freeVec-"^Int.toString(l));*)c_free (v))
	
	fun dgemm (transA, transB, alpha, (A, mA, nA), (B, mB, nB), beta, (C, mC, nC)) =
		let
			val isTransA = if transA = NOTRANS then 0 else 1
			val isTransB = if transB = NOTRANS then 0 else 1
			val m = if transA = NOTRANS then mA else nA
			val n = if transB = NOTRANS then nB else mB
			val k = if transA = NOTRANS then nA else mA
			val lda = nA
			val ldb = nB
			val ldc = nC
		in
			blas_dgemm(isTransA, isTransB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
		end
	
	fun dgemv (transA, alpha, (A, mA, nA), (x, lx), beta, (y, ly)) =
		if (mA <> ly) orelse (nA <> lx) then
			(print ("Error in dgemv()"); raise UnmatchedDimension)
		else
			if transA = NOTRANS then
				blas_dgemv( 0, mA, nA, alpha, A, x, beta, y)
			else
				blas_dgemv( 1, mA, nA, alpha, A, x, beta, y)
	
	
	(*fun mdapp (f, (A, mA, nA)) = dapp (f, (A, mA*nA))*)
	fun mdcopy (A, mA, nA) =
	let
		val B = newMat (mA, nA)
		val _ = blas_dcopy(mA * nA, A, #1 B)
	in
		B
	end
	fun mdset ((A, mA, nA), alpha) = dset ((A, mA*nA), alpha)
	fun mdotmul ((A, mA, nA), (B, mB, nB)) =
		fromVec ((dotmul ((A, mA * nA), (B, mB * nB))), (mA, nA))
	fun mdaxpy (alpha, (X, mX, nX), (Y, mY, nY)) =
		daxpy (alpha, (X, mX * nX), (Y, mY * nY))
	fun mdscal (alpha, (A, mA, nA)) = dscal (alpha, (A, mA * nA))
	fun mdscalRows ((A, mA, nA), (v, l)) = 
		if l <> mA then (print ("Error in mdscalRows()"); raise UnmatchedDimension)
		else c_mdscalRows (A, mA, nA, v)
	
	fun freeMat (A, mA, nA) = ((*print ("freeMat-"^Int.toString(mA*nA));*) c_free(A))
	fun sumRows ((A, mA, nA), (v, l)) =
		if l <> nA then (print ("Error in sumRows()"); raise UnmatchedDimension)
		else c_sumRows (A, mA, nA, v)
	fun sumCols ((A, mA, nA), (v, l)) =
		if l <> mA then (print ("Error in sumCols()"); raise UnmatchedDimension)
		else c_sumCols (A, mA, nA, v)
	
	fun sumAll (A, mA, nA) =
	let
		val temp = newVec (nA)
		val _ = sumRows ((A, mA, nA), temp)
		val result = dsum (temp)
		val _ = freeVec (temp)
	in
		result
	end
	fun maxCols ((A, mA, nA), (v, l)) =
		if l <> mA then (print ("Error in maxCols()"); raise UnmatchedDimension)
		else c_maxCols (A, mA, nA, v)
	
	fun maxColsIdx ((A, mA, nA), (v, l)) =
		if l <> mA then (print ("Error in maxColsIdx()"); raise UnmatchedDimension)
		else c_maxColsIdx (A, mA, nA, v)
	fun meq ((A, mA, nA), (B, mB, nB)) =
		if ((mA <> mB) orelse (nA <> nB)) then (print ("Error in meq()"); raise UnmatchedDimension)
		else eq((A, mA*nA), (B, mB*nB))
	(*
    fun transpose m =
	case m of
	    COLMATRIX (vs, rows, cols) => changeType(ROWMATRIX (vs, cols, rows))
	  | ROWMATRIX (vs, rows, cols) => changeType(COLMATRIX (vs, cols, rows))
	
    fun transpose_changeType m =
	case m of
	    COLMATRIX (vs, rows, cols) => ROWMATRIX (vs, cols, rows)
	  | ROWMATRIX (vs, rows, cols) => COLMATRIX (vs, cols, rows)

    fun	fromVectors2List [] = []
      | fromVectors2List (v::vs) = v @ fromVectors2List(vs)
	
    fun toVector m =
	case m of
	    ROWMATRIX (vs, rows, cols) => fromVectors2List (vs)
	   |COLMATRIX (vs, rows, cols) => fromVectors2List (changeVectors(vs))
	  
    fun sizeVectors vs =
	let
	    val nElems = List.foldl (fn (v, lengthv) => 
					case (lengthv, (List.length(v) = lengthv)) of
					    (0, _)    => List.length(v)
					  | (_, true) => List.length(v)
					  | (_, false) => ~1)
				    0 vs
	    val nVectors = List.length(vs)
	in
	    (nElems, nVectors)
	end

    fun checkValid (COLMATRIX(vs, rows, cols)) = 
	let val (nElems, nVectors) = sizeVectors(vs)
	in if (nElems = rows) andalso (nVectors = cols) then true else false
	end
      | checkValid (ROWMATRIX(vs, rows, cols))  = 
	let val (nElems, nVectors) = sizeVectors(vs)
	in if (nElems = cols) andalso (nVectors = rows)then true else false
	end
	
    fun initVector (value, n, acc) =
	case n of
	    0 => acc
	  | n => initVector(value, n-1, value::acc)

    fun initVectors (value, rows, cols) =
	let
	    fun reduce(cols, acc) =
		case cols of
		    0 => acc
		 | cols => reduce(cols-1, 
					initVector(value, rows, [])::acc)
	in
	    reduce(cols, [])
	end
    fun initCols (value, (rows, cols)) = COLMATRIX (initVectors(value, rows, cols),
						 rows, cols)
    fun initRows (value, (rows, cols)) =  transpose_changeType (initCols(value, (cols, rows)))
	
    fun uniRandVectors RandState (nVectors, length, max) = 
	let
	    fun uniRandList(n) =
		if n = 0 then []
		else (max*(2.0*(Random.randReal RandState)-1.0))::uniRandList(n-1)
	    fun uniRandVectors(nVectors) =
		if nVectors = 0 then []
		else uniRandList(length)::uniRandVectors(nVectors-1)
	in
	    uniRandVectors(nVectors)
	end

    fun uniRandRows RandState (max, (rows, cols)) =
	ROWMATRIX(uniRandVectors RandState (rows, cols, max), rows, cols)
    fun uniRandCols RandState (max, (rows, cols)) =
	COLMATRIX(uniRandVectors RandState (cols, rows, max), rows, cols)


    fun printInfo m =
	case m of
	    COLMATRIX (_, rows, cols) => 
	      print("Collum matrix "^Int.toString(rows)^" * "
		      ^Int.toString(cols)^"\n")
	    | ROWMATRIX (_, rows, cols) => 
	      print("Row matrix "^Int.toString(rows)^" * "
		      ^Int.toString(cols)^"\n")

    fun printMat printElem printEnd m =
	let
	    fun printVec v = 
		case v of
			[] => ()
		| last::[] => printEnd last
		| head::v'  => (printElem head; printVec v')
	in
	    case m of
		ROWMATRIX (vs, rows, cols)=> 
		(printInfo m; List.app printVec vs)
	      | COLMATRIX (vs, rows, cols)=> 
		(printInfo m; List.app printVec (changeVectors vs))
	end
	fun real2str x =
		if x >= 0.0 then Real.toString(x)
		else "-" ^ Real.toString(~x)
	
	fun int2str x =
		if x >= 0 then Int.toString(x)
		else "-" ^ Int.toString(~x)
    val printMatReal = printMat (fn a => print (real2str(a)^", ")) 
										(fn a => print (real2str(a) ^"\n"))
    val printMatInt  = printMat (fn a => print (int2str(a)^ ", "))
										(fn a => print (int2str(a) ^ "\n"))
	
	fun printfMatrixReal (fname, m) = 
		let val fout =  TextIO.openOut (fname)
		in
			((printMat (fn a => TextIO.output (fout, (real2str(a)^", ")))
						(fn a => TextIO.output (fout, (real2str(a)^"\n")))
						m);
			TextIO.flushOut(fout);
			TextIO.closeOut(fout))
		end

	fun printfMatrixInt (fname, m) = 
		let val fout =  TextIO.openOut (fname)
		in
			((printMat (fn a => TextIO.output (fout, (int2str(a)^", ")))
						(fn a => TextIO.output (fout, (int2str(a)^"\n")))
						m);
			TextIO.flushOut (fout);
			TextIO.closeOut (fout))
		end

    fun size (COLMATRIX(_, rows, cols)) = (rows, cols)
      | size (ROWMATRIX(_, rows, cols)) = (rows, cols)
					      
    fun fromList2Vectors (a, nElems, nVectors) =
	let
	    fun fromList2Vector (a, nElems) =
		case (a, nElems) of
		    (_, 0) => (a, [])
		  | ([], nElems) => raise UnmatchedDimension
		  | (x::a', nElems) => case fromList2Vector(a', nElems - 1) of
					   (a'', acc) => (a'', x::acc)
	in
	    case nVectors of
		0 => []
	      | nVectors => case fromList2Vector(a, nElems) of
				(a', v) => v::fromList2Vectors(a', nElems, nVectors-1)
	end
    fun fromVector2Rows (a, (rows, cols)) =
	if List.length(a) <> rows*cols 
	then raise UnmatchedDimension
	else ROWMATRIX(fromList2Vectors(a, cols, rows), rows, cols)
    fun fromVector2Cols (a, (rows, cols)) = transpose_changeType(fromVector2Rows(a, (cols, rows)))

	fun fromVectors2Cols (vs, (rows, cols)) = 
	    if checkValid(COLMATRIX(vs, rows, cols)) then COLMATRIX(vs, rows, cols) else raise UnmatchedDimension
	fun fromVectors2Rows (vs, (rows, cols)) =
	    if checkValid(ROWMATRIX(vs, rows, cols)) then ROWMATRIX(vs, rows, cols) else raise UnmatchedDimension
    
    fun mapVectors f vs =
	List.map (fn v => List.map f v) vs
    fun map f (COLMATRIX(vs, rows, cols)) = COLMATRIX (mapVectors f vs, rows, cols)
      | map f (ROWMATRIX(vs, rows, cols)) = ROWMATRIX (mapVectors f vs, rows, cols)
    fun addScalarInt (m, x:int) = map (fn a => a+x) m
    fun addScalarReal (m, x:real) = map (fn a => a+x) m
    fun mulScalarInt (m, x:int) = map (fn a => a*x) m
    fun mulScalarReal (m, x:real) = map (fn a => a*x) m

    fun mergeVector f (v1, v2) =
	case (v1, v2) of
	    ([], [])=>[]
	  | (hdv1::v1', hdv2::v2') => f(hdv1, hdv2)::(mergeVector f (v1', v2'))
	  | _ => raise UnmatchedDimension

    fun mergeVectors f vs1 vs2 = mergeVector (fn (v1, v2) => mergeVector f (v1, v2)) (vs1, vs2)

    fun merge f (m1, m2) =
	case (m1, m2) of
	    (COLMATRIX (vs1, rows1, cols1), COLMATRIX(vs2, rows2, cols2)) =>
	    COLMATRIX(mergeVectors f vs1 vs2, rows1, cols1)
	  | (ROWMATRIX(vs1, rows1, cols1), ROWMATRIX(vs2, rows2, cols2)) => 
	    ROWMATRIX(mergeVectors f vs1 vs2, rows1, cols1)
	  | _ => raise WrongMatrixType


    val dotMulMatrixInt  = merge (fn (a:int, b)=>a*b)
    val dotMulMatrixReal = merge (fn (a:real, b)=>a*b)
    val addMatrixInt     = merge (fn (a:int, b)=>a+b)
    val addMatrixReal    = merge (fn (a:real, b)=>a+b)
    
    fun mergeVect2Vects f (v1, v2s) =
	case v2s of
	    [] => []
	  | v2::v2s' => (mergeVector f (v1, v2))::(mergeVect2Vects f (v1, v2s'))
    fun mergeVect2Matrix f (vx, m) =
	case m of
	    COLMATRIX(vs, rows, cols) => COLMATRIX(mergeVect2Vects f (vx, vs), rows, cols)
	  | ROWMATRIX(vs, rows, cols) => ROWMATRIX(mergeVect2Vects f (vx, vs), rows, cols)
    val addVect2MatrixReal  = mergeVect2Matrix (fn (v1:real, v2)=> v1+v2)
    val addVect2MatrixInt   = mergeVect2Matrix (fn (v1:int, v2)=> v1+v2)
	
	fun mergeVect'2Vects f (v1, v2s) =
	let
		fun curf a b = f (a, b)
	in
		case (v1, v2s) of
			([], []) => []
		|   (x::v1', v2::v2s') => (List.map (curf x) v2) :: (mergeVect'2Vects f (v1', v2s'))
		|   _ => raise UnmatchedDimension
	end	
	fun mergeVect'2Matrix f (vx, m) =
	case m of
	    COLMATRIX(vs, rows, cols) => COLMATRIX(mergeVect'2Vects f (vx, vs), rows, cols)
	  | ROWMATRIX(vs, rows, cols) => ROWMATRIX(mergeVect'2Vects f (vx, vs), rows, cols)
	
    fun foldl f acc m = 
	let fun foldlVectors(vs) =
		case vs of
		    []      => []
		  | v::vs' => (List.foldl f acc v)::foldlVectors(vs')
	in
	    case m of
		COLMATRIX(vs, rows, cols) => foldlVectors(vs)
	     | 	ROWMATRIX(vs, rows, cols) => foldlVectors(vs)
	end
    val sumInt = foldl (fn (x, acc:int) => acc+x) 0
    val sumReal = foldl (fn (x, acc:real) => acc+x) 0.0

    fun foldl2Vectors f acc (v1, v2) =
	let fun foldl(v1, v2, acc) =
		case (v1, v2) of
		    (hdv1::v1', hdv2::v2') => foldl(v1', v2', f(hdv1, hdv2, acc))
		  | ([], [])               => acc
		  | _                      => raise UnmatchedDimension
	in
	    foldl(v1, v2, acc)
	end
    fun mulVectors mul2VectorsFun (v1s, v2s) = 
	let     
	    fun mulVectorsVector (v1s, v2) =
		case v1s of
		    []      => []
		  | v1::v1s' => (mul2VectorsFun(v1, v2))::
			       (mulVectorsVector (v1s',v2))
	in
	    case v2s of
		[]        => []
	      | v2::v2s' => (mulVectorsVector (v1s, v2))::(mulVectors mul2VectorsFun (v1s, v2s'))
	end
    val mul2VectorsInt   = foldl2Vectors (fn (x, y, acc:int) => acc+x*y) 0 
    val mul2VectorsReal  = foldl2Vectors (fn (x, y, acc:real) => acc+x*y) 0.0
    val mulVectorsInt = mulVectors mul2VectorsInt
    val mulVectorsReal = mulVectors mul2VectorsReal
    fun mulMatrix mulVectorsFun returnRow ((ROWMATRIX (vs1, rows1, cols1)), (COLMATRIX (vs2, rows2, cols2))) = 
	if cols1 <> rows2
	then raise UnmatchedDimension
	else if returnRow then
	    (ROWMATRIX (mulVectorsFun(vs2, vs1), rows1, cols2))
	else
	    (COLMATRIX (mulVectorsFun(vs1, vs2), rows1, cols2))
      | mulMatrix mulVectorsFun returnRow (_, _) = raise WrongMatrixType
    val mulMatrixIntC = mulMatrix mulVectorsInt false
    val mulMatrixIntR = mulMatrix mulVectorsInt true
    val mulMatrixRealC = mulMatrix mulVectorsReal false
    val mulMatrixRealR = mulMatrix mulVectorsReal true			
		*)
end



