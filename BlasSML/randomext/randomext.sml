structure Randomext :> RANDOMEXT =
struct
(* 	Original Box_Muller Algorithm 

    val cached_rand_normal = ref 0.0;
	val rand_use_last = ref false;
	fun rand_normal RandState (mean, std) =
		let
		fun box_muller () =
			let
			val x1 = 2.0 * (Random.randReal RandState) - 1.0
			val x2 = 2.0 * (Random.randReal RandState) - 1.0
			val w = x1 * x1 + x2 * x2
			in
			if (w < 1.0) then
				let
				val v = Math.sqrt((~2.0 * Math.ln(w)) / w)
				in
				(rand_use_last := true;
				 cached_rand_normal := x2 * v;
				 mean + x1 * v * std)
				end
			else
				box_muller()
			end
		in
		case !rand_use_last of
			true => (rand_use_last := false; mean + !cached_rand_normal*std)
		  | false => box_muller()
		end *)
		
	(* Short version for Box Muller written in ADATE ML*)
	fun realLess(a: real, b:real) = a<b;
	fun realAdd(a:real, b:real) = a+b;
	fun realSubtract(a:real, b:real) = a-b;
	fun realMultiply(a:real, b:real) = a*b;
	fun realDivide(a:real, b:real) = a/b;
	fun fromInt(a:int) = Real.fromInt(a);
	fun sqrt(a:real) = Math.sqrt(a);
	fun ln (a:real) = Math.ln(a);
	
	fun rand_normal RandState ( ( Mean, Std) : real * real ) : real =
    let
	  fun box_muller( ) : real  =
		  case realSubtract(realMultiply(2.0,
		  (Random.randReal RandState)), 1.0) of X1
		  => case realSubtract(realMultiply(2.0,
		  (Random.randReal RandState)), 1.0 ) of X2
		  => case realAdd(realMultiply(X1, X1), realMultiply( X2, X2)) of W
		  => case realLess(W, 1.0) of
		   true   => realAdd(Mean,
			   realMultiply(Std,
			   realMultiply(X1,
			   sqrt(realDivide(realMultiply(~2.0, ln(W)), W)))))
		  | false => box_muller()
		in
	  box_muller() 
    end
	
	fun rand_perm  RandState (m, n) = 
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
end

