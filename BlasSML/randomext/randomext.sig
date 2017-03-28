(*
   File: randomext.sml
   Content: extend random library
   Author: Dang Ha The Hien, Hiof, hdthe@hiof.no
*)
signature RANDOMEXT =
sig
    val rand_normal: Random.rand -> real * real -> real
    val rand_perm:   Random.rand -> int * int -> int list
end
