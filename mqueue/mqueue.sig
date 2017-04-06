(*
  File: mqueue.sml
  Content: mqueue library for MLTon, designed for evaluating ADATE programs
  Author: Dang Ha The Hien
  Convention:
    eq: evaluating queue, all ADATE and Evaluator processes use this queue
    rq: result queue, 
    return code: 
       - -1: error
       - 0: OK
*)

signature MQUEUE =
sig
  exception MQUEUE_ERROR

  (*val eq_open: unit -> int
   val eq_close: unit -> int *)
  (*open eq queue, use process id to automatically open a result queue, then wait for the result, close eq queue and return*)
  val eq_evaluate: string -> real
  
end
