open NN;

print ("Testing Neural Network using iris dataset \n");
val RandState = Random.rand (10, 1);

(* make default weight matrix *)
val W1 = Matrix.fromVec(Matrix.fromList2Vec(
	[ 0.268833569773050, 0.159382619929490, 1.78919846986288, 0.362702112473053, 
	  ~0.0620721741081560, 0.335748566804040, 0.244446885155895, 0.146935733548329,
	  0.916942507297543, ~0.653844148152637, 1.38471851494244, ~0.0315274365948281, 
	  0.744848803892732, ~0.603743461342519, 0.517346504958930, ~0.393641401879319,
	  ~1.12942343050182, ~0.216796011152842, ~0.674943470078261, 0.357371451913048,
	  0.704517244900240, 0.358619325664419, 0.363442566691619, 0.444197815878821,
	  0.431086660184060, 0.171312233269325, 1.51746173316593, ~0.102483029149887,
	  0.708596206714807, 0.815117644582365, ~0.151720462393008, ~0.573535053484575]), (4, 8));
val W2 = Matrix.fromVec(Matrix.fromList2Vec(
	[~0.377902774590294, ~0.0361481634760753, 0.386527784108133,
	 ~0.286201008094743, ~0.0853644202089966, 0.392187335466152,
	 ~1.04096164834337, 0.112856624932186, ~0.305347383509559,
	 0.508544229487324, 0.110612217617416, 0.0273505689623758,
	 0.114972217813595, ~0.305781227226092, ~0.429255197347310,
	 ~0.266907466897330, ~0.0106247376619135, ~0.393681962580770,
	 0.484473694975681, ~0.0582935362790490,  ~0.00242160317422451,
	 ~0.605112432941783, 0.221928039805888, 0.541866642020088]), (8, 3));
	 
(* Test SIGM - MSE - no Weight decay - no momentum *)
val irisparams:params =  
	{batchsize = 10,   (* number of training cases per batch *)
	      nBatches = 12,    (* number of batches *)
	      testsize = 30,    (* number of testing cases *)
		  nTestBatches = 1, (* because of memory issue, we should split big test into smaller batches *)
	      lambda = 0.0,   (* momentum coefficient *)
		  momentumSchedule = false,
		  maxLambda = 0.0,
	      lr = 0.005,        (* learning rate *)
	      costType = MSE,   (* cost function type *)
	      initType = NORMAL,(* initialization type *)
	      actType = SIGM,   (* activation function type *)
	      layerSizes = [4, 8, 3], (* structure of network *)
	      nItrs = 10,         (* number of iterations/epoches *)
		  initWs = [SOME (Matrix.mdcopy(W1)), SOME (Matrix.mdcopy(W2))],
		  initBs = [],
		  wdType = L0,
		  wdValue = 0.0,
		  verbose = false
	     };
val files:fileNames = { data_train   = "../datasets/Iris/data_train.csv",
		  labels_train = "../datasets/Iris/labels_train.csv",
		  data_test    = "../datasets/Iris/data_test.csv",
		  labels_test  = "../datasets/Iris/labels_test.csv"
		};
NN.setParams(irisparams);

val expect_output_fprop_sigm = Matrix.fromList2Vec([0.305016214491485, 0.496920937696479, 0.375356107479804,
			   0.296595372457597, 0.490981425809185, 0.391388450437457, 
			   0.306301243937375, 0.475155141795554, 0.432003686636061,
			   0.296129238678094, 0.479820533472700, 0.440763636791806,
			   0.299935147998812, 0.478870958395587, 0.440112356711004,
			   0.295673182396037, 0.477484566674150, 0.430055158521248,
			   0.299969908281724, 0.491645955318928, 0.382166831456531,
			   0.291058417131523, 0.493184674712689, 0.397959279993849,
			   0.292551288596198, 0.493128514525123, 0.395116901630389,
			   0.304417542199083, 0.500336883383730, 0.375172793661932]);
val expect_MSE_cost = 0.365573772730045;
val expect_gradient_out = Matrix.fromList2Vec([0.0305016214491485, 0.0496920937696479, ~0.0624643892520196, 
			   0.0296595372457597, ~0.0509018574190815, 0.0391388450437457,
			   ~0.0693698756062625, 0.0475155141795554, 0.0432003686636061,
			   ~0.0703870761321906, 0.0479820533472700, 0.0440763636791806,
			   ~0.0700064852001188,	0.0478870958395587, 0.0440112356711004,
			   ~0.0704326817603963,	0.0477484566674150, 0.0430055158521248,
			   0.0299969908281724,	0.0491645955318928, ~0.0617833168543469,
			   0.0291058417131523, ~0.0506815325287311, 0.0397959279993849,
			   0.0292551288596198, ~0.0506871485474877, 0.0395116901630389,
			   0.0304417542199083, 0.0500336883383730, ~0.0624827206338068]);
val expect_bproped_gradient = Matrix.fromList2Vec([~0.000891111094200543, ~0.000330540498275334, 0.00164553675879294,
			       3.64399337506018e~05, 5.25190929898768e~05, 0.000504527467795780,
			       ~0.000787210930388351, 0.000154539687671734, 0.000598470687944687,
			       ~0.00211585506151018, 0.000883039621326874, ~0.00228613639581374,
			       0.000699680325311161,~0.00204401703918776, 0.000407665527325771,
			       ~0.00227068173407500, 0.000667024065565804, ~0.00205087459963230,
			       0.000591990102948399, ~0.00230558100752066, 0.000696329765537579,
			       ~0.00207531519526938, 0.000629453863922907, ~0.00220585771765557,
			       ~0.000991381482750058, ~0.000427156388546111, 0.00182344504218146,
			       5.69004324033357e~05, 9.75959713565362e~05, 0.000449368999298318,
			       ~0.000823301840055290, 0.000132950886377768, 0.000100756939640410,
			       0.000426718422132309, ~0.000800176280621577, 0.000163948544716788,
			       ~0.000811570649953576, ~0.000153399453618402, 0.00145058296398899,
			       ~2.29889141214365e~05]);
val expect_output_test = Matrix.fromList2Vec([0.314557186811370,0.447536937762953,0.416547042123549,
			  0.292326151551991,0.467571091764818,0.381135726704670,
			  0.313410384055149,0.462281526998532,0.353948428909056,
			  0.297740914585621,0.452030188572930,0.419562167743577,
			  0.309425770871500,0.476054553613960,0.355847070465814,
			  0.293507377318868,0.461728402556457,0.380916695840949,
			  0.294316165641544,0.464304062210079,0.379988552244028,
			  0.300224212566987,0.451863563735034,0.424077898921584,
			  0.306187380752394,0.474681643158968,0.360078718837744,
			  0.309478695143130,0.462882057444723,0.358668035346452,
			  0.303210104697344,0.464827097573617,0.366118078573927,
			  0.296943697490070,0.465575577126681,0.374119949690426,
			  0.305770783010568,0.465756637297524,0.362595099084275,
			  0.310315387769099,0.467165038711548,0.357202548497858,
			  0.310939107731653,0.463854844012381,0.357082530832018,
			  0.310421852011067,0.471223266717580,0.356456213326297,
			  0.308447185596048,0.449163727532041,0.417437893663019,
			  0.293979868857243,0.463978203600539,0.379427067701171,
			  0.295319694221117,0.457216935355717,0.381630896993142,
			  0.305451170808886,0.449646357032908,0.425714230543220,
			  0.306683367520299,0.463211013988830,0.364955739258140,
			  0.311582829187015,0.468515601806012,0.355065645408438,
			  0.307951297597511,0.448903124954147,0.417670627630633,
			  0.307052965016680,0.448783173258273,0.419499883264977,
			  0.302834343502340,0.466246287242706,0.368843676139514,
			  0.301334367691949,0.451904733480779,0.422870470280684,
			  0.316848113184097,0.447779650523088,0.415155508119000,
			  0.306955165033968,0.469010798129960,0.360578181537099,
			  0.299768694259980,0.450803451467746,0.420582068843711,
			  0.308186460782349,0.476212743056655,0.357490889750124
			 ])
(* fun initlayer_W (W, actType) =
	let
	val input = Matrix.initRows (0.0, (1, 1))
	val output = Matrix.initRows (0.0, (1, 1))
	val GradW = Matrix.initCols (0.0,(Matrix.size(W)))
	val GradB = Matrix.toVector(Matrix.initRows(0.0, (#2(Matrix.size(W)), 1)))
	val B = Matrix.toVector(Matrix.initRows(0.0, (#2(Matrix.size(W)), 1)))
    in
	{input = input, output = output, GradW = GradW, 
	 GradB = GradB, B = B, W = W, actType = actType}
    end *)
	
val [layer1, layer2] = initLayers RandState (#layerSizes(irisparams), [SOME (Matrix.mdcopy(W1)), SOME (Matrix.mdcopy(W2))], 
									[]);
val network = [layer1, layer2];

val data_train = readData(#data_train(files), 
				  hd(#layerSizes(irisparams)),
				  #batchsize(irisparams), #nBatches(irisparams));
val labels_train = readData(#labels_train(files), 
				    List.last(#layerSizes(irisparams)),
				  #batchsize(irisparams), #nBatches(irisparams));
val data_test = readData(#data_test(files), 
				 hd(#layerSizes(irisparams)),
				 #testsize(irisparams), 1);
val labels_test = readData(#labels_test(files), 
				   List.last(#layerSizes(irisparams)),
				   #testsize(irisparams), 1);
(*Matrix.printMat(hd(data_train));
Matrix.printMat(W1);*)
fun printAssert (a, dsc) = print (dsc ^ Bool.toString(a) ^ "\n")
fun isEqReal(a, b) = Real.abs(a - b) < 0.0000001

val (fpropedNetwork, output_fprop_sigm) = fprop(network, hd(data_train));
(*val _ = Matrix.printMat (output_fprop_sigm)*)
val _ = printAssert (Matrix.eq(Matrix.toVector(output_fprop_sigm), expect_output_fprop_sigm), "Test fprop() with sigm activation function:  ")
val (mse_cost, gradient_out) = computeCost(output_fprop_sigm, hd(labels_train), MSE);
val _ = printAssert (isEqReal(mse_cost, expect_MSE_cost), "Test computeCost() MSE error output: ") 
val _ = printAssert (Matrix.eq(Matrix.toVector(gradient_out), expect_gradient_out), "Test computeCost() MSE gradient output: ");

val (bpropedNetwork, bproped_gradient) = bprop(fpropedNetwork, gradient_out);
(* val _ = Matrix.printMat(bproped_gradient) *)
val _ = printAssert (Matrix.eq(Matrix.toVector(bproped_gradient), expect_bproped_gradient), "Test bprop() sigm function, no momentum: ");


val [layer1, layer2] = initLayers RandState (#layerSizes(irisparams), [SOME (Matrix.mdcopy(W1)), SOME (Matrix.mdcopy(W2))], 
									[]);
val network = [layer1, layer2];
val trainedNetwork = trainNN(network, data_train, labels_train);
val (_, output_test) = fprop(trainedNetwork, hd(data_test));
(* val _ = Matrix.printMat (output_test) *)
val _ = printAssert (Matrix.eq(Matrix.toVector(output_test), expect_output_test), "Test trainNN() function with sigm without momentum");

(* Test TANH - NLL - Momentum schedule - no Weight Decay *)
val irisparams:params =  
	{batchsize = 10,   (* number of training cases per batch *)
	      nBatches = 12,    (* number of batches *)
	      testsize = 30,    (* number of testing cases *)
		  nTestBatches = 1, (* because of memory issue, we should split big test into smaller batches *)
	      lambda = 0.95,   (* momentum coefficient *)
		  momentumSchedule = false,
		  maxLambda = 0.0,
	      lr = 0.005,        (* learning rate *)
	      costType = NLL,   (* cost function type *)
	      initType = NORMAL,(* initialization type *)
	      actType = TANH,   (* activation function type *)
	      layerSizes = [4, 8, 3], (* structure of network *)
	      nItrs = 10,         (* number of iterations/epoches *)
		  initWs = [SOME (Matrix.mdcopy(W1)), SOME (Matrix.mdcopy(W2))],
		  initBs = [],
		  wdType = L0,
		  wdValue = 0.0,
		  verbose = false
	     };
NN.setParams(irisparams);

val expect_output_fprop_tanh = Matrix.fromList2Vec([~0.171769437164186, 0.125251208119796, ~1.28572875152613,
						~0.371587517189631,	0.0587321744769369,	~1.06306246612303,
						0.0222268951192020, ~0.159487943967483, ~0.970979655810838,
						~0.168221643780007, ~0.0968418367537368, ~0.822009916626929,
						~0.105400725694416, ~0.109242537642415, ~0.829922371945778,
						~0.171985666752481, ~0.123770234353647, ~0.980697123505234,
						~0.288591610352196, 0.0700363460004952, ~1.15342367150382,
						~0.367876088995726, 0.0876215367553500, ~1.06850618558453,
						~0.338358520023072, 0.0888490964039660, ~1.09994747201995,
						~0.161878587181706, 0.150325576251707, ~1.30200139588689])
val expect_nll_cost = 1.172489497789706;
val expect_nll_gradient = Matrix.fromList2Vec([0.0373957621495670, 0.0503288278776276, ~0.0877245900271946,
0.0329100457962038, ~0.0493926083239136, 0.0164825625277098,
~0.0546326054032533, 0.0378291067917365, 0.0168034986115168,
~0.0614504042260915, 0.0414018440131392, 0.0200485602129523,
~0.0596891527569958, 0.0401562776521968, 0.0195328751047990,
~0.0599170822247741, 0.0420628818410649, 0.0178542003837092,
0.0350571341993510, 0.0501795258852212, ~0.0852366600845722,
0.0325390504656789, ~0.0486872865715030, 0.0161482361058241,
0.0333345020832335, ~0.0488991783861487, 0.0155646763029151,
0.0372271244148187, 0.0508683951334055, ~0.0880955195482242])
val expect_test_output_momentum = Matrix.fromList2Vec([0.499418935161419, ~0.492130044918502, ~0.760940615148993,
~0.988082513781522, ~0.160504865077884, ~0.210267855074960,
~0.816411593755126, ~0.201069454934034, ~0.384592250902163,
0.111923641979239, ~0.422975935465221, ~0.688066892007752,
~1.02629386041075, ~0.115808626917073, ~0.205922445570854,
~0.799297464139543, ~0.210625609616719, ~0.386145143746000,
~0.902716215280470, ~0.187970571320450, ~0.279141800087047,
0.227370425678425, ~0.437648406704225, ~0.665714304969460,
~1.03292783980947, ~0.121378368909046, ~0.194907100649621,
~0.809966452932810, ~0.198067393730138, ~0.394075601592336,
~0.884541488323771, ~0.179271769648907, ~0.321489307247901,
~0.909704789168073, ~0.173011467200952, ~0.295871822254003,
~0.893772816268216, ~0.171980808666140, ~0.317375649216781,
~0.914157753345138, ~0.160780237420447, ~0.300026274870026,
~0.837637901762361, ~0.188884087289178, ~0.367237153232971,
~0.964587462639307, ~0.136072745920876, ~0.260117309958382,
0.370623213798278, ~0.469128835357053, ~0.741779400305161,
~0.856326398627885, ~0.188193579063067, ~0.340175915021164,
~0.612521111407497, ~0.269785769423782, ~0.542834534668197,
0.410926303546946, ~0.474620223013553, ~0.692274180835528,
~0.785280211826634, ~0.200907226603125, ~0.408459729829556,
~0.932330564353231, ~0.152595459104237, ~0.288173124375201,
0.381308746803987, ~0.476561849656148, ~0.762340637709012,
0.384500872563469, ~0.476072627092880, ~0.742023606012043,
~0.881029309003654, ~0.171482777970208, ~0.321915748534526,
0.226445583364993, ~0.434185140115929, ~0.663588893063126,
0.508678343667918, ~0.488019416182038, ~0.750577865974859,
~0.966263007003914, ~0.148285772936908, ~0.251348693473654,
0.212486804840382, ~0.447503723822102, ~0.719056819565306,
~1.02099667927735, ~0.114686915339751, ~0.211324536698714])
val [layer1, layer2] = initLayers RandState (#layerSizes(irisparams), #initWs(irisparams), 
									#initBs(irisparams));
val network = [layer1, layer2];

val (fpropedNetwork, output_fprop_tanh) = fprop(network, hd(data_train));
val _ = printAssert (Matrix.eq(Matrix.toVector(output_fprop_tanh), expect_output_fprop_tanh), 
	     "Test fprop() with tanh activation function: ");
val (nll_cost, gradient_nll_out) = computeCost(output_fprop_tanh, hd(labels_train), NLL);
val _ = printAssert(isEqReal(nll_cost, expect_nll_cost), 
	     "Test computeCost() NLL error output: ");
val _ = printAssert(Matrix.eq(Matrix.toVector(gradient_nll_out), expect_nll_gradient), 
	     "Test computeCost() NLL gradient output: ");

val [layer1, layer2] = initLayers RandState (#layerSizes(irisparams), [SOME (Matrix.mdcopy(W1)), SOME (Matrix.mdcopy(W2))], 
									[]);
val network = [layer1, layer2];
val trainedNetwork = trainNN(network, data_train, labels_train);
val (predicted, output_test) = fprop(trainedNetwork, hd(data_test));
(* val _ = Matrix.printMat (output_test) *)
val _ = printAssert (Matrix.eq(Matrix.toVector(output_test), expect_test_output_momentum), 
	"Test trainNN() function with tanh, momentum, nll cost function");
val (error, _) = computeCost(output_test, hd(labels_test), PER);
(* val _ = print (Real.toString (error)) *)
val _ = printAssert (isEqReal (error, 0.4), 
	     "Test computeCost() PER error output");

(* Test SIGM - CE - Momentum - L2 Weight Decay *)
val irisparams:params =  
	{batchsize = 10,   (* number of training cases per batch *)
	      nBatches = 12,    (* number of batches *)
	      testsize = 30,    (* number of testing cases *)
		  nTestBatches = 1, (* because of memory issue, we should split big test into smaller batches *)
	      lambda = 0.95,   (* momentum coefficient *)
	      lr = 0.005,        (* learning rate *)
		  momentumSchedule = false,
		  maxLambda = 0.0,
	      costType = CE,   (* cost function type *)
	      initType = NORMAL,(* initialization type *)
	      actType = SIGM,   (* activation function type *)
	      layerSizes = [4, 8, 3], (* structure of network *)
	      nItrs = 10,         (* number of iterations/epoches *)
		  initWs = [SOME (Matrix.mdcopy(W1)), SOME (Matrix.mdcopy(W2))],
		  initBs = [],
		  wdType = L2,
		  wdValue = 0.00001,
		  verbose = false
	     };
NN.setParams(irisparams);
val expect_ce_cost = 2.044786962903261;
val expect_ce_gradient = Matrix.fromList2Vec([0.0305016214491485, 0.0496920937696479, ~0.0624643892520196,
0.0296595372457597, ~0.0509018574190815, 0.0391388450437457,
~0.0693698756062625, 0.0475155141795554, 0.0432003686636061,
~0.0703870761321906, 0.0479820533472700, 0.0440763636791806,
~0.0700064852001188, 0.0478870958395587, 0.0440112356711004,
~0.0704326817603963, 0.0477484566674150, 0.0430055158521248,
0.0299969908281724, 0.0491645955318928, ~0.0617833168543469,
0.0291058417131523, ~0.0506815325287311, 0.0397959279993849,
0.0292551288596198, ~0.0506871485474877, 0.0395116901630389,
0.0304417542199083, 0.0500336883383730, ~0.0624827206338068]);
val expect_output_test = Matrix.fromList2Vec([~0.686098801338695, ~0.429442033445893, ~0.495229231088862,
~0.894376208525007, ~0.349326253131392, ~0.559502917151008,
~0.805264192686281, ~0.368791778604887, ~0.664767082659308,
~0.778372438711949, ~0.412871829833224, ~0.474272774847858,
~0.835557478311331, ~0.308831302677432, ~0.649717388123587,
~0.871474310819477, ~0.377644208909052, ~0.575271743651768,
~0.877034976621512, ~0.359172235998760, ~0.570184018402506,
~0.762695170417632, ~0.411922366345331, ~0.457606554634973,
~0.849894577921242, ~0.316147583738860, ~0.631596131214544,
~0.820001044985765, ~0.370590016669807, ~0.647713906483357,
~0.847271287235755, ~0.361781835937063, ~0.618165259783687,
~0.872454845400024, ~0.360986634939408, ~0.588548328285420,
~0.839951047275876, ~0.358316290899705, ~0.629044093005363,
~0.820953746595999, ~0.347147154870560, ~0.650770462147542,
~0.813481702907425, ~0.362181002126333, ~0.654560286079093,
~0.824538387881811, ~0.330121587877338, ~0.651409737777428,
~0.718907056018133, ~0.423634335163673, ~0.488965304467892,
~0.875950562340081, ~0.368920525498838, ~0.575779696584919,
~0.847020810678770, ~0.395373329246594, ~0.585790299108106,
~0.733694575067170, ~0.417161876033054, ~0.451594016711719,
~0.821708267456172, ~0.367032389064735, ~0.631073799576388,
~0.822156242824718, ~0.344902567810890, ~0.653515164465869,
~0.719142626713216, ~0.425320272312119, ~0.490149593736210,
~0.725933036224796, ~0.423314266249004, ~0.478965086819342,
~0.844353855711070, ~0.354546639607971, ~0.611014863764273,
~0.757354935786903, ~0.412461022390110, ~0.462969837300596,
~0.675438340436423, ~0.429189977868341, ~0.501390892300618,
~0.839945326858988, ~0.341102301307798, ~0.633586850860768,
~0.764886094561146, ~0.416768228115622, ~0.472407461719039,
~0.838944089168667, ~0.307897282046727, ~0.644986680892834])
val [layer1, layer2] = initLayers RandState (#layerSizes(irisparams), #initWs(irisparams), 
									#initBs(irisparams));
val network = [layer1, layer2];
val (fpropedNetwork, output_fprop) = fprop(network, hd(data_train));
val (ce_cost, gradient_ce_out) = computeCost(output_fprop, hd(labels_train), CE);
val _ = printAssert (isEqReal (ce_cost, expect_ce_cost), "Test computeCost() CE cost output: ");
val _ = printAssert (Matrix.eq(Matrix.toVector(gradient_ce_out), expect_ce_gradient), 
	     "Test computeCost() CE gradient output");
val [layer1, layer2] = initLayers RandState (#layerSizes(irisparams), [SOME (Matrix.mdcopy(W1)), SOME (Matrix.mdcopy(W2))], 
									#initBs(irisparams));
val network = [layer1, layer2];
val trainedNetwork = trainNN(network, data_train, labels_train);
val (predicted, output_test) = fprop(trainedNetwork, hd(data_test));
val _ = printAssert (Matrix.eq(Matrix.toVector(output_test), expect_output_test), 
	"Test trainNN() function with sigm, momentum, L2, CE cost function: ");