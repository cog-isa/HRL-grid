       £K"	  јЂdЊ÷Abrain.Event:2}сшжv     ыљН	oшќЂdЊ÷A"ўн	
o
over_options/obs_t_phPlaceholder*
dtype0*
shape: */
_output_shapes
:€€€€€€€€€	
y
over_options/CastCastover_options/obs_t_ph*

SrcT0*

DstT0*/
_output_shapes
:€€€€€€€€€	
_
over_options/obs_t_float/yConst*
valueB
 *  C*
dtype0*
_output_shapes
: 
М
over_options/obs_t_floatRealDivover_options/Castover_options/obs_t_float/y*
T0*/
_output_shapes
:€€€€€€€€€	
я
Iover_options/q_func/convnet/Conv/weights/Initializer/random_uniform/shapeConst*%
valueB"             *
dtype0*;
_class1
/-loc:@over_options/q_func/convnet/Conv/weights*
_output_shapes
:
…
Gover_options/q_func/convnet/Conv/weights/Initializer/random_uniform/minConst*
valueB
 *чьSљ*
dtype0*;
_class1
/-loc:@over_options/q_func/convnet/Conv/weights*
_output_shapes
: 
…
Gover_options/q_func/convnet/Conv/weights/Initializer/random_uniform/maxConst*
valueB
 *чьS=*
dtype0*;
_class1
/-loc:@over_options/q_func/convnet/Conv/weights*
_output_shapes
: 
Ѕ
Qover_options/q_func/convnet/Conv/weights/Initializer/random_uniform/RandomUniformRandomUniformIover_options/q_func/convnet/Conv/weights/Initializer/random_uniform/shape*

seed *
seed2 *
dtype0*
T0*;
_class1
/-loc:@over_options/q_func/convnet/Conv/weights*&
_output_shapes
: 
Њ
Gover_options/q_func/convnet/Conv/weights/Initializer/random_uniform/subSubGover_options/q_func/convnet/Conv/weights/Initializer/random_uniform/maxGover_options/q_func/convnet/Conv/weights/Initializer/random_uniform/min*
T0*;
_class1
/-loc:@over_options/q_func/convnet/Conv/weights*
_output_shapes
: 
Ў
Gover_options/q_func/convnet/Conv/weights/Initializer/random_uniform/mulMulQover_options/q_func/convnet/Conv/weights/Initializer/random_uniform/RandomUniformGover_options/q_func/convnet/Conv/weights/Initializer/random_uniform/sub*
T0*;
_class1
/-loc:@over_options/q_func/convnet/Conv/weights*&
_output_shapes
: 
 
Cover_options/q_func/convnet/Conv/weights/Initializer/random_uniformAddGover_options/q_func/convnet/Conv/weights/Initializer/random_uniform/mulGover_options/q_func/convnet/Conv/weights/Initializer/random_uniform/min*
T0*;
_class1
/-loc:@over_options/q_func/convnet/Conv/weights*&
_output_shapes
: 
й
(over_options/q_func/convnet/Conv/weights
VariableV2*
shape: *
dtype0*
	container *
shared_name *;
_class1
/-loc:@over_options/q_func/convnet/Conv/weights*&
_output_shapes
: 
њ
/over_options/q_func/convnet/Conv/weights/AssignAssign(over_options/q_func/convnet/Conv/weightsCover_options/q_func/convnet/Conv/weights/Initializer/random_uniform*
T0*
validate_shape(*
use_locking(*;
_class1
/-loc:@over_options/q_func/convnet/Conv/weights*&
_output_shapes
: 
—
-over_options/q_func/convnet/Conv/weights/readIdentity(over_options/q_func/convnet/Conv/weights*
T0*;
_class1
/-loc:@over_options/q_func/convnet/Conv/weights*&
_output_shapes
: 
¬
9over_options/q_func/convnet/Conv/biases/Initializer/ConstConst*
valueB *    *
dtype0*:
_class0
.,loc:@over_options/q_func/convnet/Conv/biases*
_output_shapes
: 
ѕ
'over_options/q_func/convnet/Conv/biases
VariableV2*
shape: *
dtype0*
	container *
shared_name *:
_class0
.,loc:@over_options/q_func/convnet/Conv/biases*
_output_shapes
: 
¶
.over_options/q_func/convnet/Conv/biases/AssignAssign'over_options/q_func/convnet/Conv/biases9over_options/q_func/convnet/Conv/biases/Initializer/Const*
T0*
validate_shape(*
use_locking(*:
_class0
.,loc:@over_options/q_func/convnet/Conv/biases*
_output_shapes
: 
¬
,over_options/q_func/convnet/Conv/biases/readIdentity'over_options/q_func/convnet/Conv/biases*
T0*:
_class0
.,loc:@over_options/q_func/convnet/Conv/biases*
_output_shapes
: 
Л
2over_options/q_func/convnet/Conv/convolution/ShapeConst*%
valueB"             *
dtype0*
_output_shapes
:
Л
:over_options/q_func/convnet/Conv/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
Р
,over_options/q_func/convnet/Conv/convolutionConv2Dover_options/obs_t_float-over_options/q_func/convnet/Conv/weights/read*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*/
_output_shapes
:€€€€€€€€€ 
а
(over_options/q_func/convnet/Conv/BiasAddBiasAdd,over_options/q_func/convnet/Conv/convolution,over_options/q_func/convnet/Conv/biases/read*
T0*
data_formatNHWC*/
_output_shapes
:€€€€€€€€€ 
С
%over_options/q_func/convnet/Conv/ReluRelu(over_options/q_func/convnet/Conv/BiasAdd*
T0*/
_output_shapes
:€€€€€€€€€ 
г
Kover_options/q_func/convnet/Conv_1/weights/Initializer/random_uniform/shapeConst*%
valueB"          @   *
dtype0*=
_class3
1/loc:@over_options/q_func/convnet/Conv_1/weights*
_output_shapes
:
Ќ
Iover_options/q_func/convnet/Conv_1/weights/Initializer/random_uniform/minConst*
valueB
 *  Аљ*
dtype0*=
_class3
1/loc:@over_options/q_func/convnet/Conv_1/weights*
_output_shapes
: 
Ќ
Iover_options/q_func/convnet/Conv_1/weights/Initializer/random_uniform/maxConst*
valueB
 *  А=*
dtype0*=
_class3
1/loc:@over_options/q_func/convnet/Conv_1/weights*
_output_shapes
: 
«
Sover_options/q_func/convnet/Conv_1/weights/Initializer/random_uniform/RandomUniformRandomUniformKover_options/q_func/convnet/Conv_1/weights/Initializer/random_uniform/shape*

seed *
seed2 *
dtype0*
T0*=
_class3
1/loc:@over_options/q_func/convnet/Conv_1/weights*&
_output_shapes
: @
∆
Iover_options/q_func/convnet/Conv_1/weights/Initializer/random_uniform/subSubIover_options/q_func/convnet/Conv_1/weights/Initializer/random_uniform/maxIover_options/q_func/convnet/Conv_1/weights/Initializer/random_uniform/min*
T0*=
_class3
1/loc:@over_options/q_func/convnet/Conv_1/weights*
_output_shapes
: 
а
Iover_options/q_func/convnet/Conv_1/weights/Initializer/random_uniform/mulMulSover_options/q_func/convnet/Conv_1/weights/Initializer/random_uniform/RandomUniformIover_options/q_func/convnet/Conv_1/weights/Initializer/random_uniform/sub*
T0*=
_class3
1/loc:@over_options/q_func/convnet/Conv_1/weights*&
_output_shapes
: @
“
Eover_options/q_func/convnet/Conv_1/weights/Initializer/random_uniformAddIover_options/q_func/convnet/Conv_1/weights/Initializer/random_uniform/mulIover_options/q_func/convnet/Conv_1/weights/Initializer/random_uniform/min*
T0*=
_class3
1/loc:@over_options/q_func/convnet/Conv_1/weights*&
_output_shapes
: @
н
*over_options/q_func/convnet/Conv_1/weights
VariableV2*
shape: @*
dtype0*
	container *
shared_name *=
_class3
1/loc:@over_options/q_func/convnet/Conv_1/weights*&
_output_shapes
: @
«
1over_options/q_func/convnet/Conv_1/weights/AssignAssign*over_options/q_func/convnet/Conv_1/weightsEover_options/q_func/convnet/Conv_1/weights/Initializer/random_uniform*
T0*
validate_shape(*
use_locking(*=
_class3
1/loc:@over_options/q_func/convnet/Conv_1/weights*&
_output_shapes
: @
„
/over_options/q_func/convnet/Conv_1/weights/readIdentity*over_options/q_func/convnet/Conv_1/weights*
T0*=
_class3
1/loc:@over_options/q_func/convnet/Conv_1/weights*&
_output_shapes
: @
∆
;over_options/q_func/convnet/Conv_1/biases/Initializer/ConstConst*
valueB@*    *
dtype0*<
_class2
0.loc:@over_options/q_func/convnet/Conv_1/biases*
_output_shapes
:@
”
)over_options/q_func/convnet/Conv_1/biases
VariableV2*
shape:@*
dtype0*
	container *
shared_name *<
_class2
0.loc:@over_options/q_func/convnet/Conv_1/biases*
_output_shapes
:@
Ѓ
0over_options/q_func/convnet/Conv_1/biases/AssignAssign)over_options/q_func/convnet/Conv_1/biases;over_options/q_func/convnet/Conv_1/biases/Initializer/Const*
T0*
validate_shape(*
use_locking(*<
_class2
0.loc:@over_options/q_func/convnet/Conv_1/biases*
_output_shapes
:@
»
.over_options/q_func/convnet/Conv_1/biases/readIdentity)over_options/q_func/convnet/Conv_1/biases*
T0*<
_class2
0.loc:@over_options/q_func/convnet/Conv_1/biases*
_output_shapes
:@
Н
4over_options/q_func/convnet/Conv_1/convolution/ShapeConst*%
valueB"          @   *
dtype0*
_output_shapes
:
Н
<over_options/q_func/convnet/Conv_1/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
°
.over_options/q_func/convnet/Conv_1/convolutionConv2D%over_options/q_func/convnet/Conv/Relu/over_options/q_func/convnet/Conv_1/weights/read*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*/
_output_shapes
:€€€€€€€€€@
ж
*over_options/q_func/convnet/Conv_1/BiasAddBiasAdd.over_options/q_func/convnet/Conv_1/convolution.over_options/q_func/convnet/Conv_1/biases/read*
T0*
data_formatNHWC*/
_output_shapes
:€€€€€€€€€@
Х
'over_options/q_func/convnet/Conv_1/ReluRelu*over_options/q_func/convnet/Conv_1/BiasAdd*
T0*/
_output_shapes
:€€€€€€€€€@
г
Kover_options/q_func/convnet/Conv_2/weights/Initializer/random_uniform/shapeConst*%
valueB"      @   @   *
dtype0*=
_class3
1/loc:@over_options/q_func/convnet/Conv_2/weights*
_output_shapes
:
Ќ
Iover_options/q_func/convnet/Conv_2/weights/Initializer/random_uniform/minConst*
valueB
 *:ЌУљ*
dtype0*=
_class3
1/loc:@over_options/q_func/convnet/Conv_2/weights*
_output_shapes
: 
Ќ
Iover_options/q_func/convnet/Conv_2/weights/Initializer/random_uniform/maxConst*
valueB
 *:ЌУ=*
dtype0*=
_class3
1/loc:@over_options/q_func/convnet/Conv_2/weights*
_output_shapes
: 
«
Sover_options/q_func/convnet/Conv_2/weights/Initializer/random_uniform/RandomUniformRandomUniformKover_options/q_func/convnet/Conv_2/weights/Initializer/random_uniform/shape*

seed *
seed2 *
dtype0*
T0*=
_class3
1/loc:@over_options/q_func/convnet/Conv_2/weights*&
_output_shapes
:@@
∆
Iover_options/q_func/convnet/Conv_2/weights/Initializer/random_uniform/subSubIover_options/q_func/convnet/Conv_2/weights/Initializer/random_uniform/maxIover_options/q_func/convnet/Conv_2/weights/Initializer/random_uniform/min*
T0*=
_class3
1/loc:@over_options/q_func/convnet/Conv_2/weights*
_output_shapes
: 
а
Iover_options/q_func/convnet/Conv_2/weights/Initializer/random_uniform/mulMulSover_options/q_func/convnet/Conv_2/weights/Initializer/random_uniform/RandomUniformIover_options/q_func/convnet/Conv_2/weights/Initializer/random_uniform/sub*
T0*=
_class3
1/loc:@over_options/q_func/convnet/Conv_2/weights*&
_output_shapes
:@@
“
Eover_options/q_func/convnet/Conv_2/weights/Initializer/random_uniformAddIover_options/q_func/convnet/Conv_2/weights/Initializer/random_uniform/mulIover_options/q_func/convnet/Conv_2/weights/Initializer/random_uniform/min*
T0*=
_class3
1/loc:@over_options/q_func/convnet/Conv_2/weights*&
_output_shapes
:@@
н
*over_options/q_func/convnet/Conv_2/weights
VariableV2*
shape:@@*
dtype0*
	container *
shared_name *=
_class3
1/loc:@over_options/q_func/convnet/Conv_2/weights*&
_output_shapes
:@@
«
1over_options/q_func/convnet/Conv_2/weights/AssignAssign*over_options/q_func/convnet/Conv_2/weightsEover_options/q_func/convnet/Conv_2/weights/Initializer/random_uniform*
T0*
validate_shape(*
use_locking(*=
_class3
1/loc:@over_options/q_func/convnet/Conv_2/weights*&
_output_shapes
:@@
„
/over_options/q_func/convnet/Conv_2/weights/readIdentity*over_options/q_func/convnet/Conv_2/weights*
T0*=
_class3
1/loc:@over_options/q_func/convnet/Conv_2/weights*&
_output_shapes
:@@
∆
;over_options/q_func/convnet/Conv_2/biases/Initializer/ConstConst*
valueB@*    *
dtype0*<
_class2
0.loc:@over_options/q_func/convnet/Conv_2/biases*
_output_shapes
:@
”
)over_options/q_func/convnet/Conv_2/biases
VariableV2*
shape:@*
dtype0*
	container *
shared_name *<
_class2
0.loc:@over_options/q_func/convnet/Conv_2/biases*
_output_shapes
:@
Ѓ
0over_options/q_func/convnet/Conv_2/biases/AssignAssign)over_options/q_func/convnet/Conv_2/biases;over_options/q_func/convnet/Conv_2/biases/Initializer/Const*
T0*
validate_shape(*
use_locking(*<
_class2
0.loc:@over_options/q_func/convnet/Conv_2/biases*
_output_shapes
:@
»
.over_options/q_func/convnet/Conv_2/biases/readIdentity)over_options/q_func/convnet/Conv_2/biases*
T0*<
_class2
0.loc:@over_options/q_func/convnet/Conv_2/biases*
_output_shapes
:@
Н
4over_options/q_func/convnet/Conv_2/convolution/ShapeConst*%
valueB"      @   @   *
dtype0*
_output_shapes
:
Н
<over_options/q_func/convnet/Conv_2/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
£
.over_options/q_func/convnet/Conv_2/convolutionConv2D'over_options/q_func/convnet/Conv_1/Relu/over_options/q_func/convnet/Conv_2/weights/read*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*/
_output_shapes
:€€€€€€€€€@
ж
*over_options/q_func/convnet/Conv_2/BiasAddBiasAdd.over_options/q_func/convnet/Conv_2/convolution.over_options/q_func/convnet/Conv_2/biases/read*
T0*
data_formatNHWC*/
_output_shapes
:€€€€€€€€€@
Х
'over_options/q_func/convnet/Conv_2/ReluRelu*over_options/q_func/convnet/Conv_2/BiasAdd*
T0*/
_output_shapes
:€€€€€€€€€@
И
!over_options/q_func/Flatten/ShapeShape'over_options/q_func/convnet/Conv_2/Relu*
T0*
out_type0*
_output_shapes
:
q
'over_options/q_func/Flatten/Slice/beginConst*
valueB: *
dtype0*
_output_shapes
:
p
&over_options/q_func/Flatten/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
–
!over_options/q_func/Flatten/SliceSlice!over_options/q_func/Flatten/Shape'over_options/q_func/Flatten/Slice/begin&over_options/q_func/Flatten/Slice/size*
T0*
Index0*
_output_shapes
:
s
)over_options/q_func/Flatten/Slice_1/beginConst*
valueB:*
dtype0*
_output_shapes
:
r
(over_options/q_func/Flatten/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
÷
#over_options/q_func/Flatten/Slice_1Slice!over_options/q_func/Flatten/Shape)over_options/q_func/Flatten/Slice_1/begin(over_options/q_func/Flatten/Slice_1/size*
T0*
Index0*
_output_shapes
:
k
!over_options/q_func/Flatten/ConstConst*
valueB: *
dtype0*
_output_shapes
:
Ѓ
 over_options/q_func/Flatten/ProdProd#over_options/q_func/Flatten/Slice_1!over_options/q_func/Flatten/Const*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
l
*over_options/q_func/Flatten/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
≥
&over_options/q_func/Flatten/ExpandDims
ExpandDims over_options/q_func/Flatten/Prod*over_options/q_func/Flatten/ExpandDims/dim*
T0*

Tdim0*
_output_shapes
:
i
'over_options/q_func/Flatten/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
№
"over_options/q_func/Flatten/concatConcatV2!over_options/q_func/Flatten/Slice&over_options/q_func/Flatten/ExpandDims'over_options/q_func/Flatten/concat/axis*
N*
T0*

Tidx0*
_output_shapes
:
Љ
#over_options/q_func/Flatten/ReshapeReshape'over_options/q_func/convnet/Conv_2/Relu"over_options/q_func/Flatten/concat*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
ч
Yover_options/q_func/action_value/fully_connected/weights/Initializer/random_uniform/shapeConst*
valueB"      *
dtype0*K
_classA
?=loc:@over_options/q_func/action_value/fully_connected/weights*
_output_shapes
:
й
Wover_options/q_func/action_value/fully_connected/weights/Initializer/random_uniform/minConst*
valueB
 *„≥Ёљ*
dtype0*K
_classA
?=loc:@over_options/q_func/action_value/fully_connected/weights*
_output_shapes
: 
й
Wover_options/q_func/action_value/fully_connected/weights/Initializer/random_uniform/maxConst*
valueB
 *„≥Ё=*
dtype0*K
_classA
?=loc:@over_options/q_func/action_value/fully_connected/weights*
_output_shapes
: 
л
aover_options/q_func/action_value/fully_connected/weights/Initializer/random_uniform/RandomUniformRandomUniformYover_options/q_func/action_value/fully_connected/weights/Initializer/random_uniform/shape*

seed *
seed2 *
dtype0*
T0*K
_classA
?=loc:@over_options/q_func/action_value/fully_connected/weights* 
_output_shapes
:
АА
ю
Wover_options/q_func/action_value/fully_connected/weights/Initializer/random_uniform/subSubWover_options/q_func/action_value/fully_connected/weights/Initializer/random_uniform/maxWover_options/q_func/action_value/fully_connected/weights/Initializer/random_uniform/min*
T0*K
_classA
?=loc:@over_options/q_func/action_value/fully_connected/weights*
_output_shapes
: 
Т
Wover_options/q_func/action_value/fully_connected/weights/Initializer/random_uniform/mulMulaover_options/q_func/action_value/fully_connected/weights/Initializer/random_uniform/RandomUniformWover_options/q_func/action_value/fully_connected/weights/Initializer/random_uniform/sub*
T0*K
_classA
?=loc:@over_options/q_func/action_value/fully_connected/weights* 
_output_shapes
:
АА
Д
Sover_options/q_func/action_value/fully_connected/weights/Initializer/random_uniformAddWover_options/q_func/action_value/fully_connected/weights/Initializer/random_uniform/mulWover_options/q_func/action_value/fully_connected/weights/Initializer/random_uniform/min*
T0*K
_classA
?=loc:@over_options/q_func/action_value/fully_connected/weights* 
_output_shapes
:
АА
э
8over_options/q_func/action_value/fully_connected/weights
VariableV2*
shape:
АА*
dtype0*
	container *
shared_name *K
_classA
?=loc:@over_options/q_func/action_value/fully_connected/weights* 
_output_shapes
:
АА
щ
?over_options/q_func/action_value/fully_connected/weights/AssignAssign8over_options/q_func/action_value/fully_connected/weightsSover_options/q_func/action_value/fully_connected/weights/Initializer/random_uniform*
T0*
validate_shape(*
use_locking(*K
_classA
?=loc:@over_options/q_func/action_value/fully_connected/weights* 
_output_shapes
:
АА
ы
=over_options/q_func/action_value/fully_connected/weights/readIdentity8over_options/q_func/action_value/fully_connected/weights*
T0*K
_classA
?=loc:@over_options/q_func/action_value/fully_connected/weights* 
_output_shapes
:
АА
д
Iover_options/q_func/action_value/fully_connected/biases/Initializer/ConstConst*
valueBА*    *
dtype0*J
_class@
><loc:@over_options/q_func/action_value/fully_connected/biases*
_output_shapes	
:А
с
7over_options/q_func/action_value/fully_connected/biases
VariableV2*
shape:А*
dtype0*
	container *
shared_name *J
_class@
><loc:@over_options/q_func/action_value/fully_connected/biases*
_output_shapes	
:А
з
>over_options/q_func/action_value/fully_connected/biases/AssignAssign7over_options/q_func/action_value/fully_connected/biasesIover_options/q_func/action_value/fully_connected/biases/Initializer/Const*
T0*
validate_shape(*
use_locking(*J
_class@
><loc:@over_options/q_func/action_value/fully_connected/biases*
_output_shapes	
:А
у
<over_options/q_func/action_value/fully_connected/biases/readIdentity7over_options/q_func/action_value/fully_connected/biases*
T0*J
_class@
><loc:@over_options/q_func/action_value/fully_connected/biases*
_output_shapes	
:А
ю
7over_options/q_func/action_value/fully_connected/MatMulMatMul#over_options/q_func/Flatten/Reshape=over_options/q_func/action_value/fully_connected/weights/read*
transpose_a( *
transpose_b( *
T0*(
_output_shapes
:€€€€€€€€€А
Д
8over_options/q_func/action_value/fully_connected/BiasAddBiasAdd7over_options/q_func/action_value/fully_connected/MatMul<over_options/q_func/action_value/fully_connected/biases/read*
T0*
data_formatNHWC*(
_output_shapes
:€€€€€€€€€А
™
5over_options/q_func/action_value/fully_connected/ReluRelu8over_options/q_func/action_value/fully_connected/BiasAdd*
T0*(
_output_shapes
:€€€€€€€€€А
ы
[over_options/q_func/action_value/fully_connected_1/weights/Initializer/random_uniform/shapeConst*
valueB"      *
dtype0*M
_classC
A?loc:@over_options/q_func/action_value/fully_connected_1/weights*
_output_shapes
:
н
Yover_options/q_func/action_value/fully_connected_1/weights/Initializer/random_uniform/minConst*
valueB
 *≤_Њ*
dtype0*M
_classC
A?loc:@over_options/q_func/action_value/fully_connected_1/weights*
_output_shapes
: 
н
Yover_options/q_func/action_value/fully_connected_1/weights/Initializer/random_uniform/maxConst*
valueB
 *≤_>*
dtype0*M
_classC
A?loc:@over_options/q_func/action_value/fully_connected_1/weights*
_output_shapes
: 
р
cover_options/q_func/action_value/fully_connected_1/weights/Initializer/random_uniform/RandomUniformRandomUniform[over_options/q_func/action_value/fully_connected_1/weights/Initializer/random_uniform/shape*

seed *
seed2 *
dtype0*
T0*M
_classC
A?loc:@over_options/q_func/action_value/fully_connected_1/weights*
_output_shapes
:	А
Ж
Yover_options/q_func/action_value/fully_connected_1/weights/Initializer/random_uniform/subSubYover_options/q_func/action_value/fully_connected_1/weights/Initializer/random_uniform/maxYover_options/q_func/action_value/fully_connected_1/weights/Initializer/random_uniform/min*
T0*M
_classC
A?loc:@over_options/q_func/action_value/fully_connected_1/weights*
_output_shapes
: 
Щ
Yover_options/q_func/action_value/fully_connected_1/weights/Initializer/random_uniform/mulMulcover_options/q_func/action_value/fully_connected_1/weights/Initializer/random_uniform/RandomUniformYover_options/q_func/action_value/fully_connected_1/weights/Initializer/random_uniform/sub*
T0*M
_classC
A?loc:@over_options/q_func/action_value/fully_connected_1/weights*
_output_shapes
:	А
Л
Uover_options/q_func/action_value/fully_connected_1/weights/Initializer/random_uniformAddYover_options/q_func/action_value/fully_connected_1/weights/Initializer/random_uniform/mulYover_options/q_func/action_value/fully_connected_1/weights/Initializer/random_uniform/min*
T0*M
_classC
A?loc:@over_options/q_func/action_value/fully_connected_1/weights*
_output_shapes
:	А
€
:over_options/q_func/action_value/fully_connected_1/weights
VariableV2*
shape:	А*
dtype0*
	container *
shared_name *M
_classC
A?loc:@over_options/q_func/action_value/fully_connected_1/weights*
_output_shapes
:	А
А
Aover_options/q_func/action_value/fully_connected_1/weights/AssignAssign:over_options/q_func/action_value/fully_connected_1/weightsUover_options/q_func/action_value/fully_connected_1/weights/Initializer/random_uniform*
T0*
validate_shape(*
use_locking(*M
_classC
A?loc:@over_options/q_func/action_value/fully_connected_1/weights*
_output_shapes
:	А
А
?over_options/q_func/action_value/fully_connected_1/weights/readIdentity:over_options/q_func/action_value/fully_connected_1/weights*
T0*M
_classC
A?loc:@over_options/q_func/action_value/fully_connected_1/weights*
_output_shapes
:	А
ж
Kover_options/q_func/action_value/fully_connected_1/biases/Initializer/ConstConst*
valueB*    *
dtype0*L
_classB
@>loc:@over_options/q_func/action_value/fully_connected_1/biases*
_output_shapes
:
у
9over_options/q_func/action_value/fully_connected_1/biases
VariableV2*
shape:*
dtype0*
	container *
shared_name *L
_classB
@>loc:@over_options/q_func/action_value/fully_connected_1/biases*
_output_shapes
:
о
@over_options/q_func/action_value/fully_connected_1/biases/AssignAssign9over_options/q_func/action_value/fully_connected_1/biasesKover_options/q_func/action_value/fully_connected_1/biases/Initializer/Const*
T0*
validate_shape(*
use_locking(*L
_classB
@>loc:@over_options/q_func/action_value/fully_connected_1/biases*
_output_shapes
:
ш
>over_options/q_func/action_value/fully_connected_1/biases/readIdentity9over_options/q_func/action_value/fully_connected_1/biases*
T0*L
_classB
@>loc:@over_options/q_func/action_value/fully_connected_1/biases*
_output_shapes
:
У
9over_options/q_func/action_value/fully_connected_1/MatMulMatMul5over_options/q_func/action_value/fully_connected/Relu?over_options/q_func/action_value/fully_connected_1/weights/read*
transpose_a( *
transpose_b( *
T0*'
_output_shapes
:€€€€€€€€€
Й
:over_options/q_func/action_value/fully_connected_1/BiasAddBiasAdd9over_options/q_func/action_value/fully_connected_1/MatMul>over_options/q_func/action_value/fully_connected_1/biases/read*
T0*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€
`
over_options/pred_ac/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
і
over_options/pred_acArgMax:over_options/q_func/action_value/fully_connected_1/BiasAddover_options/pred_ac/dimension*
T0*

Tidx0*#
_output_shapes
:€€€€€€€€€
V
act_t_phPlaceholder*
dtype0*
shape: *#
_output_shapes
:€€€€€€€€€
V
rew_t_phPlaceholder*
dtype0*
shape: *#
_output_shapes
:€€€€€€€€€
o
obs_tp1_ph/obs_tp1_phPlaceholder*
dtype0*
shape: */
_output_shapes
:€€€€€€€€€	
w
obs_tp1_ph/CastCastobs_tp1_ph/obs_tp1_ph*

SrcT0*

DstT0*/
_output_shapes
:€€€€€€€€€	
Y
obs_tp1_ph/truediv/yConst*
valueB
 *  C*
dtype0*
_output_shapes
: 
~
obs_tp1_ph/truedivRealDivobs_tp1_ph/Castobs_tp1_ph/truediv/y*
T0*/
_output_shapes
:€€€€€€€€€	
Z
done_mask_phPlaceholder*
dtype0*
shape: *#
_output_shapes
:€€€€€€€€€
W
	opt_stepsPlaceholder*
dtype0*
shape: *#
_output_shapes
:€€€€€€€€€
^
pred_q_a/one_hot/on_valueConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
_
pred_q_a/one_hot/off_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
X
pred_q_a/one_hot/depthConst*
value	B :*
dtype0*
_output_shapes
: 
ƒ
pred_q_a/one_hotOneHotact_t_phpred_q_a/one_hot/depthpred_q_a/one_hot/on_valuepred_q_a/one_hot/off_value*
axis€€€€€€€€€*
T0*
TI0*'
_output_shapes
:€€€€€€€€€
У
pred_q_a/mulMul:over_options/q_func/action_value/fully_connected_1/BiasAddpred_q_a/one_hot*
T0*'
_output_shapes
:€€€€€€€€€
e
#pred_q_a/pred_q_a/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
Ц
pred_q_a/pred_q_aSumpred_q_a/mul#pred_q_a/pred_q_a/reduction_indices*
	keep_dims( *
T0*

Tidx0*#
_output_shapes
:€€€€€€€€€
”
Ctarget_q_func/convnet/Conv/weights/Initializer/random_uniform/shapeConst*%
valueB"             *
dtype0*5
_class+
)'loc:@target_q_func/convnet/Conv/weights*
_output_shapes
:
љ
Atarget_q_func/convnet/Conv/weights/Initializer/random_uniform/minConst*
valueB
 *чьSљ*
dtype0*5
_class+
)'loc:@target_q_func/convnet/Conv/weights*
_output_shapes
: 
љ
Atarget_q_func/convnet/Conv/weights/Initializer/random_uniform/maxConst*
valueB
 *чьS=*
dtype0*5
_class+
)'loc:@target_q_func/convnet/Conv/weights*
_output_shapes
: 
ѓ
Ktarget_q_func/convnet/Conv/weights/Initializer/random_uniform/RandomUniformRandomUniformCtarget_q_func/convnet/Conv/weights/Initializer/random_uniform/shape*

seed *
seed2 *
dtype0*
T0*5
_class+
)'loc:@target_q_func/convnet/Conv/weights*&
_output_shapes
: 
¶
Atarget_q_func/convnet/Conv/weights/Initializer/random_uniform/subSubAtarget_q_func/convnet/Conv/weights/Initializer/random_uniform/maxAtarget_q_func/convnet/Conv/weights/Initializer/random_uniform/min*
T0*5
_class+
)'loc:@target_q_func/convnet/Conv/weights*
_output_shapes
: 
ј
Atarget_q_func/convnet/Conv/weights/Initializer/random_uniform/mulMulKtarget_q_func/convnet/Conv/weights/Initializer/random_uniform/RandomUniformAtarget_q_func/convnet/Conv/weights/Initializer/random_uniform/sub*
T0*5
_class+
)'loc:@target_q_func/convnet/Conv/weights*&
_output_shapes
: 
≤
=target_q_func/convnet/Conv/weights/Initializer/random_uniformAddAtarget_q_func/convnet/Conv/weights/Initializer/random_uniform/mulAtarget_q_func/convnet/Conv/weights/Initializer/random_uniform/min*
T0*5
_class+
)'loc:@target_q_func/convnet/Conv/weights*&
_output_shapes
: 
Ё
"target_q_func/convnet/Conv/weights
VariableV2*
shape: *
dtype0*
	container *
shared_name *5
_class+
)'loc:@target_q_func/convnet/Conv/weights*&
_output_shapes
: 
І
)target_q_func/convnet/Conv/weights/AssignAssign"target_q_func/convnet/Conv/weights=target_q_func/convnet/Conv/weights/Initializer/random_uniform*
T0*
validate_shape(*
use_locking(*5
_class+
)'loc:@target_q_func/convnet/Conv/weights*&
_output_shapes
: 
њ
'target_q_func/convnet/Conv/weights/readIdentity"target_q_func/convnet/Conv/weights*
T0*5
_class+
)'loc:@target_q_func/convnet/Conv/weights*&
_output_shapes
: 
ґ
3target_q_func/convnet/Conv/biases/Initializer/ConstConst*
valueB *    *
dtype0*4
_class*
(&loc:@target_q_func/convnet/Conv/biases*
_output_shapes
: 
√
!target_q_func/convnet/Conv/biases
VariableV2*
shape: *
dtype0*
	container *
shared_name *4
_class*
(&loc:@target_q_func/convnet/Conv/biases*
_output_shapes
: 
О
(target_q_func/convnet/Conv/biases/AssignAssign!target_q_func/convnet/Conv/biases3target_q_func/convnet/Conv/biases/Initializer/Const*
T0*
validate_shape(*
use_locking(*4
_class*
(&loc:@target_q_func/convnet/Conv/biases*
_output_shapes
: 
∞
&target_q_func/convnet/Conv/biases/readIdentity!target_q_func/convnet/Conv/biases*
T0*4
_class*
(&loc:@target_q_func/convnet/Conv/biases*
_output_shapes
: 
Е
,target_q_func/convnet/Conv/convolution/ShapeConst*%
valueB"             *
dtype0*
_output_shapes
:
Е
4target_q_func/convnet/Conv/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
ю
&target_q_func/convnet/Conv/convolutionConv2Dobs_tp1_ph/truediv'target_q_func/convnet/Conv/weights/read*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*/
_output_shapes
:€€€€€€€€€ 
ќ
"target_q_func/convnet/Conv/BiasAddBiasAdd&target_q_func/convnet/Conv/convolution&target_q_func/convnet/Conv/biases/read*
T0*
data_formatNHWC*/
_output_shapes
:€€€€€€€€€ 
Е
target_q_func/convnet/Conv/ReluRelu"target_q_func/convnet/Conv/BiasAdd*
T0*/
_output_shapes
:€€€€€€€€€ 
„
Etarget_q_func/convnet/Conv_1/weights/Initializer/random_uniform/shapeConst*%
valueB"          @   *
dtype0*7
_class-
+)loc:@target_q_func/convnet/Conv_1/weights*
_output_shapes
:
Ѕ
Ctarget_q_func/convnet/Conv_1/weights/Initializer/random_uniform/minConst*
valueB
 *  Аљ*
dtype0*7
_class-
+)loc:@target_q_func/convnet/Conv_1/weights*
_output_shapes
: 
Ѕ
Ctarget_q_func/convnet/Conv_1/weights/Initializer/random_uniform/maxConst*
valueB
 *  А=*
dtype0*7
_class-
+)loc:@target_q_func/convnet/Conv_1/weights*
_output_shapes
: 
µ
Mtarget_q_func/convnet/Conv_1/weights/Initializer/random_uniform/RandomUniformRandomUniformEtarget_q_func/convnet/Conv_1/weights/Initializer/random_uniform/shape*

seed *
seed2 *
dtype0*
T0*7
_class-
+)loc:@target_q_func/convnet/Conv_1/weights*&
_output_shapes
: @
Ѓ
Ctarget_q_func/convnet/Conv_1/weights/Initializer/random_uniform/subSubCtarget_q_func/convnet/Conv_1/weights/Initializer/random_uniform/maxCtarget_q_func/convnet/Conv_1/weights/Initializer/random_uniform/min*
T0*7
_class-
+)loc:@target_q_func/convnet/Conv_1/weights*
_output_shapes
: 
»
Ctarget_q_func/convnet/Conv_1/weights/Initializer/random_uniform/mulMulMtarget_q_func/convnet/Conv_1/weights/Initializer/random_uniform/RandomUniformCtarget_q_func/convnet/Conv_1/weights/Initializer/random_uniform/sub*
T0*7
_class-
+)loc:@target_q_func/convnet/Conv_1/weights*&
_output_shapes
: @
Ї
?target_q_func/convnet/Conv_1/weights/Initializer/random_uniformAddCtarget_q_func/convnet/Conv_1/weights/Initializer/random_uniform/mulCtarget_q_func/convnet/Conv_1/weights/Initializer/random_uniform/min*
T0*7
_class-
+)loc:@target_q_func/convnet/Conv_1/weights*&
_output_shapes
: @
б
$target_q_func/convnet/Conv_1/weights
VariableV2*
shape: @*
dtype0*
	container *
shared_name *7
_class-
+)loc:@target_q_func/convnet/Conv_1/weights*&
_output_shapes
: @
ѓ
+target_q_func/convnet/Conv_1/weights/AssignAssign$target_q_func/convnet/Conv_1/weights?target_q_func/convnet/Conv_1/weights/Initializer/random_uniform*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@target_q_func/convnet/Conv_1/weights*&
_output_shapes
: @
≈
)target_q_func/convnet/Conv_1/weights/readIdentity$target_q_func/convnet/Conv_1/weights*
T0*7
_class-
+)loc:@target_q_func/convnet/Conv_1/weights*&
_output_shapes
: @
Ї
5target_q_func/convnet/Conv_1/biases/Initializer/ConstConst*
valueB@*    *
dtype0*6
_class,
*(loc:@target_q_func/convnet/Conv_1/biases*
_output_shapes
:@
«
#target_q_func/convnet/Conv_1/biases
VariableV2*
shape:@*
dtype0*
	container *
shared_name *6
_class,
*(loc:@target_q_func/convnet/Conv_1/biases*
_output_shapes
:@
Ц
*target_q_func/convnet/Conv_1/biases/AssignAssign#target_q_func/convnet/Conv_1/biases5target_q_func/convnet/Conv_1/biases/Initializer/Const*
T0*
validate_shape(*
use_locking(*6
_class,
*(loc:@target_q_func/convnet/Conv_1/biases*
_output_shapes
:@
ґ
(target_q_func/convnet/Conv_1/biases/readIdentity#target_q_func/convnet/Conv_1/biases*
T0*6
_class,
*(loc:@target_q_func/convnet/Conv_1/biases*
_output_shapes
:@
З
.target_q_func/convnet/Conv_1/convolution/ShapeConst*%
valueB"          @   *
dtype0*
_output_shapes
:
З
6target_q_func/convnet/Conv_1/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
П
(target_q_func/convnet/Conv_1/convolutionConv2Dtarget_q_func/convnet/Conv/Relu)target_q_func/convnet/Conv_1/weights/read*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*/
_output_shapes
:€€€€€€€€€@
‘
$target_q_func/convnet/Conv_1/BiasAddBiasAdd(target_q_func/convnet/Conv_1/convolution(target_q_func/convnet/Conv_1/biases/read*
T0*
data_formatNHWC*/
_output_shapes
:€€€€€€€€€@
Й
!target_q_func/convnet/Conv_1/ReluRelu$target_q_func/convnet/Conv_1/BiasAdd*
T0*/
_output_shapes
:€€€€€€€€€@
„
Etarget_q_func/convnet/Conv_2/weights/Initializer/random_uniform/shapeConst*%
valueB"      @   @   *
dtype0*7
_class-
+)loc:@target_q_func/convnet/Conv_2/weights*
_output_shapes
:
Ѕ
Ctarget_q_func/convnet/Conv_2/weights/Initializer/random_uniform/minConst*
valueB
 *:ЌУљ*
dtype0*7
_class-
+)loc:@target_q_func/convnet/Conv_2/weights*
_output_shapes
: 
Ѕ
Ctarget_q_func/convnet/Conv_2/weights/Initializer/random_uniform/maxConst*
valueB
 *:ЌУ=*
dtype0*7
_class-
+)loc:@target_q_func/convnet/Conv_2/weights*
_output_shapes
: 
µ
Mtarget_q_func/convnet/Conv_2/weights/Initializer/random_uniform/RandomUniformRandomUniformEtarget_q_func/convnet/Conv_2/weights/Initializer/random_uniform/shape*

seed *
seed2 *
dtype0*
T0*7
_class-
+)loc:@target_q_func/convnet/Conv_2/weights*&
_output_shapes
:@@
Ѓ
Ctarget_q_func/convnet/Conv_2/weights/Initializer/random_uniform/subSubCtarget_q_func/convnet/Conv_2/weights/Initializer/random_uniform/maxCtarget_q_func/convnet/Conv_2/weights/Initializer/random_uniform/min*
T0*7
_class-
+)loc:@target_q_func/convnet/Conv_2/weights*
_output_shapes
: 
»
Ctarget_q_func/convnet/Conv_2/weights/Initializer/random_uniform/mulMulMtarget_q_func/convnet/Conv_2/weights/Initializer/random_uniform/RandomUniformCtarget_q_func/convnet/Conv_2/weights/Initializer/random_uniform/sub*
T0*7
_class-
+)loc:@target_q_func/convnet/Conv_2/weights*&
_output_shapes
:@@
Ї
?target_q_func/convnet/Conv_2/weights/Initializer/random_uniformAddCtarget_q_func/convnet/Conv_2/weights/Initializer/random_uniform/mulCtarget_q_func/convnet/Conv_2/weights/Initializer/random_uniform/min*
T0*7
_class-
+)loc:@target_q_func/convnet/Conv_2/weights*&
_output_shapes
:@@
б
$target_q_func/convnet/Conv_2/weights
VariableV2*
shape:@@*
dtype0*
	container *
shared_name *7
_class-
+)loc:@target_q_func/convnet/Conv_2/weights*&
_output_shapes
:@@
ѓ
+target_q_func/convnet/Conv_2/weights/AssignAssign$target_q_func/convnet/Conv_2/weights?target_q_func/convnet/Conv_2/weights/Initializer/random_uniform*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@target_q_func/convnet/Conv_2/weights*&
_output_shapes
:@@
≈
)target_q_func/convnet/Conv_2/weights/readIdentity$target_q_func/convnet/Conv_2/weights*
T0*7
_class-
+)loc:@target_q_func/convnet/Conv_2/weights*&
_output_shapes
:@@
Ї
5target_q_func/convnet/Conv_2/biases/Initializer/ConstConst*
valueB@*    *
dtype0*6
_class,
*(loc:@target_q_func/convnet/Conv_2/biases*
_output_shapes
:@
«
#target_q_func/convnet/Conv_2/biases
VariableV2*
shape:@*
dtype0*
	container *
shared_name *6
_class,
*(loc:@target_q_func/convnet/Conv_2/biases*
_output_shapes
:@
Ц
*target_q_func/convnet/Conv_2/biases/AssignAssign#target_q_func/convnet/Conv_2/biases5target_q_func/convnet/Conv_2/biases/Initializer/Const*
T0*
validate_shape(*
use_locking(*6
_class,
*(loc:@target_q_func/convnet/Conv_2/biases*
_output_shapes
:@
ґ
(target_q_func/convnet/Conv_2/biases/readIdentity#target_q_func/convnet/Conv_2/biases*
T0*6
_class,
*(loc:@target_q_func/convnet/Conv_2/biases*
_output_shapes
:@
З
.target_q_func/convnet/Conv_2/convolution/ShapeConst*%
valueB"      @   @   *
dtype0*
_output_shapes
:
З
6target_q_func/convnet/Conv_2/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
С
(target_q_func/convnet/Conv_2/convolutionConv2D!target_q_func/convnet/Conv_1/Relu)target_q_func/convnet/Conv_2/weights/read*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*/
_output_shapes
:€€€€€€€€€@
‘
$target_q_func/convnet/Conv_2/BiasAddBiasAdd(target_q_func/convnet/Conv_2/convolution(target_q_func/convnet/Conv_2/biases/read*
T0*
data_formatNHWC*/
_output_shapes
:€€€€€€€€€@
Й
!target_q_func/convnet/Conv_2/ReluRelu$target_q_func/convnet/Conv_2/BiasAdd*
T0*/
_output_shapes
:€€€€€€€€€@
|
target_q_func/Flatten/ShapeShape!target_q_func/convnet/Conv_2/Relu*
T0*
out_type0*
_output_shapes
:
k
!target_q_func/Flatten/Slice/beginConst*
valueB: *
dtype0*
_output_shapes
:
j
 target_q_func/Flatten/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
Є
target_q_func/Flatten/SliceSlicetarget_q_func/Flatten/Shape!target_q_func/Flatten/Slice/begin target_q_func/Flatten/Slice/size*
T0*
Index0*
_output_shapes
:
m
#target_q_func/Flatten/Slice_1/beginConst*
valueB:*
dtype0*
_output_shapes
:
l
"target_q_func/Flatten/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
Њ
target_q_func/Flatten/Slice_1Slicetarget_q_func/Flatten/Shape#target_q_func/Flatten/Slice_1/begin"target_q_func/Flatten/Slice_1/size*
T0*
Index0*
_output_shapes
:
e
target_q_func/Flatten/ConstConst*
valueB: *
dtype0*
_output_shapes
:
Ь
target_q_func/Flatten/ProdProdtarget_q_func/Flatten/Slice_1target_q_func/Flatten/Const*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
f
$target_q_func/Flatten/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
°
 target_q_func/Flatten/ExpandDims
ExpandDimstarget_q_func/Flatten/Prod$target_q_func/Flatten/ExpandDims/dim*
T0*

Tdim0*
_output_shapes
:
c
!target_q_func/Flatten/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ƒ
target_q_func/Flatten/concatConcatV2target_q_func/Flatten/Slice target_q_func/Flatten/ExpandDims!target_q_func/Flatten/concat/axis*
N*
T0*

Tidx0*
_output_shapes
:
™
target_q_func/Flatten/ReshapeReshape!target_q_func/convnet/Conv_2/Relutarget_q_func/Flatten/concat*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
л
Starget_q_func/action_value/fully_connected/weights/Initializer/random_uniform/shapeConst*
valueB"      *
dtype0*E
_class;
97loc:@target_q_func/action_value/fully_connected/weights*
_output_shapes
:
Ё
Qtarget_q_func/action_value/fully_connected/weights/Initializer/random_uniform/minConst*
valueB
 *„≥Ёљ*
dtype0*E
_class;
97loc:@target_q_func/action_value/fully_connected/weights*
_output_shapes
: 
Ё
Qtarget_q_func/action_value/fully_connected/weights/Initializer/random_uniform/maxConst*
valueB
 *„≥Ё=*
dtype0*E
_class;
97loc:@target_q_func/action_value/fully_connected/weights*
_output_shapes
: 
ў
[target_q_func/action_value/fully_connected/weights/Initializer/random_uniform/RandomUniformRandomUniformStarget_q_func/action_value/fully_connected/weights/Initializer/random_uniform/shape*

seed *
seed2 *
dtype0*
T0*E
_class;
97loc:@target_q_func/action_value/fully_connected/weights* 
_output_shapes
:
АА
ж
Qtarget_q_func/action_value/fully_connected/weights/Initializer/random_uniform/subSubQtarget_q_func/action_value/fully_connected/weights/Initializer/random_uniform/maxQtarget_q_func/action_value/fully_connected/weights/Initializer/random_uniform/min*
T0*E
_class;
97loc:@target_q_func/action_value/fully_connected/weights*
_output_shapes
: 
ъ
Qtarget_q_func/action_value/fully_connected/weights/Initializer/random_uniform/mulMul[target_q_func/action_value/fully_connected/weights/Initializer/random_uniform/RandomUniformQtarget_q_func/action_value/fully_connected/weights/Initializer/random_uniform/sub*
T0*E
_class;
97loc:@target_q_func/action_value/fully_connected/weights* 
_output_shapes
:
АА
м
Mtarget_q_func/action_value/fully_connected/weights/Initializer/random_uniformAddQtarget_q_func/action_value/fully_connected/weights/Initializer/random_uniform/mulQtarget_q_func/action_value/fully_connected/weights/Initializer/random_uniform/min*
T0*E
_class;
97loc:@target_q_func/action_value/fully_connected/weights* 
_output_shapes
:
АА
с
2target_q_func/action_value/fully_connected/weights
VariableV2*
shape:
АА*
dtype0*
	container *
shared_name *E
_class;
97loc:@target_q_func/action_value/fully_connected/weights* 
_output_shapes
:
АА
б
9target_q_func/action_value/fully_connected/weights/AssignAssign2target_q_func/action_value/fully_connected/weightsMtarget_q_func/action_value/fully_connected/weights/Initializer/random_uniform*
T0*
validate_shape(*
use_locking(*E
_class;
97loc:@target_q_func/action_value/fully_connected/weights* 
_output_shapes
:
АА
й
7target_q_func/action_value/fully_connected/weights/readIdentity2target_q_func/action_value/fully_connected/weights*
T0*E
_class;
97loc:@target_q_func/action_value/fully_connected/weights* 
_output_shapes
:
АА
Ў
Ctarget_q_func/action_value/fully_connected/biases/Initializer/ConstConst*
valueBА*    *
dtype0*D
_class:
86loc:@target_q_func/action_value/fully_connected/biases*
_output_shapes	
:А
е
1target_q_func/action_value/fully_connected/biases
VariableV2*
shape:А*
dtype0*
	container *
shared_name *D
_class:
86loc:@target_q_func/action_value/fully_connected/biases*
_output_shapes	
:А
ѕ
8target_q_func/action_value/fully_connected/biases/AssignAssign1target_q_func/action_value/fully_connected/biasesCtarget_q_func/action_value/fully_connected/biases/Initializer/Const*
T0*
validate_shape(*
use_locking(*D
_class:
86loc:@target_q_func/action_value/fully_connected/biases*
_output_shapes	
:А
б
6target_q_func/action_value/fully_connected/biases/readIdentity1target_q_func/action_value/fully_connected/biases*
T0*D
_class:
86loc:@target_q_func/action_value/fully_connected/biases*
_output_shapes	
:А
м
1target_q_func/action_value/fully_connected/MatMulMatMultarget_q_func/Flatten/Reshape7target_q_func/action_value/fully_connected/weights/read*
transpose_a( *
transpose_b( *
T0*(
_output_shapes
:€€€€€€€€€А
т
2target_q_func/action_value/fully_connected/BiasAddBiasAdd1target_q_func/action_value/fully_connected/MatMul6target_q_func/action_value/fully_connected/biases/read*
T0*
data_formatNHWC*(
_output_shapes
:€€€€€€€€€А
Ю
/target_q_func/action_value/fully_connected/ReluRelu2target_q_func/action_value/fully_connected/BiasAdd*
T0*(
_output_shapes
:€€€€€€€€€А
п
Utarget_q_func/action_value/fully_connected_1/weights/Initializer/random_uniform/shapeConst*
valueB"      *
dtype0*G
_class=
;9loc:@target_q_func/action_value/fully_connected_1/weights*
_output_shapes
:
б
Starget_q_func/action_value/fully_connected_1/weights/Initializer/random_uniform/minConst*
valueB
 *≤_Њ*
dtype0*G
_class=
;9loc:@target_q_func/action_value/fully_connected_1/weights*
_output_shapes
: 
б
Starget_q_func/action_value/fully_connected_1/weights/Initializer/random_uniform/maxConst*
valueB
 *≤_>*
dtype0*G
_class=
;9loc:@target_q_func/action_value/fully_connected_1/weights*
_output_shapes
: 
ё
]target_q_func/action_value/fully_connected_1/weights/Initializer/random_uniform/RandomUniformRandomUniformUtarget_q_func/action_value/fully_connected_1/weights/Initializer/random_uniform/shape*

seed *
seed2 *
dtype0*
T0*G
_class=
;9loc:@target_q_func/action_value/fully_connected_1/weights*
_output_shapes
:	А
о
Starget_q_func/action_value/fully_connected_1/weights/Initializer/random_uniform/subSubStarget_q_func/action_value/fully_connected_1/weights/Initializer/random_uniform/maxStarget_q_func/action_value/fully_connected_1/weights/Initializer/random_uniform/min*
T0*G
_class=
;9loc:@target_q_func/action_value/fully_connected_1/weights*
_output_shapes
: 
Б
Starget_q_func/action_value/fully_connected_1/weights/Initializer/random_uniform/mulMul]target_q_func/action_value/fully_connected_1/weights/Initializer/random_uniform/RandomUniformStarget_q_func/action_value/fully_connected_1/weights/Initializer/random_uniform/sub*
T0*G
_class=
;9loc:@target_q_func/action_value/fully_connected_1/weights*
_output_shapes
:	А
у
Otarget_q_func/action_value/fully_connected_1/weights/Initializer/random_uniformAddStarget_q_func/action_value/fully_connected_1/weights/Initializer/random_uniform/mulStarget_q_func/action_value/fully_connected_1/weights/Initializer/random_uniform/min*
T0*G
_class=
;9loc:@target_q_func/action_value/fully_connected_1/weights*
_output_shapes
:	А
у
4target_q_func/action_value/fully_connected_1/weights
VariableV2*
shape:	А*
dtype0*
	container *
shared_name *G
_class=
;9loc:@target_q_func/action_value/fully_connected_1/weights*
_output_shapes
:	А
и
;target_q_func/action_value/fully_connected_1/weights/AssignAssign4target_q_func/action_value/fully_connected_1/weightsOtarget_q_func/action_value/fully_connected_1/weights/Initializer/random_uniform*
T0*
validate_shape(*
use_locking(*G
_class=
;9loc:@target_q_func/action_value/fully_connected_1/weights*
_output_shapes
:	А
о
9target_q_func/action_value/fully_connected_1/weights/readIdentity4target_q_func/action_value/fully_connected_1/weights*
T0*G
_class=
;9loc:@target_q_func/action_value/fully_connected_1/weights*
_output_shapes
:	А
Џ
Etarget_q_func/action_value/fully_connected_1/biases/Initializer/ConstConst*
valueB*    *
dtype0*F
_class<
:8loc:@target_q_func/action_value/fully_connected_1/biases*
_output_shapes
:
з
3target_q_func/action_value/fully_connected_1/biases
VariableV2*
shape:*
dtype0*
	container *
shared_name *F
_class<
:8loc:@target_q_func/action_value/fully_connected_1/biases*
_output_shapes
:
÷
:target_q_func/action_value/fully_connected_1/biases/AssignAssign3target_q_func/action_value/fully_connected_1/biasesEtarget_q_func/action_value/fully_connected_1/biases/Initializer/Const*
T0*
validate_shape(*
use_locking(*F
_class<
:8loc:@target_q_func/action_value/fully_connected_1/biases*
_output_shapes
:
ж
8target_q_func/action_value/fully_connected_1/biases/readIdentity3target_q_func/action_value/fully_connected_1/biases*
T0*F
_class<
:8loc:@target_q_func/action_value/fully_connected_1/biases*
_output_shapes
:
Б
3target_q_func/action_value/fully_connected_1/MatMulMatMul/target_q_func/action_value/fully_connected/Relu9target_q_func/action_value/fully_connected_1/weights/read*
transpose_a( *
transpose_b( *
T0*'
_output_shapes
:€€€€€€€€€
ч
4target_q_func/action_value/fully_connected_1/BiasAddBiasAdd3target_q_func/action_value/fully_connected_1/MatMul8target_q_func/action_value/fully_connected_1/biases/read*
T0*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€
U
target_q_a/sub/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
c
target_q_a/subSubtarget_q_a/sub/xdone_mask_ph*
T0*#
_output_shapes
:€€€€€€€€€
U
target_q_a/Pow/xConst*
valueB
 *§p}?*
dtype0*
_output_shapes
: 
`
target_q_a/PowPowtarget_q_a/Pow/x	opt_steps*
T0*#
_output_shapes
:€€€€€€€€€
c
target_q_a/mulMultarget_q_a/subtarget_q_a/Pow*
T0*#
_output_shapes
:€€€€€€€€€
b
 target_q_a/Max/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
Є
target_q_a/MaxMax4target_q_func/action_value/fully_connected_1/BiasAdd target_q_a/Max/reduction_indices*
	keep_dims( *
T0*

Tidx0*#
_output_shapes
:€€€€€€€€€
e
target_q_a/mul_1Multarget_q_a/multarget_q_a/Max*
T0*#
_output_shapes
:€€€€€€€€€
_
target_q_a/addAddrew_t_phtarget_q_a/mul_1*
T0*#
_output_shapes
:€€€€€€€€€
p
"Compute_bellman_error/StopGradientStopGradienttarget_q_a/add*
T0*#
_output_shapes
:€€€€€€€€€
Е
Compute_bellman_error/subSubpred_q_a/pred_q_a"Compute_bellman_error/StopGradient*
T0*#
_output_shapes
:€€€€€€€€€
i
Compute_bellman_error/AbsAbsCompute_bellman_error/sub*
T0*#
_output_shapes
:€€€€€€€€€
a
Compute_bellman_error/Less/yConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Й
Compute_bellman_error/LessLessCompute_bellman_error/AbsCompute_bellman_error/Less/y*
T0*#
_output_shapes
:€€€€€€€€€
o
Compute_bellman_error/SquareSquareCompute_bellman_error/sub*
T0*#
_output_shapes
:€€€€€€€€€
`
Compute_bellman_error/mul/yConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
Й
Compute_bellman_error/mulMulCompute_bellman_error/SquareCompute_bellman_error/mul/y*
T0*#
_output_shapes
:€€€€€€€€€
k
Compute_bellman_error/Abs_1AbsCompute_bellman_error/sub*
T0*#
_output_shapes
:€€€€€€€€€
b
Compute_bellman_error/sub_1/yConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
М
Compute_bellman_error/sub_1SubCompute_bellman_error/Abs_1Compute_bellman_error/sub_1/y*
T0*#
_output_shapes
:€€€€€€€€€
b
Compute_bellman_error/mul_1/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
М
Compute_bellman_error/mul_1MulCompute_bellman_error/mul_1/xCompute_bellman_error/sub_1*
T0*#
_output_shapes
:€€€€€€€€€
®
Compute_bellman_error/SelectSelectCompute_bellman_error/LessCompute_bellman_error/mulCompute_bellman_error/mul_1*
T0*#
_output_shapes
:€€€€€€€€€
e
Compute_bellman_error/ConstConst*
valueB: *
dtype0*
_output_shapes
:
°
!Compute_bellman_error/total_errorSumCompute_bellman_error/SelectCompute_bellman_error/Const*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
N
learning_ratePlaceholder*
dtype0*
shape: *
_output_shapes
: 
\
Optimizer/gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
^
Optimizer/gradients/ConstConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
w
Optimizer/gradients/FillFillOptimizer/gradients/ShapeOptimizer/gradients/Const*
T0*
_output_shapes
: 
Т
HOptimizer/gradients/Compute_bellman_error/total_error_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
д
BOptimizer/gradients/Compute_bellman_error/total_error_grad/ReshapeReshapeOptimizer/gradients/FillHOptimizer/gradients/Compute_bellman_error/total_error_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
Ь
@Optimizer/gradients/Compute_bellman_error/total_error_grad/ShapeShapeCompute_bellman_error/Select*
T0*
out_type0*
_output_shapes
:
Н
?Optimizer/gradients/Compute_bellman_error/total_error_grad/TileTileBOptimizer/gradients/Compute_bellman_error/total_error_grad/Reshape@Optimizer/gradients/Compute_bellman_error/total_error_grad/Shape*
T0*

Tmultiples0*#
_output_shapes
:€€€€€€€€€
Ц
@Optimizer/gradients/Compute_bellman_error/Select_grad/zeros_like	ZerosLikeCompute_bellman_error/mul*
T0*#
_output_shapes
:€€€€€€€€€
У
<Optimizer/gradients/Compute_bellman_error/Select_grad/SelectSelectCompute_bellman_error/Less?Optimizer/gradients/Compute_bellman_error/total_error_grad/Tile@Optimizer/gradients/Compute_bellman_error/Select_grad/zeros_like*
T0*#
_output_shapes
:€€€€€€€€€
Х
>Optimizer/gradients/Compute_bellman_error/Select_grad/Select_1SelectCompute_bellman_error/Less@Optimizer/gradients/Compute_bellman_error/Select_grad/zeros_like?Optimizer/gradients/Compute_bellman_error/total_error_grad/Tile*
T0*#
_output_shapes
:€€€€€€€€€
ќ
FOptimizer/gradients/Compute_bellman_error/Select_grad/tuple/group_depsNoOp=^Optimizer/gradients/Compute_bellman_error/Select_grad/Select?^Optimizer/gradients/Compute_bellman_error/Select_grad/Select_1
а
NOptimizer/gradients/Compute_bellman_error/Select_grad/tuple/control_dependencyIdentity<Optimizer/gradients/Compute_bellman_error/Select_grad/SelectG^Optimizer/gradients/Compute_bellman_error/Select_grad/tuple/group_deps*
T0*O
_classE
CAloc:@Optimizer/gradients/Compute_bellman_error/Select_grad/Select*#
_output_shapes
:€€€€€€€€€
ж
POptimizer/gradients/Compute_bellman_error/Select_grad/tuple/control_dependency_1Identity>Optimizer/gradients/Compute_bellman_error/Select_grad/Select_1G^Optimizer/gradients/Compute_bellman_error/Select_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@Optimizer/gradients/Compute_bellman_error/Select_grad/Select_1*#
_output_shapes
:€€€€€€€€€
Ф
8Optimizer/gradients/Compute_bellman_error/mul_grad/ShapeShapeCompute_bellman_error/Square*
T0*
out_type0*
_output_shapes
:
}
:Optimizer/gradients/Compute_bellman_error/mul_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
Ф
HOptimizer/gradients/Compute_bellman_error/mul_grad/BroadcastGradientArgsBroadcastGradientArgs8Optimizer/gradients/Compute_bellman_error/mul_grad/Shape:Optimizer/gradients/Compute_bellman_error/mul_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Ў
6Optimizer/gradients/Compute_bellman_error/mul_grad/mulMulNOptimizer/gradients/Compute_bellman_error/Select_grad/tuple/control_dependencyCompute_bellman_error/mul/y*
T0*#
_output_shapes
:€€€€€€€€€
€
6Optimizer/gradients/Compute_bellman_error/mul_grad/SumSum6Optimizer/gradients/Compute_bellman_error/mul_grad/mulHOptimizer/gradients/Compute_bellman_error/mul_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
у
:Optimizer/gradients/Compute_bellman_error/mul_grad/ReshapeReshape6Optimizer/gradients/Compute_bellman_error/mul_grad/Sum8Optimizer/gradients/Compute_bellman_error/mul_grad/Shape*
T0*
Tshape0*#
_output_shapes
:€€€€€€€€€
џ
8Optimizer/gradients/Compute_bellman_error/mul_grad/mul_1MulCompute_bellman_error/SquareNOptimizer/gradients/Compute_bellman_error/Select_grad/tuple/control_dependency*
T0*#
_output_shapes
:€€€€€€€€€
Е
8Optimizer/gradients/Compute_bellman_error/mul_grad/Sum_1Sum8Optimizer/gradients/Compute_bellman_error/mul_grad/mul_1JOptimizer/gradients/Compute_bellman_error/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
м
<Optimizer/gradients/Compute_bellman_error/mul_grad/Reshape_1Reshape8Optimizer/gradients/Compute_bellman_error/mul_grad/Sum_1:Optimizer/gradients/Compute_bellman_error/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
«
COptimizer/gradients/Compute_bellman_error/mul_grad/tuple/group_depsNoOp;^Optimizer/gradients/Compute_bellman_error/mul_grad/Reshape=^Optimizer/gradients/Compute_bellman_error/mul_grad/Reshape_1
÷
KOptimizer/gradients/Compute_bellman_error/mul_grad/tuple/control_dependencyIdentity:Optimizer/gradients/Compute_bellman_error/mul_grad/ReshapeD^Optimizer/gradients/Compute_bellman_error/mul_grad/tuple/group_deps*
T0*M
_classC
A?loc:@Optimizer/gradients/Compute_bellman_error/mul_grad/Reshape*#
_output_shapes
:€€€€€€€€€
ѕ
MOptimizer/gradients/Compute_bellman_error/mul_grad/tuple/control_dependency_1Identity<Optimizer/gradients/Compute_bellman_error/mul_grad/Reshape_1D^Optimizer/gradients/Compute_bellman_error/mul_grad/tuple/group_deps*
T0*O
_classE
CAloc:@Optimizer/gradients/Compute_bellman_error/mul_grad/Reshape_1*
_output_shapes
: 
}
:Optimizer/gradients/Compute_bellman_error/mul_1_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Ч
<Optimizer/gradients/Compute_bellman_error/mul_1_grad/Shape_1ShapeCompute_bellman_error/sub_1*
T0*
out_type0*
_output_shapes
:
Ъ
JOptimizer/gradients/Compute_bellman_error/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs:Optimizer/gradients/Compute_bellman_error/mul_1_grad/Shape<Optimizer/gradients/Compute_bellman_error/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
№
8Optimizer/gradients/Compute_bellman_error/mul_1_grad/mulMulPOptimizer/gradients/Compute_bellman_error/Select_grad/tuple/control_dependency_1Compute_bellman_error/sub_1*
T0*#
_output_shapes
:€€€€€€€€€
Е
8Optimizer/gradients/Compute_bellman_error/mul_1_grad/SumSum8Optimizer/gradients/Compute_bellman_error/mul_1_grad/mulJOptimizer/gradients/Compute_bellman_error/mul_1_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
м
<Optimizer/gradients/Compute_bellman_error/mul_1_grad/ReshapeReshape8Optimizer/gradients/Compute_bellman_error/mul_1_grad/Sum:Optimizer/gradients/Compute_bellman_error/mul_1_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
а
:Optimizer/gradients/Compute_bellman_error/mul_1_grad/mul_1MulCompute_bellman_error/mul_1/xPOptimizer/gradients/Compute_bellman_error/Select_grad/tuple/control_dependency_1*
T0*#
_output_shapes
:€€€€€€€€€
Л
:Optimizer/gradients/Compute_bellman_error/mul_1_grad/Sum_1Sum:Optimizer/gradients/Compute_bellman_error/mul_1_grad/mul_1LOptimizer/gradients/Compute_bellman_error/mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
€
>Optimizer/gradients/Compute_bellman_error/mul_1_grad/Reshape_1Reshape:Optimizer/gradients/Compute_bellman_error/mul_1_grad/Sum_1<Optimizer/gradients/Compute_bellman_error/mul_1_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:€€€€€€€€€
Ќ
EOptimizer/gradients/Compute_bellman_error/mul_1_grad/tuple/group_depsNoOp=^Optimizer/gradients/Compute_bellman_error/mul_1_grad/Reshape?^Optimizer/gradients/Compute_bellman_error/mul_1_grad/Reshape_1
—
MOptimizer/gradients/Compute_bellman_error/mul_1_grad/tuple/control_dependencyIdentity<Optimizer/gradients/Compute_bellman_error/mul_1_grad/ReshapeF^Optimizer/gradients/Compute_bellman_error/mul_1_grad/tuple/group_deps*
T0*O
_classE
CAloc:@Optimizer/gradients/Compute_bellman_error/mul_1_grad/Reshape*
_output_shapes
: 
д
OOptimizer/gradients/Compute_bellman_error/mul_1_grad/tuple/control_dependency_1Identity>Optimizer/gradients/Compute_bellman_error/mul_1_grad/Reshape_1F^Optimizer/gradients/Compute_bellman_error/mul_1_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@Optimizer/gradients/Compute_bellman_error/mul_1_grad/Reshape_1*#
_output_shapes
:€€€€€€€€€
ќ
;Optimizer/gradients/Compute_bellman_error/Square_grad/mul/xConstL^Optimizer/gradients/Compute_bellman_error/mul_grad/tuple/control_dependency*
valueB
 *   @*
dtype0*
_output_shapes
: 
∆
9Optimizer/gradients/Compute_bellman_error/Square_grad/mulMul;Optimizer/gradients/Compute_bellman_error/Square_grad/mul/xCompute_bellman_error/sub*
T0*#
_output_shapes
:€€€€€€€€€
ш
;Optimizer/gradients/Compute_bellman_error/Square_grad/mul_1MulKOptimizer/gradients/Compute_bellman_error/mul_grad/tuple/control_dependency9Optimizer/gradients/Compute_bellman_error/Square_grad/mul*
T0*#
_output_shapes
:€€€€€€€€€
Х
:Optimizer/gradients/Compute_bellman_error/sub_1_grad/ShapeShapeCompute_bellman_error/Abs_1*
T0*
out_type0*
_output_shapes
:

<Optimizer/gradients/Compute_bellman_error/sub_1_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
Ъ
JOptimizer/gradients/Compute_bellman_error/sub_1_grad/BroadcastGradientArgsBroadcastGradientArgs:Optimizer/gradients/Compute_bellman_error/sub_1_grad/Shape<Optimizer/gradients/Compute_bellman_error/sub_1_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Ь
8Optimizer/gradients/Compute_bellman_error/sub_1_grad/SumSumOOptimizer/gradients/Compute_bellman_error/mul_1_grad/tuple/control_dependency_1JOptimizer/gradients/Compute_bellman_error/sub_1_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
щ
<Optimizer/gradients/Compute_bellman_error/sub_1_grad/ReshapeReshape8Optimizer/gradients/Compute_bellman_error/sub_1_grad/Sum:Optimizer/gradients/Compute_bellman_error/sub_1_grad/Shape*
T0*
Tshape0*#
_output_shapes
:€€€€€€€€€
†
:Optimizer/gradients/Compute_bellman_error/sub_1_grad/Sum_1SumOOptimizer/gradients/Compute_bellman_error/mul_1_grad/tuple/control_dependency_1LOptimizer/gradients/Compute_bellman_error/sub_1_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
Ю
8Optimizer/gradients/Compute_bellman_error/sub_1_grad/NegNeg:Optimizer/gradients/Compute_bellman_error/sub_1_grad/Sum_1*
T0*
_output_shapes
:
р
>Optimizer/gradients/Compute_bellman_error/sub_1_grad/Reshape_1Reshape8Optimizer/gradients/Compute_bellman_error/sub_1_grad/Neg<Optimizer/gradients/Compute_bellman_error/sub_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
Ќ
EOptimizer/gradients/Compute_bellman_error/sub_1_grad/tuple/group_depsNoOp=^Optimizer/gradients/Compute_bellman_error/sub_1_grad/Reshape?^Optimizer/gradients/Compute_bellman_error/sub_1_grad/Reshape_1
ё
MOptimizer/gradients/Compute_bellman_error/sub_1_grad/tuple/control_dependencyIdentity<Optimizer/gradients/Compute_bellman_error/sub_1_grad/ReshapeF^Optimizer/gradients/Compute_bellman_error/sub_1_grad/tuple/group_deps*
T0*O
_classE
CAloc:@Optimizer/gradients/Compute_bellman_error/sub_1_grad/Reshape*#
_output_shapes
:€€€€€€€€€
„
OOptimizer/gradients/Compute_bellman_error/sub_1_grad/tuple/control_dependency_1Identity>Optimizer/gradients/Compute_bellman_error/sub_1_grad/Reshape_1F^Optimizer/gradients/Compute_bellman_error/sub_1_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@Optimizer/gradients/Compute_bellman_error/sub_1_grad/Reshape_1*
_output_shapes
: 
К
9Optimizer/gradients/Compute_bellman_error/Abs_1_grad/SignSignCompute_bellman_error/sub*
T0*#
_output_shapes
:€€€€€€€€€
ч
8Optimizer/gradients/Compute_bellman_error/Abs_1_grad/mulMulMOptimizer/gradients/Compute_bellman_error/sub_1_grad/tuple/control_dependency9Optimizer/gradients/Compute_bellman_error/Abs_1_grad/Sign*
T0*#
_output_shapes
:€€€€€€€€€
Ю
Optimizer/gradients/AddNAddN;Optimizer/gradients/Compute_bellman_error/Square_grad/mul_18Optimizer/gradients/Compute_bellman_error/Abs_1_grad/mul*
N*
T0*N
_classD
B@loc:@Optimizer/gradients/Compute_bellman_error/Square_grad/mul_1*#
_output_shapes
:€€€€€€€€€
Й
8Optimizer/gradients/Compute_bellman_error/sub_grad/ShapeShapepred_q_a/pred_q_a*
T0*
out_type0*
_output_shapes
:
Ь
:Optimizer/gradients/Compute_bellman_error/sub_grad/Shape_1Shape"Compute_bellman_error/StopGradient*
T0*
out_type0*
_output_shapes
:
Ф
HOptimizer/gradients/Compute_bellman_error/sub_grad/BroadcastGradientArgsBroadcastGradientArgs8Optimizer/gradients/Compute_bellman_error/sub_grad/Shape:Optimizer/gradients/Compute_bellman_error/sub_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
б
6Optimizer/gradients/Compute_bellman_error/sub_grad/SumSumOptimizer/gradients/AddNHOptimizer/gradients/Compute_bellman_error/sub_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
у
:Optimizer/gradients/Compute_bellman_error/sub_grad/ReshapeReshape6Optimizer/gradients/Compute_bellman_error/sub_grad/Sum8Optimizer/gradients/Compute_bellman_error/sub_grad/Shape*
T0*
Tshape0*#
_output_shapes
:€€€€€€€€€
е
8Optimizer/gradients/Compute_bellman_error/sub_grad/Sum_1SumOptimizer/gradients/AddNJOptimizer/gradients/Compute_bellman_error/sub_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
Ъ
6Optimizer/gradients/Compute_bellman_error/sub_grad/NegNeg8Optimizer/gradients/Compute_bellman_error/sub_grad/Sum_1*
T0*
_output_shapes
:
ч
<Optimizer/gradients/Compute_bellman_error/sub_grad/Reshape_1Reshape6Optimizer/gradients/Compute_bellman_error/sub_grad/Neg:Optimizer/gradients/Compute_bellman_error/sub_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:€€€€€€€€€
«
COptimizer/gradients/Compute_bellman_error/sub_grad/tuple/group_depsNoOp;^Optimizer/gradients/Compute_bellman_error/sub_grad/Reshape=^Optimizer/gradients/Compute_bellman_error/sub_grad/Reshape_1
÷
KOptimizer/gradients/Compute_bellman_error/sub_grad/tuple/control_dependencyIdentity:Optimizer/gradients/Compute_bellman_error/sub_grad/ReshapeD^Optimizer/gradients/Compute_bellman_error/sub_grad/tuple/group_deps*
T0*M
_classC
A?loc:@Optimizer/gradients/Compute_bellman_error/sub_grad/Reshape*#
_output_shapes
:€€€€€€€€€
№
MOptimizer/gradients/Compute_bellman_error/sub_grad/tuple/control_dependency_1Identity<Optimizer/gradients/Compute_bellman_error/sub_grad/Reshape_1D^Optimizer/gradients/Compute_bellman_error/sub_grad/tuple/group_deps*
T0*O
_classE
CAloc:@Optimizer/gradients/Compute_bellman_error/sub_grad/Reshape_1*#
_output_shapes
:€€€€€€€€€
|
0Optimizer/gradients/pred_q_a/pred_q_a_grad/ShapeShapepred_q_a/mul*
T0*
out_type0*
_output_shapes
:
q
/Optimizer/gradients/pred_q_a/pred_q_a_grad/SizeConst*
value	B :*
dtype0*
_output_shapes
: 
ђ
.Optimizer/gradients/pred_q_a/pred_q_a_grad/addAdd#pred_q_a/pred_q_a/reduction_indices/Optimizer/gradients/pred_q_a/pred_q_a_grad/Size*
T0*
_output_shapes
: 
Љ
.Optimizer/gradients/pred_q_a/pred_q_a_grad/modFloorMod.Optimizer/gradients/pred_q_a/pred_q_a_grad/add/Optimizer/gradients/pred_q_a/pred_q_a_grad/Size*
T0*
_output_shapes
: 
u
2Optimizer/gradients/pred_q_a/pred_q_a_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
x
6Optimizer/gradients/pred_q_a/pred_q_a_grad/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
x
6Optimizer/gradients/pred_q_a/pred_q_a_grad/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
В
0Optimizer/gradients/pred_q_a/pred_q_a_grad/rangeRange6Optimizer/gradients/pred_q_a/pred_q_a_grad/range/start/Optimizer/gradients/pred_q_a/pred_q_a_grad/Size6Optimizer/gradients/pred_q_a/pred_q_a_grad/range/delta*

Tidx0*
_output_shapes
:
w
5Optimizer/gradients/pred_q_a/pred_q_a_grad/Fill/valueConst*
value	B :*
dtype0*
_output_shapes
: 
√
/Optimizer/gradients/pred_q_a/pred_q_a_grad/FillFill2Optimizer/gradients/pred_q_a/pred_q_a_grad/Shape_15Optimizer/gradients/pred_q_a/pred_q_a_grad/Fill/value*
T0*
_output_shapes
: 
≈
8Optimizer/gradients/pred_q_a/pred_q_a_grad/DynamicStitchDynamicStitch0Optimizer/gradients/pred_q_a/pred_q_a_grad/range.Optimizer/gradients/pred_q_a/pred_q_a_grad/mod0Optimizer/gradients/pred_q_a/pred_q_a_grad/Shape/Optimizer/gradients/pred_q_a/pred_q_a_grad/Fill*
N*
T0*#
_output_shapes
:€€€€€€€€€
v
4Optimizer/gradients/pred_q_a/pred_q_a_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
џ
2Optimizer/gradients/pred_q_a/pred_q_a_grad/MaximumMaximum8Optimizer/gradients/pred_q_a/pred_q_a_grad/DynamicStitch4Optimizer/gradients/pred_q_a/pred_q_a_grad/Maximum/y*
T0*#
_output_shapes
:€€€€€€€€€
 
3Optimizer/gradients/pred_q_a/pred_q_a_grad/floordivFloorDiv0Optimizer/gradients/pred_q_a/pred_q_a_grad/Shape2Optimizer/gradients/pred_q_a/pred_q_a_grad/Maximum*
T0*
_output_shapes
:
х
2Optimizer/gradients/pred_q_a/pred_q_a_grad/ReshapeReshapeKOptimizer/gradients/Compute_bellman_error/sub_grad/tuple/control_dependency8Optimizer/gradients/pred_q_a/pred_q_a_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
д
/Optimizer/gradients/pred_q_a/pred_q_a_grad/TileTile2Optimizer/gradients/pred_q_a/pred_q_a_grad/Reshape3Optimizer/gradients/pred_q_a/pred_q_a_grad/floordiv*
T0*

Tmultiples0*'
_output_shapes
:€€€€€€€€€
•
+Optimizer/gradients/pred_q_a/mul_grad/ShapeShape:over_options/q_func/action_value/fully_connected_1/BiasAdd*
T0*
out_type0*
_output_shapes
:
}
-Optimizer/gradients/pred_q_a/mul_grad/Shape_1Shapepred_q_a/one_hot*
T0*
out_type0*
_output_shapes
:
н
;Optimizer/gradients/pred_q_a/mul_grad/BroadcastGradientArgsBroadcastGradientArgs+Optimizer/gradients/pred_q_a/mul_grad/Shape-Optimizer/gradients/pred_q_a/mul_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
•
)Optimizer/gradients/pred_q_a/mul_grad/mulMul/Optimizer/gradients/pred_q_a/pred_q_a_grad/Tilepred_q_a/one_hot*
T0*'
_output_shapes
:€€€€€€€€€
Ў
)Optimizer/gradients/pred_q_a/mul_grad/SumSum)Optimizer/gradients/pred_q_a/mul_grad/mul;Optimizer/gradients/pred_q_a/mul_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
–
-Optimizer/gradients/pred_q_a/mul_grad/ReshapeReshape)Optimizer/gradients/pred_q_a/mul_grad/Sum+Optimizer/gradients/pred_q_a/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
—
+Optimizer/gradients/pred_q_a/mul_grad/mul_1Mul:over_options/q_func/action_value/fully_connected_1/BiasAdd/Optimizer/gradients/pred_q_a/pred_q_a_grad/Tile*
T0*'
_output_shapes
:€€€€€€€€€
ё
+Optimizer/gradients/pred_q_a/mul_grad/Sum_1Sum+Optimizer/gradients/pred_q_a/mul_grad/mul_1=Optimizer/gradients/pred_q_a/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
÷
/Optimizer/gradients/pred_q_a/mul_grad/Reshape_1Reshape+Optimizer/gradients/pred_q_a/mul_grad/Sum_1-Optimizer/gradients/pred_q_a/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
†
6Optimizer/gradients/pred_q_a/mul_grad/tuple/group_depsNoOp.^Optimizer/gradients/pred_q_a/mul_grad/Reshape0^Optimizer/gradients/pred_q_a/mul_grad/Reshape_1
¶
>Optimizer/gradients/pred_q_a/mul_grad/tuple/control_dependencyIdentity-Optimizer/gradients/pred_q_a/mul_grad/Reshape7^Optimizer/gradients/pred_q_a/mul_grad/tuple/group_deps*
T0*@
_class6
42loc:@Optimizer/gradients/pred_q_a/mul_grad/Reshape*'
_output_shapes
:€€€€€€€€€
ђ
@Optimizer/gradients/pred_q_a/mul_grad/tuple/control_dependency_1Identity/Optimizer/gradients/pred_q_a/mul_grad/Reshape_17^Optimizer/gradients/pred_q_a/mul_grad/tuple/group_deps*
T0*B
_class8
64loc:@Optimizer/gradients/pred_q_a/mul_grad/Reshape_1*'
_output_shapes
:€€€€€€€€€
к
_Optimizer/gradients/over_options/q_func/action_value/fully_connected_1/BiasAdd_grad/BiasAddGradBiasAddGrad>Optimizer/gradients/pred_q_a/mul_grad/tuple/control_dependency*
T0*
data_formatNHWC*
_output_shapes
:
П
dOptimizer/gradients/over_options/q_func/action_value/fully_connected_1/BiasAdd_grad/tuple/group_depsNoOp?^Optimizer/gradients/pred_q_a/mul_grad/tuple/control_dependency`^Optimizer/gradients/over_options/q_func/action_value/fully_connected_1/BiasAdd_grad/BiasAddGrad
У
lOptimizer/gradients/over_options/q_func/action_value/fully_connected_1/BiasAdd_grad/tuple/control_dependencyIdentity>Optimizer/gradients/pred_q_a/mul_grad/tuple/control_dependencye^Optimizer/gradients/over_options/q_func/action_value/fully_connected_1/BiasAdd_grad/tuple/group_deps*
T0*@
_class6
42loc:@Optimizer/gradients/pred_q_a/mul_grad/Reshape*'
_output_shapes
:€€€€€€€€€
џ
nOptimizer/gradients/over_options/q_func/action_value/fully_connected_1/BiasAdd_grad/tuple/control_dependency_1Identity_Optimizer/gradients/over_options/q_func/action_value/fully_connected_1/BiasAdd_grad/BiasAddGrade^Optimizer/gradients/over_options/q_func/action_value/fully_connected_1/BiasAdd_grad/tuple/group_deps*
T0*r
_classh
fdloc:@Optimizer/gradients/over_options/q_func/action_value/fully_connected_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
л
YOptimizer/gradients/over_options/q_func/action_value/fully_connected_1/MatMul_grad/MatMulMatMullOptimizer/gradients/over_options/q_func/action_value/fully_connected_1/BiasAdd_grad/tuple/control_dependency?over_options/q_func/action_value/fully_connected_1/weights/read*
transpose_a( *
transpose_b(*
T0*(
_output_shapes
:€€€€€€€€€А
Џ
[Optimizer/gradients/over_options/q_func/action_value/fully_connected_1/MatMul_grad/MatMul_1MatMul5over_options/q_func/action_value/fully_connected/RelulOptimizer/gradients/over_options/q_func/action_value/fully_connected_1/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
T0*
_output_shapes
:	А
•
cOptimizer/gradients/over_options/q_func/action_value/fully_connected_1/MatMul_grad/tuple/group_depsNoOpZ^Optimizer/gradients/over_options/q_func/action_value/fully_connected_1/MatMul_grad/MatMul\^Optimizer/gradients/over_options/q_func/action_value/fully_connected_1/MatMul_grad/MatMul_1
ў
kOptimizer/gradients/over_options/q_func/action_value/fully_connected_1/MatMul_grad/tuple/control_dependencyIdentityYOptimizer/gradients/over_options/q_func/action_value/fully_connected_1/MatMul_grad/MatMuld^Optimizer/gradients/over_options/q_func/action_value/fully_connected_1/MatMul_grad/tuple/group_deps*
T0*l
_classb
`^loc:@Optimizer/gradients/over_options/q_func/action_value/fully_connected_1/MatMul_grad/MatMul*(
_output_shapes
:€€€€€€€€€А
÷
mOptimizer/gradients/over_options/q_func/action_value/fully_connected_1/MatMul_grad/tuple/control_dependency_1Identity[Optimizer/gradients/over_options/q_func/action_value/fully_connected_1/MatMul_grad/MatMul_1d^Optimizer/gradients/over_options/q_func/action_value/fully_connected_1/MatMul_grad/tuple/group_deps*
T0*n
_classd
b`loc:@Optimizer/gradients/over_options/q_func/action_value/fully_connected_1/MatMul_grad/MatMul_1*
_output_shapes
:	А
Ї
WOptimizer/gradients/over_options/q_func/action_value/fully_connected/Relu_grad/ReluGradReluGradkOptimizer/gradients/over_options/q_func/action_value/fully_connected_1/MatMul_grad/tuple/control_dependency5over_options/q_func/action_value/fully_connected/Relu*
T0*(
_output_shapes
:€€€€€€€€€А
В
]Optimizer/gradients/over_options/q_func/action_value/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradWOptimizer/gradients/over_options/q_func/action_value/fully_connected/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:А
§
bOptimizer/gradients/over_options/q_func/action_value/fully_connected/BiasAdd_grad/tuple/group_depsNoOpX^Optimizer/gradients/over_options/q_func/action_value/fully_connected/Relu_grad/ReluGrad^^Optimizer/gradients/over_options/q_func/action_value/fully_connected/BiasAdd_grad/BiasAddGrad
”
jOptimizer/gradients/over_options/q_func/action_value/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentityWOptimizer/gradients/over_options/q_func/action_value/fully_connected/Relu_grad/ReluGradc^Optimizer/gradients/over_options/q_func/action_value/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*j
_class`
^\loc:@Optimizer/gradients/over_options/q_func/action_value/fully_connected/Relu_grad/ReluGrad*(
_output_shapes
:€€€€€€€€€А
‘
lOptimizer/gradients/over_options/q_func/action_value/fully_connected/BiasAdd_grad/tuple/control_dependency_1Identity]Optimizer/gradients/over_options/q_func/action_value/fully_connected/BiasAdd_grad/BiasAddGradc^Optimizer/gradients/over_options/q_func/action_value/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*p
_classf
dbloc:@Optimizer/gradients/over_options/q_func/action_value/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:А
е
WOptimizer/gradients/over_options/q_func/action_value/fully_connected/MatMul_grad/MatMulMatMuljOptimizer/gradients/over_options/q_func/action_value/fully_connected/BiasAdd_grad/tuple/control_dependency=over_options/q_func/action_value/fully_connected/weights/read*
transpose_a( *
transpose_b(*
T0*(
_output_shapes
:€€€€€€€€€А
≈
YOptimizer/gradients/over_options/q_func/action_value/fully_connected/MatMul_grad/MatMul_1MatMul#over_options/q_func/Flatten/ReshapejOptimizer/gradients/over_options/q_func/action_value/fully_connected/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
T0* 
_output_shapes
:
АА
Я
aOptimizer/gradients/over_options/q_func/action_value/fully_connected/MatMul_grad/tuple/group_depsNoOpX^Optimizer/gradients/over_options/q_func/action_value/fully_connected/MatMul_grad/MatMulZ^Optimizer/gradients/over_options/q_func/action_value/fully_connected/MatMul_grad/MatMul_1
—
iOptimizer/gradients/over_options/q_func/action_value/fully_connected/MatMul_grad/tuple/control_dependencyIdentityWOptimizer/gradients/over_options/q_func/action_value/fully_connected/MatMul_grad/MatMulb^Optimizer/gradients/over_options/q_func/action_value/fully_connected/MatMul_grad/tuple/group_deps*
T0*j
_class`
^\loc:@Optimizer/gradients/over_options/q_func/action_value/fully_connected/MatMul_grad/MatMul*(
_output_shapes
:€€€€€€€€€А
ѕ
kOptimizer/gradients/over_options/q_func/action_value/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityYOptimizer/gradients/over_options/q_func/action_value/fully_connected/MatMul_grad/MatMul_1b^Optimizer/gradients/over_options/q_func/action_value/fully_connected/MatMul_grad/tuple/group_deps*
T0*l
_classb
`^loc:@Optimizer/gradients/over_options/q_func/action_value/fully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
АА
©
BOptimizer/gradients/over_options/q_func/Flatten/Reshape_grad/ShapeShape'over_options/q_func/convnet/Conv_2/Relu*
T0*
out_type0*
_output_shapes
:
∆
DOptimizer/gradients/over_options/q_func/Flatten/Reshape_grad/ReshapeReshapeiOptimizer/gradients/over_options/q_func/action_value/fully_connected/MatMul_grad/tuple/control_dependencyBOptimizer/gradients/over_options/q_func/Flatten/Reshape_grad/Shape*
T0*
Tshape0*/
_output_shapes
:€€€€€€€€€@
ю
IOptimizer/gradients/over_options/q_func/convnet/Conv_2/Relu_grad/ReluGradReluGradDOptimizer/gradients/over_options/q_func/Flatten/Reshape_grad/Reshape'over_options/q_func/convnet/Conv_2/Relu*
T0*/
_output_shapes
:€€€€€€€€€@
е
OOptimizer/gradients/over_options/q_func/convnet/Conv_2/BiasAdd_grad/BiasAddGradBiasAddGradIOptimizer/gradients/over_options/q_func/convnet/Conv_2/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@
ъ
TOptimizer/gradients/over_options/q_func/convnet/Conv_2/BiasAdd_grad/tuple/group_depsNoOpJ^Optimizer/gradients/over_options/q_func/convnet/Conv_2/Relu_grad/ReluGradP^Optimizer/gradients/over_options/q_func/convnet/Conv_2/BiasAdd_grad/BiasAddGrad
Ґ
\Optimizer/gradients/over_options/q_func/convnet/Conv_2/BiasAdd_grad/tuple/control_dependencyIdentityIOptimizer/gradients/over_options/q_func/convnet/Conv_2/Relu_grad/ReluGradU^Optimizer/gradients/over_options/q_func/convnet/Conv_2/BiasAdd_grad/tuple/group_deps*
T0*\
_classR
PNloc:@Optimizer/gradients/over_options/q_func/convnet/Conv_2/Relu_grad/ReluGrad*/
_output_shapes
:€€€€€€€€€@
Ы
^Optimizer/gradients/over_options/q_func/convnet/Conv_2/BiasAdd_grad/tuple/control_dependency_1IdentityOOptimizer/gradients/over_options/q_func/convnet/Conv_2/BiasAdd_grad/BiasAddGradU^Optimizer/gradients/over_options/q_func/convnet/Conv_2/BiasAdd_grad/tuple/group_deps*
T0*b
_classX
VTloc:@Optimizer/gradients/over_options/q_func/convnet/Conv_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
і
MOptimizer/gradients/over_options/q_func/convnet/Conv_2/convolution_grad/ShapeShape'over_options/q_func/convnet/Conv_1/Relu*
T0*
out_type0*
_output_shapes
:
ь
[Optimizer/gradients/over_options/q_func/convnet/Conv_2/convolution_grad/Conv2DBackpropInputConv2DBackpropInputMOptimizer/gradients/over_options/q_func/convnet/Conv_2/convolution_grad/Shape/over_options/q_func/convnet/Conv_2/weights/read\Optimizer/gradients/over_options/q_func/convnet/Conv_2/BiasAdd_grad/tuple/control_dependency*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
®
OOptimizer/gradients/over_options/q_func/convnet/Conv_2/convolution_grad/Shape_1Const*%
valueB"      @   @   *
dtype0*
_output_shapes
:
‘
\Optimizer/gradients/over_options/q_func/convnet/Conv_2/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilter'over_options/q_func/convnet/Conv_1/ReluOOptimizer/gradients/over_options/q_func/convnet/Conv_2/convolution_grad/Shape_1\Optimizer/gradients/over_options/q_func/convnet/Conv_2/BiasAdd_grad/tuple/control_dependency*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*&
_output_shapes
:@@
Э
XOptimizer/gradients/over_options/q_func/convnet/Conv_2/convolution_grad/tuple/group_depsNoOp\^Optimizer/gradients/over_options/q_func/convnet/Conv_2/convolution_grad/Conv2DBackpropInput]^Optimizer/gradients/over_options/q_func/convnet/Conv_2/convolution_grad/Conv2DBackpropFilter
ќ
`Optimizer/gradients/over_options/q_func/convnet/Conv_2/convolution_grad/tuple/control_dependencyIdentity[Optimizer/gradients/over_options/q_func/convnet/Conv_2/convolution_grad/Conv2DBackpropInputY^Optimizer/gradients/over_options/q_func/convnet/Conv_2/convolution_grad/tuple/group_deps*
T0*n
_classd
b`loc:@Optimizer/gradients/over_options/q_func/convnet/Conv_2/convolution_grad/Conv2DBackpropInput*/
_output_shapes
:€€€€€€€€€@
…
bOptimizer/gradients/over_options/q_func/convnet/Conv_2/convolution_grad/tuple/control_dependency_1Identity\Optimizer/gradients/over_options/q_func/convnet/Conv_2/convolution_grad/Conv2DBackpropFilterY^Optimizer/gradients/over_options/q_func/convnet/Conv_2/convolution_grad/tuple/group_deps*
T0*o
_classe
caloc:@Optimizer/gradients/over_options/q_func/convnet/Conv_2/convolution_grad/Conv2DBackpropFilter*&
_output_shapes
:@@
Ъ
IOptimizer/gradients/over_options/q_func/convnet/Conv_1/Relu_grad/ReluGradReluGrad`Optimizer/gradients/over_options/q_func/convnet/Conv_2/convolution_grad/tuple/control_dependency'over_options/q_func/convnet/Conv_1/Relu*
T0*/
_output_shapes
:€€€€€€€€€@
е
OOptimizer/gradients/over_options/q_func/convnet/Conv_1/BiasAdd_grad/BiasAddGradBiasAddGradIOptimizer/gradients/over_options/q_func/convnet/Conv_1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@
ъ
TOptimizer/gradients/over_options/q_func/convnet/Conv_1/BiasAdd_grad/tuple/group_depsNoOpJ^Optimizer/gradients/over_options/q_func/convnet/Conv_1/Relu_grad/ReluGradP^Optimizer/gradients/over_options/q_func/convnet/Conv_1/BiasAdd_grad/BiasAddGrad
Ґ
\Optimizer/gradients/over_options/q_func/convnet/Conv_1/BiasAdd_grad/tuple/control_dependencyIdentityIOptimizer/gradients/over_options/q_func/convnet/Conv_1/Relu_grad/ReluGradU^Optimizer/gradients/over_options/q_func/convnet/Conv_1/BiasAdd_grad/tuple/group_deps*
T0*\
_classR
PNloc:@Optimizer/gradients/over_options/q_func/convnet/Conv_1/Relu_grad/ReluGrad*/
_output_shapes
:€€€€€€€€€@
Ы
^Optimizer/gradients/over_options/q_func/convnet/Conv_1/BiasAdd_grad/tuple/control_dependency_1IdentityOOptimizer/gradients/over_options/q_func/convnet/Conv_1/BiasAdd_grad/BiasAddGradU^Optimizer/gradients/over_options/q_func/convnet/Conv_1/BiasAdd_grad/tuple/group_deps*
T0*b
_classX
VTloc:@Optimizer/gradients/over_options/q_func/convnet/Conv_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
≤
MOptimizer/gradients/over_options/q_func/convnet/Conv_1/convolution_grad/ShapeShape%over_options/q_func/convnet/Conv/Relu*
T0*
out_type0*
_output_shapes
:
ь
[Optimizer/gradients/over_options/q_func/convnet/Conv_1/convolution_grad/Conv2DBackpropInputConv2DBackpropInputMOptimizer/gradients/over_options/q_func/convnet/Conv_1/convolution_grad/Shape/over_options/q_func/convnet/Conv_1/weights/read\Optimizer/gradients/over_options/q_func/convnet/Conv_1/BiasAdd_grad/tuple/control_dependency*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
®
OOptimizer/gradients/over_options/q_func/convnet/Conv_1/convolution_grad/Shape_1Const*%
valueB"          @   *
dtype0*
_output_shapes
:
“
\Optimizer/gradients/over_options/q_func/convnet/Conv_1/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilter%over_options/q_func/convnet/Conv/ReluOOptimizer/gradients/over_options/q_func/convnet/Conv_1/convolution_grad/Shape_1\Optimizer/gradients/over_options/q_func/convnet/Conv_1/BiasAdd_grad/tuple/control_dependency*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*&
_output_shapes
: @
Э
XOptimizer/gradients/over_options/q_func/convnet/Conv_1/convolution_grad/tuple/group_depsNoOp\^Optimizer/gradients/over_options/q_func/convnet/Conv_1/convolution_grad/Conv2DBackpropInput]^Optimizer/gradients/over_options/q_func/convnet/Conv_1/convolution_grad/Conv2DBackpropFilter
ќ
`Optimizer/gradients/over_options/q_func/convnet/Conv_1/convolution_grad/tuple/control_dependencyIdentity[Optimizer/gradients/over_options/q_func/convnet/Conv_1/convolution_grad/Conv2DBackpropInputY^Optimizer/gradients/over_options/q_func/convnet/Conv_1/convolution_grad/tuple/group_deps*
T0*n
_classd
b`loc:@Optimizer/gradients/over_options/q_func/convnet/Conv_1/convolution_grad/Conv2DBackpropInput*/
_output_shapes
:€€€€€€€€€ 
…
bOptimizer/gradients/over_options/q_func/convnet/Conv_1/convolution_grad/tuple/control_dependency_1Identity\Optimizer/gradients/over_options/q_func/convnet/Conv_1/convolution_grad/Conv2DBackpropFilterY^Optimizer/gradients/over_options/q_func/convnet/Conv_1/convolution_grad/tuple/group_deps*
T0*o
_classe
caloc:@Optimizer/gradients/over_options/q_func/convnet/Conv_1/convolution_grad/Conv2DBackpropFilter*&
_output_shapes
: @
Ц
GOptimizer/gradients/over_options/q_func/convnet/Conv/Relu_grad/ReluGradReluGrad`Optimizer/gradients/over_options/q_func/convnet/Conv_1/convolution_grad/tuple/control_dependency%over_options/q_func/convnet/Conv/Relu*
T0*/
_output_shapes
:€€€€€€€€€ 
б
MOptimizer/gradients/over_options/q_func/convnet/Conv/BiasAdd_grad/BiasAddGradBiasAddGradGOptimizer/gradients/over_options/q_func/convnet/Conv/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
: 
ф
ROptimizer/gradients/over_options/q_func/convnet/Conv/BiasAdd_grad/tuple/group_depsNoOpH^Optimizer/gradients/over_options/q_func/convnet/Conv/Relu_grad/ReluGradN^Optimizer/gradients/over_options/q_func/convnet/Conv/BiasAdd_grad/BiasAddGrad
Ъ
ZOptimizer/gradients/over_options/q_func/convnet/Conv/BiasAdd_grad/tuple/control_dependencyIdentityGOptimizer/gradients/over_options/q_func/convnet/Conv/Relu_grad/ReluGradS^Optimizer/gradients/over_options/q_func/convnet/Conv/BiasAdd_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@Optimizer/gradients/over_options/q_func/convnet/Conv/Relu_grad/ReluGrad*/
_output_shapes
:€€€€€€€€€ 
У
\Optimizer/gradients/over_options/q_func/convnet/Conv/BiasAdd_grad/tuple/control_dependency_1IdentityMOptimizer/gradients/over_options/q_func/convnet/Conv/BiasAdd_grad/BiasAddGradS^Optimizer/gradients/over_options/q_func/convnet/Conv/BiasAdd_grad/tuple/group_deps*
T0*`
_classV
TRloc:@Optimizer/gradients/over_options/q_func/convnet/Conv/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 
£
KOptimizer/gradients/over_options/q_func/convnet/Conv/convolution_grad/ShapeShapeover_options/obs_t_float*
T0*
out_type0*
_output_shapes
:
ф
YOptimizer/gradients/over_options/q_func/convnet/Conv/convolution_grad/Conv2DBackpropInputConv2DBackpropInputKOptimizer/gradients/over_options/q_func/convnet/Conv/convolution_grad/Shape-over_options/q_func/convnet/Conv/weights/readZOptimizer/gradients/over_options/q_func/convnet/Conv/BiasAdd_grad/tuple/control_dependency*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
¶
MOptimizer/gradients/over_options/q_func/convnet/Conv/convolution_grad/Shape_1Const*%
valueB"             *
dtype0*
_output_shapes
:
њ
ZOptimizer/gradients/over_options/q_func/convnet/Conv/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterover_options/obs_t_floatMOptimizer/gradients/over_options/q_func/convnet/Conv/convolution_grad/Shape_1ZOptimizer/gradients/over_options/q_func/convnet/Conv/BiasAdd_grad/tuple/control_dependency*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*&
_output_shapes
: 
Ч
VOptimizer/gradients/over_options/q_func/convnet/Conv/convolution_grad/tuple/group_depsNoOpZ^Optimizer/gradients/over_options/q_func/convnet/Conv/convolution_grad/Conv2DBackpropInput[^Optimizer/gradients/over_options/q_func/convnet/Conv/convolution_grad/Conv2DBackpropFilter
∆
^Optimizer/gradients/over_options/q_func/convnet/Conv/convolution_grad/tuple/control_dependencyIdentityYOptimizer/gradients/over_options/q_func/convnet/Conv/convolution_grad/Conv2DBackpropInputW^Optimizer/gradients/over_options/q_func/convnet/Conv/convolution_grad/tuple/group_deps*
T0*l
_classb
`^loc:@Optimizer/gradients/over_options/q_func/convnet/Conv/convolution_grad/Conv2DBackpropInput*/
_output_shapes
:€€€€€€€€€	
Ѕ
`Optimizer/gradients/over_options/q_func/convnet/Conv/convolution_grad/tuple/control_dependency_1IdentityZOptimizer/gradients/over_options/q_func/convnet/Conv/convolution_grad/Conv2DBackpropFilterW^Optimizer/gradients/over_options/q_func/convnet/Conv/convolution_grad/tuple/group_deps*
T0*m
_classc
a_loc:@Optimizer/gradients/over_options/q_func/convnet/Conv/convolution_grad/Conv2DBackpropFilter*&
_output_shapes
: 
Ц
Optimizer/clip_by_norm/mulMul`Optimizer/gradients/over_options/q_func/convnet/Conv/convolution_grad/tuple/control_dependency_1`Optimizer/gradients/over_options/q_func/convnet/Conv/convolution_grad/tuple/control_dependency_1*
T0*&
_output_shapes
: 
u
Optimizer/clip_by_norm/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:
©
Optimizer/clip_by_norm/SumSumOptimizer/clip_by_norm/mulOptimizer/clip_by_norm/Const*
	keep_dims(*
T0*

Tidx0*&
_output_shapes
:
r
Optimizer/clip_by_norm/RsqrtRsqrtOptimizer/clip_by_norm/Sum*
T0*&
_output_shapes
:
c
Optimizer/clip_by_norm/mul_1/yConst*
valueB
 *   A*
dtype0*
_output_shapes
: 
÷
Optimizer/clip_by_norm/mul_1Mul`Optimizer/gradients/over_options/q_func/convnet/Conv/convolution_grad/tuple/control_dependency_1Optimizer/clip_by_norm/mul_1/y*
T0*&
_output_shapes
: 
c
Optimizer/clip_by_norm/Const_1Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
e
 Optimizer/clip_by_norm/truediv/yConst*
valueB
 *   A*
dtype0*
_output_shapes
: 
М
Optimizer/clip_by_norm/truedivRealDivOptimizer/clip_by_norm/Const_1 Optimizer/clip_by_norm/truediv/y*
T0*
_output_shapes
: 
Ш
Optimizer/clip_by_norm/MinimumMinimumOptimizer/clip_by_norm/RsqrtOptimizer/clip_by_norm/truediv*
T0*&
_output_shapes
:
Т
Optimizer/clip_by_norm/mul_2MulOptimizer/clip_by_norm/mul_1Optimizer/clip_by_norm/Minimum*
T0*&
_output_shapes
: 
q
Optimizer/clip_by_normIdentityOptimizer/clip_by_norm/mul_2*
T0*&
_output_shapes
: 
Д
Optimizer/clip_by_norm_1/mulMul\Optimizer/gradients/over_options/q_func/convnet/Conv/BiasAdd_grad/tuple/control_dependency_1\Optimizer/gradients/over_options/q_func/convnet/Conv/BiasAdd_grad/tuple/control_dependency_1*
T0*
_output_shapes
: 
h
Optimizer/clip_by_norm_1/ConstConst*
valueB: *
dtype0*
_output_shapes
:
£
Optimizer/clip_by_norm_1/SumSumOptimizer/clip_by_norm_1/mulOptimizer/clip_by_norm_1/Const*
	keep_dims(*
T0*

Tidx0*
_output_shapes
:
j
Optimizer/clip_by_norm_1/RsqrtRsqrtOptimizer/clip_by_norm_1/Sum*
T0*
_output_shapes
:
e
 Optimizer/clip_by_norm_1/mul_1/yConst*
valueB
 *   A*
dtype0*
_output_shapes
: 
 
Optimizer/clip_by_norm_1/mul_1Mul\Optimizer/gradients/over_options/q_func/convnet/Conv/BiasAdd_grad/tuple/control_dependency_1 Optimizer/clip_by_norm_1/mul_1/y*
T0*
_output_shapes
: 
e
 Optimizer/clip_by_norm_1/Const_1Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
g
"Optimizer/clip_by_norm_1/truediv/yConst*
valueB
 *   A*
dtype0*
_output_shapes
: 
Т
 Optimizer/clip_by_norm_1/truedivRealDiv Optimizer/clip_by_norm_1/Const_1"Optimizer/clip_by_norm_1/truediv/y*
T0*
_output_shapes
: 
Т
 Optimizer/clip_by_norm_1/MinimumMinimumOptimizer/clip_by_norm_1/Rsqrt Optimizer/clip_by_norm_1/truediv*
T0*
_output_shapes
:
М
Optimizer/clip_by_norm_1/mul_2MulOptimizer/clip_by_norm_1/mul_1 Optimizer/clip_by_norm_1/Minimum*
T0*
_output_shapes
: 
i
Optimizer/clip_by_norm_1IdentityOptimizer/clip_by_norm_1/mul_2*
T0*
_output_shapes
: 
Ь
Optimizer/clip_by_norm_2/mulMulbOptimizer/gradients/over_options/q_func/convnet/Conv_1/convolution_grad/tuple/control_dependency_1bOptimizer/gradients/over_options/q_func/convnet/Conv_1/convolution_grad/tuple/control_dependency_1*
T0*&
_output_shapes
: @
w
Optimizer/clip_by_norm_2/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:
ѓ
Optimizer/clip_by_norm_2/SumSumOptimizer/clip_by_norm_2/mulOptimizer/clip_by_norm_2/Const*
	keep_dims(*
T0*

Tidx0*&
_output_shapes
:
v
Optimizer/clip_by_norm_2/RsqrtRsqrtOptimizer/clip_by_norm_2/Sum*
T0*&
_output_shapes
:
e
 Optimizer/clip_by_norm_2/mul_1/yConst*
valueB
 *   A*
dtype0*
_output_shapes
: 
№
Optimizer/clip_by_norm_2/mul_1MulbOptimizer/gradients/over_options/q_func/convnet/Conv_1/convolution_grad/tuple/control_dependency_1 Optimizer/clip_by_norm_2/mul_1/y*
T0*&
_output_shapes
: @
e
 Optimizer/clip_by_norm_2/Const_1Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
g
"Optimizer/clip_by_norm_2/truediv/yConst*
valueB
 *   A*
dtype0*
_output_shapes
: 
Т
 Optimizer/clip_by_norm_2/truedivRealDiv Optimizer/clip_by_norm_2/Const_1"Optimizer/clip_by_norm_2/truediv/y*
T0*
_output_shapes
: 
Ю
 Optimizer/clip_by_norm_2/MinimumMinimumOptimizer/clip_by_norm_2/Rsqrt Optimizer/clip_by_norm_2/truediv*
T0*&
_output_shapes
:
Ш
Optimizer/clip_by_norm_2/mul_2MulOptimizer/clip_by_norm_2/mul_1 Optimizer/clip_by_norm_2/Minimum*
T0*&
_output_shapes
: @
u
Optimizer/clip_by_norm_2IdentityOptimizer/clip_by_norm_2/mul_2*
T0*&
_output_shapes
: @
И
Optimizer/clip_by_norm_3/mulMul^Optimizer/gradients/over_options/q_func/convnet/Conv_1/BiasAdd_grad/tuple/control_dependency_1^Optimizer/gradients/over_options/q_func/convnet/Conv_1/BiasAdd_grad/tuple/control_dependency_1*
T0*
_output_shapes
:@
h
Optimizer/clip_by_norm_3/ConstConst*
valueB: *
dtype0*
_output_shapes
:
£
Optimizer/clip_by_norm_3/SumSumOptimizer/clip_by_norm_3/mulOptimizer/clip_by_norm_3/Const*
	keep_dims(*
T0*

Tidx0*
_output_shapes
:
j
Optimizer/clip_by_norm_3/RsqrtRsqrtOptimizer/clip_by_norm_3/Sum*
T0*
_output_shapes
:
e
 Optimizer/clip_by_norm_3/mul_1/yConst*
valueB
 *   A*
dtype0*
_output_shapes
: 
ћ
Optimizer/clip_by_norm_3/mul_1Mul^Optimizer/gradients/over_options/q_func/convnet/Conv_1/BiasAdd_grad/tuple/control_dependency_1 Optimizer/clip_by_norm_3/mul_1/y*
T0*
_output_shapes
:@
e
 Optimizer/clip_by_norm_3/Const_1Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
g
"Optimizer/clip_by_norm_3/truediv/yConst*
valueB
 *   A*
dtype0*
_output_shapes
: 
Т
 Optimizer/clip_by_norm_3/truedivRealDiv Optimizer/clip_by_norm_3/Const_1"Optimizer/clip_by_norm_3/truediv/y*
T0*
_output_shapes
: 
Т
 Optimizer/clip_by_norm_3/MinimumMinimumOptimizer/clip_by_norm_3/Rsqrt Optimizer/clip_by_norm_3/truediv*
T0*
_output_shapes
:
М
Optimizer/clip_by_norm_3/mul_2MulOptimizer/clip_by_norm_3/mul_1 Optimizer/clip_by_norm_3/Minimum*
T0*
_output_shapes
:@
i
Optimizer/clip_by_norm_3IdentityOptimizer/clip_by_norm_3/mul_2*
T0*
_output_shapes
:@
Ь
Optimizer/clip_by_norm_4/mulMulbOptimizer/gradients/over_options/q_func/convnet/Conv_2/convolution_grad/tuple/control_dependency_1bOptimizer/gradients/over_options/q_func/convnet/Conv_2/convolution_grad/tuple/control_dependency_1*
T0*&
_output_shapes
:@@
w
Optimizer/clip_by_norm_4/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:
ѓ
Optimizer/clip_by_norm_4/SumSumOptimizer/clip_by_norm_4/mulOptimizer/clip_by_norm_4/Const*
	keep_dims(*
T0*

Tidx0*&
_output_shapes
:
v
Optimizer/clip_by_norm_4/RsqrtRsqrtOptimizer/clip_by_norm_4/Sum*
T0*&
_output_shapes
:
e
 Optimizer/clip_by_norm_4/mul_1/yConst*
valueB
 *   A*
dtype0*
_output_shapes
: 
№
Optimizer/clip_by_norm_4/mul_1MulbOptimizer/gradients/over_options/q_func/convnet/Conv_2/convolution_grad/tuple/control_dependency_1 Optimizer/clip_by_norm_4/mul_1/y*
T0*&
_output_shapes
:@@
e
 Optimizer/clip_by_norm_4/Const_1Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
g
"Optimizer/clip_by_norm_4/truediv/yConst*
valueB
 *   A*
dtype0*
_output_shapes
: 
Т
 Optimizer/clip_by_norm_4/truedivRealDiv Optimizer/clip_by_norm_4/Const_1"Optimizer/clip_by_norm_4/truediv/y*
T0*
_output_shapes
: 
Ю
 Optimizer/clip_by_norm_4/MinimumMinimumOptimizer/clip_by_norm_4/Rsqrt Optimizer/clip_by_norm_4/truediv*
T0*&
_output_shapes
:
Ш
Optimizer/clip_by_norm_4/mul_2MulOptimizer/clip_by_norm_4/mul_1 Optimizer/clip_by_norm_4/Minimum*
T0*&
_output_shapes
:@@
u
Optimizer/clip_by_norm_4IdentityOptimizer/clip_by_norm_4/mul_2*
T0*&
_output_shapes
:@@
И
Optimizer/clip_by_norm_5/mulMul^Optimizer/gradients/over_options/q_func/convnet/Conv_2/BiasAdd_grad/tuple/control_dependency_1^Optimizer/gradients/over_options/q_func/convnet/Conv_2/BiasAdd_grad/tuple/control_dependency_1*
T0*
_output_shapes
:@
h
Optimizer/clip_by_norm_5/ConstConst*
valueB: *
dtype0*
_output_shapes
:
£
Optimizer/clip_by_norm_5/SumSumOptimizer/clip_by_norm_5/mulOptimizer/clip_by_norm_5/Const*
	keep_dims(*
T0*

Tidx0*
_output_shapes
:
j
Optimizer/clip_by_norm_5/RsqrtRsqrtOptimizer/clip_by_norm_5/Sum*
T0*
_output_shapes
:
e
 Optimizer/clip_by_norm_5/mul_1/yConst*
valueB
 *   A*
dtype0*
_output_shapes
: 
ћ
Optimizer/clip_by_norm_5/mul_1Mul^Optimizer/gradients/over_options/q_func/convnet/Conv_2/BiasAdd_grad/tuple/control_dependency_1 Optimizer/clip_by_norm_5/mul_1/y*
T0*
_output_shapes
:@
e
 Optimizer/clip_by_norm_5/Const_1Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
g
"Optimizer/clip_by_norm_5/truediv/yConst*
valueB
 *   A*
dtype0*
_output_shapes
: 
Т
 Optimizer/clip_by_norm_5/truedivRealDiv Optimizer/clip_by_norm_5/Const_1"Optimizer/clip_by_norm_5/truediv/y*
T0*
_output_shapes
: 
Т
 Optimizer/clip_by_norm_5/MinimumMinimumOptimizer/clip_by_norm_5/Rsqrt Optimizer/clip_by_norm_5/truediv*
T0*
_output_shapes
:
М
Optimizer/clip_by_norm_5/mul_2MulOptimizer/clip_by_norm_5/mul_1 Optimizer/clip_by_norm_5/Minimum*
T0*
_output_shapes
:@
i
Optimizer/clip_by_norm_5IdentityOptimizer/clip_by_norm_5/mul_2*
T0*
_output_shapes
:@
®
Optimizer/clip_by_norm_6/mulMulkOptimizer/gradients/over_options/q_func/action_value/fully_connected/MatMul_grad/tuple/control_dependency_1kOptimizer/gradients/over_options/q_func/action_value/fully_connected/MatMul_grad/tuple/control_dependency_1*
T0* 
_output_shapes
:
АА
o
Optimizer/clip_by_norm_6/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
І
Optimizer/clip_by_norm_6/SumSumOptimizer/clip_by_norm_6/mulOptimizer/clip_by_norm_6/Const*
	keep_dims(*
T0*

Tidx0*
_output_shapes

:
n
Optimizer/clip_by_norm_6/RsqrtRsqrtOptimizer/clip_by_norm_6/Sum*
T0*
_output_shapes

:
e
 Optimizer/clip_by_norm_6/mul_1/yConst*
valueB
 *   A*
dtype0*
_output_shapes
: 
я
Optimizer/clip_by_norm_6/mul_1MulkOptimizer/gradients/over_options/q_func/action_value/fully_connected/MatMul_grad/tuple/control_dependency_1 Optimizer/clip_by_norm_6/mul_1/y*
T0* 
_output_shapes
:
АА
e
 Optimizer/clip_by_norm_6/Const_1Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
g
"Optimizer/clip_by_norm_6/truediv/yConst*
valueB
 *   A*
dtype0*
_output_shapes
: 
Т
 Optimizer/clip_by_norm_6/truedivRealDiv Optimizer/clip_by_norm_6/Const_1"Optimizer/clip_by_norm_6/truediv/y*
T0*
_output_shapes
: 
Ц
 Optimizer/clip_by_norm_6/MinimumMinimumOptimizer/clip_by_norm_6/Rsqrt Optimizer/clip_by_norm_6/truediv*
T0*
_output_shapes

:
Т
Optimizer/clip_by_norm_6/mul_2MulOptimizer/clip_by_norm_6/mul_1 Optimizer/clip_by_norm_6/Minimum*
T0* 
_output_shapes
:
АА
o
Optimizer/clip_by_norm_6IdentityOptimizer/clip_by_norm_6/mul_2*
T0* 
_output_shapes
:
АА
•
Optimizer/clip_by_norm_7/mulMullOptimizer/gradients/over_options/q_func/action_value/fully_connected/BiasAdd_grad/tuple/control_dependency_1lOptimizer/gradients/over_options/q_func/action_value/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
T0*
_output_shapes	
:А
h
Optimizer/clip_by_norm_7/ConstConst*
valueB: *
dtype0*
_output_shapes
:
£
Optimizer/clip_by_norm_7/SumSumOptimizer/clip_by_norm_7/mulOptimizer/clip_by_norm_7/Const*
	keep_dims(*
T0*

Tidx0*
_output_shapes
:
j
Optimizer/clip_by_norm_7/RsqrtRsqrtOptimizer/clip_by_norm_7/Sum*
T0*
_output_shapes
:
e
 Optimizer/clip_by_norm_7/mul_1/yConst*
valueB
 *   A*
dtype0*
_output_shapes
: 
џ
Optimizer/clip_by_norm_7/mul_1MullOptimizer/gradients/over_options/q_func/action_value/fully_connected/BiasAdd_grad/tuple/control_dependency_1 Optimizer/clip_by_norm_7/mul_1/y*
T0*
_output_shapes	
:А
e
 Optimizer/clip_by_norm_7/Const_1Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
g
"Optimizer/clip_by_norm_7/truediv/yConst*
valueB
 *   A*
dtype0*
_output_shapes
: 
Т
 Optimizer/clip_by_norm_7/truedivRealDiv Optimizer/clip_by_norm_7/Const_1"Optimizer/clip_by_norm_7/truediv/y*
T0*
_output_shapes
: 
Т
 Optimizer/clip_by_norm_7/MinimumMinimumOptimizer/clip_by_norm_7/Rsqrt Optimizer/clip_by_norm_7/truediv*
T0*
_output_shapes
:
Н
Optimizer/clip_by_norm_7/mul_2MulOptimizer/clip_by_norm_7/mul_1 Optimizer/clip_by_norm_7/Minimum*
T0*
_output_shapes	
:А
j
Optimizer/clip_by_norm_7IdentityOptimizer/clip_by_norm_7/mul_2*
T0*
_output_shapes	
:А
Ђ
Optimizer/clip_by_norm_8/mulMulmOptimizer/gradients/over_options/q_func/action_value/fully_connected_1/MatMul_grad/tuple/control_dependency_1mOptimizer/gradients/over_options/q_func/action_value/fully_connected_1/MatMul_grad/tuple/control_dependency_1*
T0*
_output_shapes
:	А
o
Optimizer/clip_by_norm_8/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
І
Optimizer/clip_by_norm_8/SumSumOptimizer/clip_by_norm_8/mulOptimizer/clip_by_norm_8/Const*
	keep_dims(*
T0*

Tidx0*
_output_shapes

:
n
Optimizer/clip_by_norm_8/RsqrtRsqrtOptimizer/clip_by_norm_8/Sum*
T0*
_output_shapes

:
e
 Optimizer/clip_by_norm_8/mul_1/yConst*
valueB
 *   A*
dtype0*
_output_shapes
: 
а
Optimizer/clip_by_norm_8/mul_1MulmOptimizer/gradients/over_options/q_func/action_value/fully_connected_1/MatMul_grad/tuple/control_dependency_1 Optimizer/clip_by_norm_8/mul_1/y*
T0*
_output_shapes
:	А
e
 Optimizer/clip_by_norm_8/Const_1Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
g
"Optimizer/clip_by_norm_8/truediv/yConst*
valueB
 *   A*
dtype0*
_output_shapes
: 
Т
 Optimizer/clip_by_norm_8/truedivRealDiv Optimizer/clip_by_norm_8/Const_1"Optimizer/clip_by_norm_8/truediv/y*
T0*
_output_shapes
: 
Ц
 Optimizer/clip_by_norm_8/MinimumMinimumOptimizer/clip_by_norm_8/Rsqrt Optimizer/clip_by_norm_8/truediv*
T0*
_output_shapes

:
С
Optimizer/clip_by_norm_8/mul_2MulOptimizer/clip_by_norm_8/mul_1 Optimizer/clip_by_norm_8/Minimum*
T0*
_output_shapes
:	А
n
Optimizer/clip_by_norm_8IdentityOptimizer/clip_by_norm_8/mul_2*
T0*
_output_shapes
:	А
®
Optimizer/clip_by_norm_9/mulMulnOptimizer/gradients/over_options/q_func/action_value/fully_connected_1/BiasAdd_grad/tuple/control_dependency_1nOptimizer/gradients/over_options/q_func/action_value/fully_connected_1/BiasAdd_grad/tuple/control_dependency_1*
T0*
_output_shapes
:
h
Optimizer/clip_by_norm_9/ConstConst*
valueB: *
dtype0*
_output_shapes
:
£
Optimizer/clip_by_norm_9/SumSumOptimizer/clip_by_norm_9/mulOptimizer/clip_by_norm_9/Const*
	keep_dims(*
T0*

Tidx0*
_output_shapes
:
j
Optimizer/clip_by_norm_9/RsqrtRsqrtOptimizer/clip_by_norm_9/Sum*
T0*
_output_shapes
:
e
 Optimizer/clip_by_norm_9/mul_1/yConst*
valueB
 *   A*
dtype0*
_output_shapes
: 
№
Optimizer/clip_by_norm_9/mul_1MulnOptimizer/gradients/over_options/q_func/action_value/fully_connected_1/BiasAdd_grad/tuple/control_dependency_1 Optimizer/clip_by_norm_9/mul_1/y*
T0*
_output_shapes
:
e
 Optimizer/clip_by_norm_9/Const_1Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
g
"Optimizer/clip_by_norm_9/truediv/yConst*
valueB
 *   A*
dtype0*
_output_shapes
: 
Т
 Optimizer/clip_by_norm_9/truedivRealDiv Optimizer/clip_by_norm_9/Const_1"Optimizer/clip_by_norm_9/truediv/y*
T0*
_output_shapes
: 
Т
 Optimizer/clip_by_norm_9/MinimumMinimumOptimizer/clip_by_norm_9/Rsqrt Optimizer/clip_by_norm_9/truediv*
T0*
_output_shapes
:
М
Optimizer/clip_by_norm_9/mul_2MulOptimizer/clip_by_norm_9/mul_1 Optimizer/clip_by_norm_9/Minimum*
T0*
_output_shapes
:
i
Optimizer/clip_by_norm_9IdentityOptimizer/clip_by_norm_9/mul_2*
T0*
_output_shapes
:
•
#Optimizer/beta1_power/initial_valueConst*
valueB
 *fff?*
dtype0*;
_class1
/-loc:@over_options/q_func/convnet/Conv/weights*
_output_shapes
: 
ґ
Optimizer/beta1_power
VariableV2*
shape: *
dtype0*
	container *
shared_name *;
_class1
/-loc:@over_options/q_func/convnet/Conv/weights*
_output_shapes
: 
й
Optimizer/beta1_power/AssignAssignOptimizer/beta1_power#Optimizer/beta1_power/initial_value*
T0*
validate_shape(*
use_locking(*;
_class1
/-loc:@over_options/q_func/convnet/Conv/weights*
_output_shapes
: 
Ы
Optimizer/beta1_power/readIdentityOptimizer/beta1_power*
T0*;
_class1
/-loc:@over_options/q_func/convnet/Conv/weights*
_output_shapes
: 
•
#Optimizer/beta2_power/initial_valueConst*
valueB
 *wЊ?*
dtype0*;
_class1
/-loc:@over_options/q_func/convnet/Conv/weights*
_output_shapes
: 
ґ
Optimizer/beta2_power
VariableV2*
shape: *
dtype0*
	container *
shared_name *;
_class1
/-loc:@over_options/q_func/convnet/Conv/weights*
_output_shapes
: 
й
Optimizer/beta2_power/AssignAssignOptimizer/beta2_power#Optimizer/beta2_power/initial_value*
T0*
validate_shape(*
use_locking(*;
_class1
/-loc:@over_options/q_func/convnet/Conv/weights*
_output_shapes
: 
Ы
Optimizer/beta2_power/readIdentityOptimizer/beta2_power*
T0*;
_class1
/-loc:@over_options/q_func/convnet/Conv/weights*
_output_shapes
: 
л
IOptimizer/over_options/q_func/convnet/Conv/weights/Adam/Initializer/ConstConst*%
valueB *    *
dtype0*;
_class1
/-loc:@over_options/q_func/convnet/Conv/weights*&
_output_shapes
: 
ш
7Optimizer/over_options/q_func/convnet/Conv/weights/Adam
VariableV2*
shape: *
dtype0*
	container *
shared_name *;
_class1
/-loc:@over_options/q_func/convnet/Conv/weights*&
_output_shapes
: 
г
>Optimizer/over_options/q_func/convnet/Conv/weights/Adam/AssignAssign7Optimizer/over_options/q_func/convnet/Conv/weights/AdamIOptimizer/over_options/q_func/convnet/Conv/weights/Adam/Initializer/Const*
T0*
validate_shape(*
use_locking(*;
_class1
/-loc:@over_options/q_func/convnet/Conv/weights*&
_output_shapes
: 
п
<Optimizer/over_options/q_func/convnet/Conv/weights/Adam/readIdentity7Optimizer/over_options/q_func/convnet/Conv/weights/Adam*
T0*;
_class1
/-loc:@over_options/q_func/convnet/Conv/weights*&
_output_shapes
: 
н
KOptimizer/over_options/q_func/convnet/Conv/weights/Adam_1/Initializer/ConstConst*%
valueB *    *
dtype0*;
_class1
/-loc:@over_options/q_func/convnet/Conv/weights*&
_output_shapes
: 
ъ
9Optimizer/over_options/q_func/convnet/Conv/weights/Adam_1
VariableV2*
shape: *
dtype0*
	container *
shared_name *;
_class1
/-loc:@over_options/q_func/convnet/Conv/weights*&
_output_shapes
: 
й
@Optimizer/over_options/q_func/convnet/Conv/weights/Adam_1/AssignAssign9Optimizer/over_options/q_func/convnet/Conv/weights/Adam_1KOptimizer/over_options/q_func/convnet/Conv/weights/Adam_1/Initializer/Const*
T0*
validate_shape(*
use_locking(*;
_class1
/-loc:@over_options/q_func/convnet/Conv/weights*&
_output_shapes
: 
у
>Optimizer/over_options/q_func/convnet/Conv/weights/Adam_1/readIdentity9Optimizer/over_options/q_func/convnet/Conv/weights/Adam_1*
T0*;
_class1
/-loc:@over_options/q_func/convnet/Conv/weights*&
_output_shapes
: 
—
HOptimizer/over_options/q_func/convnet/Conv/biases/Adam/Initializer/ConstConst*
valueB *    *
dtype0*:
_class0
.,loc:@over_options/q_func/convnet/Conv/biases*
_output_shapes
: 
ё
6Optimizer/over_options/q_func/convnet/Conv/biases/Adam
VariableV2*
shape: *
dtype0*
	container *
shared_name *:
_class0
.,loc:@over_options/q_func/convnet/Conv/biases*
_output_shapes
: 
”
=Optimizer/over_options/q_func/convnet/Conv/biases/Adam/AssignAssign6Optimizer/over_options/q_func/convnet/Conv/biases/AdamHOptimizer/over_options/q_func/convnet/Conv/biases/Adam/Initializer/Const*
T0*
validate_shape(*
use_locking(*:
_class0
.,loc:@over_options/q_func/convnet/Conv/biases*
_output_shapes
: 
а
;Optimizer/over_options/q_func/convnet/Conv/biases/Adam/readIdentity6Optimizer/over_options/q_func/convnet/Conv/biases/Adam*
T0*:
_class0
.,loc:@over_options/q_func/convnet/Conv/biases*
_output_shapes
: 
”
JOptimizer/over_options/q_func/convnet/Conv/biases/Adam_1/Initializer/ConstConst*
valueB *    *
dtype0*:
_class0
.,loc:@over_options/q_func/convnet/Conv/biases*
_output_shapes
: 
а
8Optimizer/over_options/q_func/convnet/Conv/biases/Adam_1
VariableV2*
shape: *
dtype0*
	container *
shared_name *:
_class0
.,loc:@over_options/q_func/convnet/Conv/biases*
_output_shapes
: 
ў
?Optimizer/over_options/q_func/convnet/Conv/biases/Adam_1/AssignAssign8Optimizer/over_options/q_func/convnet/Conv/biases/Adam_1JOptimizer/over_options/q_func/convnet/Conv/biases/Adam_1/Initializer/Const*
T0*
validate_shape(*
use_locking(*:
_class0
.,loc:@over_options/q_func/convnet/Conv/biases*
_output_shapes
: 
д
=Optimizer/over_options/q_func/convnet/Conv/biases/Adam_1/readIdentity8Optimizer/over_options/q_func/convnet/Conv/biases/Adam_1*
T0*:
_class0
.,loc:@over_options/q_func/convnet/Conv/biases*
_output_shapes
: 
п
KOptimizer/over_options/q_func/convnet/Conv_1/weights/Adam/Initializer/ConstConst*%
valueB @*    *
dtype0*=
_class3
1/loc:@over_options/q_func/convnet/Conv_1/weights*&
_output_shapes
: @
ь
9Optimizer/over_options/q_func/convnet/Conv_1/weights/Adam
VariableV2*
shape: @*
dtype0*
	container *
shared_name *=
_class3
1/loc:@over_options/q_func/convnet/Conv_1/weights*&
_output_shapes
: @
л
@Optimizer/over_options/q_func/convnet/Conv_1/weights/Adam/AssignAssign9Optimizer/over_options/q_func/convnet/Conv_1/weights/AdamKOptimizer/over_options/q_func/convnet/Conv_1/weights/Adam/Initializer/Const*
T0*
validate_shape(*
use_locking(*=
_class3
1/loc:@over_options/q_func/convnet/Conv_1/weights*&
_output_shapes
: @
х
>Optimizer/over_options/q_func/convnet/Conv_1/weights/Adam/readIdentity9Optimizer/over_options/q_func/convnet/Conv_1/weights/Adam*
T0*=
_class3
1/loc:@over_options/q_func/convnet/Conv_1/weights*&
_output_shapes
: @
с
MOptimizer/over_options/q_func/convnet/Conv_1/weights/Adam_1/Initializer/ConstConst*%
valueB @*    *
dtype0*=
_class3
1/loc:@over_options/q_func/convnet/Conv_1/weights*&
_output_shapes
: @
ю
;Optimizer/over_options/q_func/convnet/Conv_1/weights/Adam_1
VariableV2*
shape: @*
dtype0*
	container *
shared_name *=
_class3
1/loc:@over_options/q_func/convnet/Conv_1/weights*&
_output_shapes
: @
с
BOptimizer/over_options/q_func/convnet/Conv_1/weights/Adam_1/AssignAssign;Optimizer/over_options/q_func/convnet/Conv_1/weights/Adam_1MOptimizer/over_options/q_func/convnet/Conv_1/weights/Adam_1/Initializer/Const*
T0*
validate_shape(*
use_locking(*=
_class3
1/loc:@over_options/q_func/convnet/Conv_1/weights*&
_output_shapes
: @
щ
@Optimizer/over_options/q_func/convnet/Conv_1/weights/Adam_1/readIdentity;Optimizer/over_options/q_func/convnet/Conv_1/weights/Adam_1*
T0*=
_class3
1/loc:@over_options/q_func/convnet/Conv_1/weights*&
_output_shapes
: @
’
JOptimizer/over_options/q_func/convnet/Conv_1/biases/Adam/Initializer/ConstConst*
valueB@*    *
dtype0*<
_class2
0.loc:@over_options/q_func/convnet/Conv_1/biases*
_output_shapes
:@
в
8Optimizer/over_options/q_func/convnet/Conv_1/biases/Adam
VariableV2*
shape:@*
dtype0*
	container *
shared_name *<
_class2
0.loc:@over_options/q_func/convnet/Conv_1/biases*
_output_shapes
:@
џ
?Optimizer/over_options/q_func/convnet/Conv_1/biases/Adam/AssignAssign8Optimizer/over_options/q_func/convnet/Conv_1/biases/AdamJOptimizer/over_options/q_func/convnet/Conv_1/biases/Adam/Initializer/Const*
T0*
validate_shape(*
use_locking(*<
_class2
0.loc:@over_options/q_func/convnet/Conv_1/biases*
_output_shapes
:@
ж
=Optimizer/over_options/q_func/convnet/Conv_1/biases/Adam/readIdentity8Optimizer/over_options/q_func/convnet/Conv_1/biases/Adam*
T0*<
_class2
0.loc:@over_options/q_func/convnet/Conv_1/biases*
_output_shapes
:@
„
LOptimizer/over_options/q_func/convnet/Conv_1/biases/Adam_1/Initializer/ConstConst*
valueB@*    *
dtype0*<
_class2
0.loc:@over_options/q_func/convnet/Conv_1/biases*
_output_shapes
:@
д
:Optimizer/over_options/q_func/convnet/Conv_1/biases/Adam_1
VariableV2*
shape:@*
dtype0*
	container *
shared_name *<
_class2
0.loc:@over_options/q_func/convnet/Conv_1/biases*
_output_shapes
:@
б
AOptimizer/over_options/q_func/convnet/Conv_1/biases/Adam_1/AssignAssign:Optimizer/over_options/q_func/convnet/Conv_1/biases/Adam_1LOptimizer/over_options/q_func/convnet/Conv_1/biases/Adam_1/Initializer/Const*
T0*
validate_shape(*
use_locking(*<
_class2
0.loc:@over_options/q_func/convnet/Conv_1/biases*
_output_shapes
:@
к
?Optimizer/over_options/q_func/convnet/Conv_1/biases/Adam_1/readIdentity:Optimizer/over_options/q_func/convnet/Conv_1/biases/Adam_1*
T0*<
_class2
0.loc:@over_options/q_func/convnet/Conv_1/biases*
_output_shapes
:@
п
KOptimizer/over_options/q_func/convnet/Conv_2/weights/Adam/Initializer/ConstConst*%
valueB@@*    *
dtype0*=
_class3
1/loc:@over_options/q_func/convnet/Conv_2/weights*&
_output_shapes
:@@
ь
9Optimizer/over_options/q_func/convnet/Conv_2/weights/Adam
VariableV2*
shape:@@*
dtype0*
	container *
shared_name *=
_class3
1/loc:@over_options/q_func/convnet/Conv_2/weights*&
_output_shapes
:@@
л
@Optimizer/over_options/q_func/convnet/Conv_2/weights/Adam/AssignAssign9Optimizer/over_options/q_func/convnet/Conv_2/weights/AdamKOptimizer/over_options/q_func/convnet/Conv_2/weights/Adam/Initializer/Const*
T0*
validate_shape(*
use_locking(*=
_class3
1/loc:@over_options/q_func/convnet/Conv_2/weights*&
_output_shapes
:@@
х
>Optimizer/over_options/q_func/convnet/Conv_2/weights/Adam/readIdentity9Optimizer/over_options/q_func/convnet/Conv_2/weights/Adam*
T0*=
_class3
1/loc:@over_options/q_func/convnet/Conv_2/weights*&
_output_shapes
:@@
с
MOptimizer/over_options/q_func/convnet/Conv_2/weights/Adam_1/Initializer/ConstConst*%
valueB@@*    *
dtype0*=
_class3
1/loc:@over_options/q_func/convnet/Conv_2/weights*&
_output_shapes
:@@
ю
;Optimizer/over_options/q_func/convnet/Conv_2/weights/Adam_1
VariableV2*
shape:@@*
dtype0*
	container *
shared_name *=
_class3
1/loc:@over_options/q_func/convnet/Conv_2/weights*&
_output_shapes
:@@
с
BOptimizer/over_options/q_func/convnet/Conv_2/weights/Adam_1/AssignAssign;Optimizer/over_options/q_func/convnet/Conv_2/weights/Adam_1MOptimizer/over_options/q_func/convnet/Conv_2/weights/Adam_1/Initializer/Const*
T0*
validate_shape(*
use_locking(*=
_class3
1/loc:@over_options/q_func/convnet/Conv_2/weights*&
_output_shapes
:@@
щ
@Optimizer/over_options/q_func/convnet/Conv_2/weights/Adam_1/readIdentity;Optimizer/over_options/q_func/convnet/Conv_2/weights/Adam_1*
T0*=
_class3
1/loc:@over_options/q_func/convnet/Conv_2/weights*&
_output_shapes
:@@
’
JOptimizer/over_options/q_func/convnet/Conv_2/biases/Adam/Initializer/ConstConst*
valueB@*    *
dtype0*<
_class2
0.loc:@over_options/q_func/convnet/Conv_2/biases*
_output_shapes
:@
в
8Optimizer/over_options/q_func/convnet/Conv_2/biases/Adam
VariableV2*
shape:@*
dtype0*
	container *
shared_name *<
_class2
0.loc:@over_options/q_func/convnet/Conv_2/biases*
_output_shapes
:@
џ
?Optimizer/over_options/q_func/convnet/Conv_2/biases/Adam/AssignAssign8Optimizer/over_options/q_func/convnet/Conv_2/biases/AdamJOptimizer/over_options/q_func/convnet/Conv_2/biases/Adam/Initializer/Const*
T0*
validate_shape(*
use_locking(*<
_class2
0.loc:@over_options/q_func/convnet/Conv_2/biases*
_output_shapes
:@
ж
=Optimizer/over_options/q_func/convnet/Conv_2/biases/Adam/readIdentity8Optimizer/over_options/q_func/convnet/Conv_2/biases/Adam*
T0*<
_class2
0.loc:@over_options/q_func/convnet/Conv_2/biases*
_output_shapes
:@
„
LOptimizer/over_options/q_func/convnet/Conv_2/biases/Adam_1/Initializer/ConstConst*
valueB@*    *
dtype0*<
_class2
0.loc:@over_options/q_func/convnet/Conv_2/biases*
_output_shapes
:@
д
:Optimizer/over_options/q_func/convnet/Conv_2/biases/Adam_1
VariableV2*
shape:@*
dtype0*
	container *
shared_name *<
_class2
0.loc:@over_options/q_func/convnet/Conv_2/biases*
_output_shapes
:@
б
AOptimizer/over_options/q_func/convnet/Conv_2/biases/Adam_1/AssignAssign:Optimizer/over_options/q_func/convnet/Conv_2/biases/Adam_1LOptimizer/over_options/q_func/convnet/Conv_2/biases/Adam_1/Initializer/Const*
T0*
validate_shape(*
use_locking(*<
_class2
0.loc:@over_options/q_func/convnet/Conv_2/biases*
_output_shapes
:@
к
?Optimizer/over_options/q_func/convnet/Conv_2/biases/Adam_1/readIdentity:Optimizer/over_options/q_func/convnet/Conv_2/biases/Adam_1*
T0*<
_class2
0.loc:@over_options/q_func/convnet/Conv_2/biases*
_output_shapes
:@
€
YOptimizer/over_options/q_func/action_value/fully_connected/weights/Adam/Initializer/ConstConst*
valueB
АА*    *
dtype0*K
_classA
?=loc:@over_options/q_func/action_value/fully_connected/weights* 
_output_shapes
:
АА
М
GOptimizer/over_options/q_func/action_value/fully_connected/weights/Adam
VariableV2*
shape:
АА*
dtype0*
	container *
shared_name *K
_classA
?=loc:@over_options/q_func/action_value/fully_connected/weights* 
_output_shapes
:
АА
Э
NOptimizer/over_options/q_func/action_value/fully_connected/weights/Adam/AssignAssignGOptimizer/over_options/q_func/action_value/fully_connected/weights/AdamYOptimizer/over_options/q_func/action_value/fully_connected/weights/Adam/Initializer/Const*
T0*
validate_shape(*
use_locking(*K
_classA
?=loc:@over_options/q_func/action_value/fully_connected/weights* 
_output_shapes
:
АА
Щ
LOptimizer/over_options/q_func/action_value/fully_connected/weights/Adam/readIdentityGOptimizer/over_options/q_func/action_value/fully_connected/weights/Adam*
T0*K
_classA
?=loc:@over_options/q_func/action_value/fully_connected/weights* 
_output_shapes
:
АА
Б
[Optimizer/over_options/q_func/action_value/fully_connected/weights/Adam_1/Initializer/ConstConst*
valueB
АА*    *
dtype0*K
_classA
?=loc:@over_options/q_func/action_value/fully_connected/weights* 
_output_shapes
:
АА
О
IOptimizer/over_options/q_func/action_value/fully_connected/weights/Adam_1
VariableV2*
shape:
АА*
dtype0*
	container *
shared_name *K
_classA
?=loc:@over_options/q_func/action_value/fully_connected/weights* 
_output_shapes
:
АА
£
POptimizer/over_options/q_func/action_value/fully_connected/weights/Adam_1/AssignAssignIOptimizer/over_options/q_func/action_value/fully_connected/weights/Adam_1[Optimizer/over_options/q_func/action_value/fully_connected/weights/Adam_1/Initializer/Const*
T0*
validate_shape(*
use_locking(*K
_classA
?=loc:@over_options/q_func/action_value/fully_connected/weights* 
_output_shapes
:
АА
Э
NOptimizer/over_options/q_func/action_value/fully_connected/weights/Adam_1/readIdentityIOptimizer/over_options/q_func/action_value/fully_connected/weights/Adam_1*
T0*K
_classA
?=loc:@over_options/q_func/action_value/fully_connected/weights* 
_output_shapes
:
АА
у
XOptimizer/over_options/q_func/action_value/fully_connected/biases/Adam/Initializer/ConstConst*
valueBА*    *
dtype0*J
_class@
><loc:@over_options/q_func/action_value/fully_connected/biases*
_output_shapes	
:А
А
FOptimizer/over_options/q_func/action_value/fully_connected/biases/Adam
VariableV2*
shape:А*
dtype0*
	container *
shared_name *J
_class@
><loc:@over_options/q_func/action_value/fully_connected/biases*
_output_shapes	
:А
Ф
MOptimizer/over_options/q_func/action_value/fully_connected/biases/Adam/AssignAssignFOptimizer/over_options/q_func/action_value/fully_connected/biases/AdamXOptimizer/over_options/q_func/action_value/fully_connected/biases/Adam/Initializer/Const*
T0*
validate_shape(*
use_locking(*J
_class@
><loc:@over_options/q_func/action_value/fully_connected/biases*
_output_shapes	
:А
С
KOptimizer/over_options/q_func/action_value/fully_connected/biases/Adam/readIdentityFOptimizer/over_options/q_func/action_value/fully_connected/biases/Adam*
T0*J
_class@
><loc:@over_options/q_func/action_value/fully_connected/biases*
_output_shapes	
:А
х
ZOptimizer/over_options/q_func/action_value/fully_connected/biases/Adam_1/Initializer/ConstConst*
valueBА*    *
dtype0*J
_class@
><loc:@over_options/q_func/action_value/fully_connected/biases*
_output_shapes	
:А
В
HOptimizer/over_options/q_func/action_value/fully_connected/biases/Adam_1
VariableV2*
shape:А*
dtype0*
	container *
shared_name *J
_class@
><loc:@over_options/q_func/action_value/fully_connected/biases*
_output_shapes	
:А
Ъ
OOptimizer/over_options/q_func/action_value/fully_connected/biases/Adam_1/AssignAssignHOptimizer/over_options/q_func/action_value/fully_connected/biases/Adam_1ZOptimizer/over_options/q_func/action_value/fully_connected/biases/Adam_1/Initializer/Const*
T0*
validate_shape(*
use_locking(*J
_class@
><loc:@over_options/q_func/action_value/fully_connected/biases*
_output_shapes	
:А
Х
MOptimizer/over_options/q_func/action_value/fully_connected/biases/Adam_1/readIdentityHOptimizer/over_options/q_func/action_value/fully_connected/biases/Adam_1*
T0*J
_class@
><loc:@over_options/q_func/action_value/fully_connected/biases*
_output_shapes	
:А
Б
[Optimizer/over_options/q_func/action_value/fully_connected_1/weights/Adam/Initializer/ConstConst*
valueB	А*    *
dtype0*M
_classC
A?loc:@over_options/q_func/action_value/fully_connected_1/weights*
_output_shapes
:	А
О
IOptimizer/over_options/q_func/action_value/fully_connected_1/weights/Adam
VariableV2*
shape:	А*
dtype0*
	container *
shared_name *M
_classC
A?loc:@over_options/q_func/action_value/fully_connected_1/weights*
_output_shapes
:	А
§
POptimizer/over_options/q_func/action_value/fully_connected_1/weights/Adam/AssignAssignIOptimizer/over_options/q_func/action_value/fully_connected_1/weights/Adam[Optimizer/over_options/q_func/action_value/fully_connected_1/weights/Adam/Initializer/Const*
T0*
validate_shape(*
use_locking(*M
_classC
A?loc:@over_options/q_func/action_value/fully_connected_1/weights*
_output_shapes
:	А
Ю
NOptimizer/over_options/q_func/action_value/fully_connected_1/weights/Adam/readIdentityIOptimizer/over_options/q_func/action_value/fully_connected_1/weights/Adam*
T0*M
_classC
A?loc:@over_options/q_func/action_value/fully_connected_1/weights*
_output_shapes
:	А
Г
]Optimizer/over_options/q_func/action_value/fully_connected_1/weights/Adam_1/Initializer/ConstConst*
valueB	А*    *
dtype0*M
_classC
A?loc:@over_options/q_func/action_value/fully_connected_1/weights*
_output_shapes
:	А
Р
KOptimizer/over_options/q_func/action_value/fully_connected_1/weights/Adam_1
VariableV2*
shape:	А*
dtype0*
	container *
shared_name *M
_classC
A?loc:@over_options/q_func/action_value/fully_connected_1/weights*
_output_shapes
:	А
™
ROptimizer/over_options/q_func/action_value/fully_connected_1/weights/Adam_1/AssignAssignKOptimizer/over_options/q_func/action_value/fully_connected_1/weights/Adam_1]Optimizer/over_options/q_func/action_value/fully_connected_1/weights/Adam_1/Initializer/Const*
T0*
validate_shape(*
use_locking(*M
_classC
A?loc:@over_options/q_func/action_value/fully_connected_1/weights*
_output_shapes
:	А
Ґ
POptimizer/over_options/q_func/action_value/fully_connected_1/weights/Adam_1/readIdentityKOptimizer/over_options/q_func/action_value/fully_connected_1/weights/Adam_1*
T0*M
_classC
A?loc:@over_options/q_func/action_value/fully_connected_1/weights*
_output_shapes
:	А
х
ZOptimizer/over_options/q_func/action_value/fully_connected_1/biases/Adam/Initializer/ConstConst*
valueB*    *
dtype0*L
_classB
@>loc:@over_options/q_func/action_value/fully_connected_1/biases*
_output_shapes
:
В
HOptimizer/over_options/q_func/action_value/fully_connected_1/biases/Adam
VariableV2*
shape:*
dtype0*
	container *
shared_name *L
_classB
@>loc:@over_options/q_func/action_value/fully_connected_1/biases*
_output_shapes
:
Ы
OOptimizer/over_options/q_func/action_value/fully_connected_1/biases/Adam/AssignAssignHOptimizer/over_options/q_func/action_value/fully_connected_1/biases/AdamZOptimizer/over_options/q_func/action_value/fully_connected_1/biases/Adam/Initializer/Const*
T0*
validate_shape(*
use_locking(*L
_classB
@>loc:@over_options/q_func/action_value/fully_connected_1/biases*
_output_shapes
:
Ц
MOptimizer/over_options/q_func/action_value/fully_connected_1/biases/Adam/readIdentityHOptimizer/over_options/q_func/action_value/fully_connected_1/biases/Adam*
T0*L
_classB
@>loc:@over_options/q_func/action_value/fully_connected_1/biases*
_output_shapes
:
ч
\Optimizer/over_options/q_func/action_value/fully_connected_1/biases/Adam_1/Initializer/ConstConst*
valueB*    *
dtype0*L
_classB
@>loc:@over_options/q_func/action_value/fully_connected_1/biases*
_output_shapes
:
Д
JOptimizer/over_options/q_func/action_value/fully_connected_1/biases/Adam_1
VariableV2*
shape:*
dtype0*
	container *
shared_name *L
_classB
@>loc:@over_options/q_func/action_value/fully_connected_1/biases*
_output_shapes
:
°
QOptimizer/over_options/q_func/action_value/fully_connected_1/biases/Adam_1/AssignAssignJOptimizer/over_options/q_func/action_value/fully_connected_1/biases/Adam_1\Optimizer/over_options/q_func/action_value/fully_connected_1/biases/Adam_1/Initializer/Const*
T0*
validate_shape(*
use_locking(*L
_classB
@>loc:@over_options/q_func/action_value/fully_connected_1/biases*
_output_shapes
:
Ъ
OOptimizer/over_options/q_func/action_value/fully_connected_1/biases/Adam_1/readIdentityJOptimizer/over_options/q_func/action_value/fully_connected_1/biases/Adam_1*
T0*L
_classB
@>loc:@over_options/q_func/action_value/fully_connected_1/biases*
_output_shapes
:
Y
Optimizer/Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
Y
Optimizer/Adam/beta2Const*
valueB
 *wЊ?*
dtype0*
_output_shapes
: 
[
Optimizer/Adam/epsilonConst*
valueB
 *Ј—8*
dtype0*
_output_shapes
: 
Ч
HOptimizer/Adam/update_over_options/q_func/convnet/Conv/weights/ApplyAdam	ApplyAdam(over_options/q_func/convnet/Conv/weights7Optimizer/over_options/q_func/convnet/Conv/weights/Adam9Optimizer/over_options/q_func/convnet/Conv/weights/Adam_1Optimizer/beta1_power/readOptimizer/beta2_power/readlearning_rateOptimizer/Adam/beta1Optimizer/Adam/beta2Optimizer/Adam/epsilonOptimizer/clip_by_norm*
T0*
use_locking( *;
_class1
/-loc:@over_options/q_func/convnet/Conv/weights*&
_output_shapes
: 
И
GOptimizer/Adam/update_over_options/q_func/convnet/Conv/biases/ApplyAdam	ApplyAdam'over_options/q_func/convnet/Conv/biases6Optimizer/over_options/q_func/convnet/Conv/biases/Adam8Optimizer/over_options/q_func/convnet/Conv/biases/Adam_1Optimizer/beta1_power/readOptimizer/beta2_power/readlearning_rateOptimizer/Adam/beta1Optimizer/Adam/beta2Optimizer/Adam/epsilonOptimizer/clip_by_norm_1*
T0*
use_locking( *:
_class0
.,loc:@over_options/q_func/convnet/Conv/biases*
_output_shapes
: 
£
JOptimizer/Adam/update_over_options/q_func/convnet/Conv_1/weights/ApplyAdam	ApplyAdam*over_options/q_func/convnet/Conv_1/weights9Optimizer/over_options/q_func/convnet/Conv_1/weights/Adam;Optimizer/over_options/q_func/convnet/Conv_1/weights/Adam_1Optimizer/beta1_power/readOptimizer/beta2_power/readlearning_rateOptimizer/Adam/beta1Optimizer/Adam/beta2Optimizer/Adam/epsilonOptimizer/clip_by_norm_2*
T0*
use_locking( *=
_class3
1/loc:@over_options/q_func/convnet/Conv_1/weights*&
_output_shapes
: @
Т
IOptimizer/Adam/update_over_options/q_func/convnet/Conv_1/biases/ApplyAdam	ApplyAdam)over_options/q_func/convnet/Conv_1/biases8Optimizer/over_options/q_func/convnet/Conv_1/biases/Adam:Optimizer/over_options/q_func/convnet/Conv_1/biases/Adam_1Optimizer/beta1_power/readOptimizer/beta2_power/readlearning_rateOptimizer/Adam/beta1Optimizer/Adam/beta2Optimizer/Adam/epsilonOptimizer/clip_by_norm_3*
T0*
use_locking( *<
_class2
0.loc:@over_options/q_func/convnet/Conv_1/biases*
_output_shapes
:@
£
JOptimizer/Adam/update_over_options/q_func/convnet/Conv_2/weights/ApplyAdam	ApplyAdam*over_options/q_func/convnet/Conv_2/weights9Optimizer/over_options/q_func/convnet/Conv_2/weights/Adam;Optimizer/over_options/q_func/convnet/Conv_2/weights/Adam_1Optimizer/beta1_power/readOptimizer/beta2_power/readlearning_rateOptimizer/Adam/beta1Optimizer/Adam/beta2Optimizer/Adam/epsilonOptimizer/clip_by_norm_4*
T0*
use_locking( *=
_class3
1/loc:@over_options/q_func/convnet/Conv_2/weights*&
_output_shapes
:@@
Т
IOptimizer/Adam/update_over_options/q_func/convnet/Conv_2/biases/ApplyAdam	ApplyAdam)over_options/q_func/convnet/Conv_2/biases8Optimizer/over_options/q_func/convnet/Conv_2/biases/Adam:Optimizer/over_options/q_func/convnet/Conv_2/biases/Adam_1Optimizer/beta1_power/readOptimizer/beta2_power/readlearning_rateOptimizer/Adam/beta1Optimizer/Adam/beta2Optimizer/Adam/epsilonOptimizer/clip_by_norm_5*
T0*
use_locking( *<
_class2
0.loc:@over_options/q_func/convnet/Conv_2/biases*
_output_shapes
:@
г
XOptimizer/Adam/update_over_options/q_func/action_value/fully_connected/weights/ApplyAdam	ApplyAdam8over_options/q_func/action_value/fully_connected/weightsGOptimizer/over_options/q_func/action_value/fully_connected/weights/AdamIOptimizer/over_options/q_func/action_value/fully_connected/weights/Adam_1Optimizer/beta1_power/readOptimizer/beta2_power/readlearning_rateOptimizer/Adam/beta1Optimizer/Adam/beta2Optimizer/Adam/epsilonOptimizer/clip_by_norm_6*
T0*
use_locking( *K
_classA
?=loc:@over_options/q_func/action_value/fully_connected/weights* 
_output_shapes
:
АА
ў
WOptimizer/Adam/update_over_options/q_func/action_value/fully_connected/biases/ApplyAdam	ApplyAdam7over_options/q_func/action_value/fully_connected/biasesFOptimizer/over_options/q_func/action_value/fully_connected/biases/AdamHOptimizer/over_options/q_func/action_value/fully_connected/biases/Adam_1Optimizer/beta1_power/readOptimizer/beta2_power/readlearning_rateOptimizer/Adam/beta1Optimizer/Adam/beta2Optimizer/Adam/epsilonOptimizer/clip_by_norm_7*
T0*
use_locking( *J
_class@
><loc:@over_options/q_func/action_value/fully_connected/biases*
_output_shapes	
:А
м
ZOptimizer/Adam/update_over_options/q_func/action_value/fully_connected_1/weights/ApplyAdam	ApplyAdam:over_options/q_func/action_value/fully_connected_1/weightsIOptimizer/over_options/q_func/action_value/fully_connected_1/weights/AdamKOptimizer/over_options/q_func/action_value/fully_connected_1/weights/Adam_1Optimizer/beta1_power/readOptimizer/beta2_power/readlearning_rateOptimizer/Adam/beta1Optimizer/Adam/beta2Optimizer/Adam/epsilonOptimizer/clip_by_norm_8*
T0*
use_locking( *M
_classC
A?loc:@over_options/q_func/action_value/fully_connected_1/weights*
_output_shapes
:	А
в
YOptimizer/Adam/update_over_options/q_func/action_value/fully_connected_1/biases/ApplyAdam	ApplyAdam9over_options/q_func/action_value/fully_connected_1/biasesHOptimizer/over_options/q_func/action_value/fully_connected_1/biases/AdamJOptimizer/over_options/q_func/action_value/fully_connected_1/biases/Adam_1Optimizer/beta1_power/readOptimizer/beta2_power/readlearning_rateOptimizer/Adam/beta1Optimizer/Adam/beta2Optimizer/Adam/epsilonOptimizer/clip_by_norm_9*
T0*
use_locking( *L
_classB
@>loc:@over_options/q_func/action_value/fully_connected_1/biases*
_output_shapes
:
ё
Optimizer/Adam/mulMulOptimizer/beta1_power/readOptimizer/Adam/beta1I^Optimizer/Adam/update_over_options/q_func/convnet/Conv/weights/ApplyAdamH^Optimizer/Adam/update_over_options/q_func/convnet/Conv/biases/ApplyAdamK^Optimizer/Adam/update_over_options/q_func/convnet/Conv_1/weights/ApplyAdamJ^Optimizer/Adam/update_over_options/q_func/convnet/Conv_1/biases/ApplyAdamK^Optimizer/Adam/update_over_options/q_func/convnet/Conv_2/weights/ApplyAdamJ^Optimizer/Adam/update_over_options/q_func/convnet/Conv_2/biases/ApplyAdamY^Optimizer/Adam/update_over_options/q_func/action_value/fully_connected/weights/ApplyAdamX^Optimizer/Adam/update_over_options/q_func/action_value/fully_connected/biases/ApplyAdam[^Optimizer/Adam/update_over_options/q_func/action_value/fully_connected_1/weights/ApplyAdamZ^Optimizer/Adam/update_over_options/q_func/action_value/fully_connected_1/biases/ApplyAdam*
T0*;
_class1
/-loc:@over_options/q_func/convnet/Conv/weights*
_output_shapes
: 
—
Optimizer/Adam/AssignAssignOptimizer/beta1_powerOptimizer/Adam/mul*
T0*
validate_shape(*
use_locking( *;
_class1
/-loc:@over_options/q_func/convnet/Conv/weights*
_output_shapes
: 
а
Optimizer/Adam/mul_1MulOptimizer/beta2_power/readOptimizer/Adam/beta2I^Optimizer/Adam/update_over_options/q_func/convnet/Conv/weights/ApplyAdamH^Optimizer/Adam/update_over_options/q_func/convnet/Conv/biases/ApplyAdamK^Optimizer/Adam/update_over_options/q_func/convnet/Conv_1/weights/ApplyAdamJ^Optimizer/Adam/update_over_options/q_func/convnet/Conv_1/biases/ApplyAdamK^Optimizer/Adam/update_over_options/q_func/convnet/Conv_2/weights/ApplyAdamJ^Optimizer/Adam/update_over_options/q_func/convnet/Conv_2/biases/ApplyAdamY^Optimizer/Adam/update_over_options/q_func/action_value/fully_connected/weights/ApplyAdamX^Optimizer/Adam/update_over_options/q_func/action_value/fully_connected/biases/ApplyAdam[^Optimizer/Adam/update_over_options/q_func/action_value/fully_connected_1/weights/ApplyAdamZ^Optimizer/Adam/update_over_options/q_func/action_value/fully_connected_1/biases/ApplyAdam*
T0*;
_class1
/-loc:@over_options/q_func/convnet/Conv/weights*
_output_shapes
: 
’
Optimizer/Adam/Assign_1AssignOptimizer/beta2_powerOptimizer/Adam/mul_1*
T0*
validate_shape(*
use_locking( *;
_class1
/-loc:@over_options/q_func/convnet/Conv/weights*
_output_shapes
: 
э
Optimizer/AdamNoOpI^Optimizer/Adam/update_over_options/q_func/convnet/Conv/weights/ApplyAdamH^Optimizer/Adam/update_over_options/q_func/convnet/Conv/biases/ApplyAdamK^Optimizer/Adam/update_over_options/q_func/convnet/Conv_1/weights/ApplyAdamJ^Optimizer/Adam/update_over_options/q_func/convnet/Conv_1/biases/ApplyAdamK^Optimizer/Adam/update_over_options/q_func/convnet/Conv_2/weights/ApplyAdamJ^Optimizer/Adam/update_over_options/q_func/convnet/Conv_2/biases/ApplyAdamY^Optimizer/Adam/update_over_options/q_func/action_value/fully_connected/weights/ApplyAdamX^Optimizer/Adam/update_over_options/q_func/action_value/fully_connected/biases/ApplyAdam[^Optimizer/Adam/update_over_options/q_func/action_value/fully_connected_1/weights/ApplyAdamZ^Optimizer/Adam/update_over_options/q_func/action_value/fully_connected_1/biases/ApplyAdam^Optimizer/Adam/Assign^Optimizer/Adam/Assign_1
Ц
AssignAssign1target_q_func/action_value/fully_connected/biases<over_options/q_func/action_value/fully_connected/biases/read*
T0*
validate_shape(*
use_locking( *D
_class:
86loc:@target_q_func/action_value/fully_connected/biases*
_output_shapes	
:А
†
Assign_1Assign2target_q_func/action_value/fully_connected/weights=over_options/q_func/action_value/fully_connected/weights/read*
T0*
validate_shape(*
use_locking( *E
_class;
97loc:@target_q_func/action_value/fully_connected/weights* 
_output_shapes
:
АА
Э
Assign_2Assign3target_q_func/action_value/fully_connected_1/biases>over_options/q_func/action_value/fully_connected_1/biases/read*
T0*
validate_shape(*
use_locking( *F
_class<
:8loc:@target_q_func/action_value/fully_connected_1/biases*
_output_shapes
:
•
Assign_3Assign4target_q_func/action_value/fully_connected_1/weights?over_options/q_func/action_value/fully_connected_1/weights/read*
T0*
validate_shape(*
use_locking( *G
_class=
;9loc:@target_q_func/action_value/fully_connected_1/weights*
_output_shapes
:	А
з
Assign_4Assign!target_q_func/convnet/Conv/biases,over_options/q_func/convnet/Conv/biases/read*
T0*
validate_shape(*
use_locking( *4
_class*
(&loc:@target_q_func/convnet/Conv/biases*
_output_shapes
: 
ц
Assign_5Assign"target_q_func/convnet/Conv/weights-over_options/q_func/convnet/Conv/weights/read*
T0*
validate_shape(*
use_locking( *5
_class+
)'loc:@target_q_func/convnet/Conv/weights*&
_output_shapes
: 
н
Assign_6Assign#target_q_func/convnet/Conv_1/biases.over_options/q_func/convnet/Conv_1/biases/read*
T0*
validate_shape(*
use_locking( *6
_class,
*(loc:@target_q_func/convnet/Conv_1/biases*
_output_shapes
:@
ь
Assign_7Assign$target_q_func/convnet/Conv_1/weights/over_options/q_func/convnet/Conv_1/weights/read*
T0*
validate_shape(*
use_locking( *7
_class-
+)loc:@target_q_func/convnet/Conv_1/weights*&
_output_shapes
: @
н
Assign_8Assign#target_q_func/convnet/Conv_2/biases.over_options/q_func/convnet/Conv_2/biases/read*
T0*
validate_shape(*
use_locking( *6
_class,
*(loc:@target_q_func/convnet/Conv_2/biases*
_output_shapes
:@
ь
Assign_9Assign$target_q_func/convnet/Conv_2/weights/over_options/q_func/convnet/Conv_2/weights/read*
T0*
validate_shape(*
use_locking( *7
_class-
+)loc:@target_q_func/convnet/Conv_2/weights*&
_output_shapes
:@@
Х
!Update_target_fn/update_target_fnNoOp^Assign	^Assign_1	^Assign_2	^Assign_3	^Assign_4	^Assign_5	^Assign_6	^Assign_7	^Assign_8	^Assign_9
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
Ќ
save/SaveV2/tensor_namesConst*А
valueцBу
B7over_options/q_func/action_value/fully_connected/biasesB8over_options/q_func/action_value/fully_connected/weightsB9over_options/q_func/action_value/fully_connected_1/biasesB:over_options/q_func/action_value/fully_connected_1/weightsB'over_options/q_func/convnet/Conv/biasesB(over_options/q_func/convnet/Conv/weightsB)over_options/q_func/convnet/Conv_1/biasesB*over_options/q_func/convnet/Conv_1/weightsB)over_options/q_func/convnet/Conv_2/biasesB*over_options/q_func/convnet/Conv_2/weights*
dtype0*
_output_shapes
:

w
save/SaveV2/shape_and_slicesConst*'
valueB
B B B B B B B B B B *
dtype0*
_output_shapes
:

ё
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slices7over_options/q_func/action_value/fully_connected/biases8over_options/q_func/action_value/fully_connected/weights9over_options/q_func/action_value/fully_connected_1/biases:over_options/q_func/action_value/fully_connected_1/weights'over_options/q_func/convnet/Conv/biases(over_options/q_func/convnet/Conv/weights)over_options/q_func/convnet/Conv_1/biases*over_options/q_func/convnet/Conv_1/weights)over_options/q_func/convnet/Conv_2/biases*over_options/q_func/convnet/Conv_2/weights*
dtypes
2

}
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const*
_output_shapes
: 
Ы
save/RestoreV2/tensor_namesConst*L
valueCBAB7over_options/q_func/action_value/fully_connected/biases*
dtype0*
_output_shapes
:
h
save/RestoreV2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Р
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:
щ
save/AssignAssign7over_options/q_func/action_value/fully_connected/biasessave/RestoreV2*
T0*
validate_shape(*
use_locking(*J
_class@
><loc:@over_options/q_func/action_value/fully_connected/biases*
_output_shapes	
:А
Ю
save/RestoreV2_1/tensor_namesConst*M
valueDBBB8over_options/q_func/action_value/fully_connected/weights*
dtype0*
_output_shapes
:
j
!save/RestoreV2_1/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ц
save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes
:
Д
save/Assign_1Assign8over_options/q_func/action_value/fully_connected/weightssave/RestoreV2_1*
T0*
validate_shape(*
use_locking(*K
_classA
?=loc:@over_options/q_func/action_value/fully_connected/weights* 
_output_shapes
:
АА
Я
save/RestoreV2_2/tensor_namesConst*N
valueEBCB9over_options/q_func/action_value/fully_connected_1/biases*
dtype0*
_output_shapes
:
j
!save/RestoreV2_2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ц
save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
dtypes
2*
_output_shapes
:
А
save/Assign_2Assign9over_options/q_func/action_value/fully_connected_1/biasessave/RestoreV2_2*
T0*
validate_shape(*
use_locking(*L
_classB
@>loc:@over_options/q_func/action_value/fully_connected_1/biases*
_output_shapes
:
†
save/RestoreV2_3/tensor_namesConst*O
valueFBDB:over_options/q_func/action_value/fully_connected_1/weights*
dtype0*
_output_shapes
:
j
!save/RestoreV2_3/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ц
save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
dtypes
2*
_output_shapes
:
З
save/Assign_3Assign:over_options/q_func/action_value/fully_connected_1/weightssave/RestoreV2_3*
T0*
validate_shape(*
use_locking(*M
_classC
A?loc:@over_options/q_func/action_value/fully_connected_1/weights*
_output_shapes
:	А
Н
save/RestoreV2_4/tensor_namesConst*<
value3B1B'over_options/q_func/convnet/Conv/biases*
dtype0*
_output_shapes
:
j
!save/RestoreV2_4/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ц
save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
dtypes
2*
_output_shapes
:
№
save/Assign_4Assign'over_options/q_func/convnet/Conv/biasessave/RestoreV2_4*
T0*
validate_shape(*
use_locking(*:
_class0
.,loc:@over_options/q_func/convnet/Conv/biases*
_output_shapes
: 
О
save/RestoreV2_5/tensor_namesConst*=
value4B2B(over_options/q_func/convnet/Conv/weights*
dtype0*
_output_shapes
:
j
!save/RestoreV2_5/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ц
save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
dtypes
2*
_output_shapes
:
к
save/Assign_5Assign(over_options/q_func/convnet/Conv/weightssave/RestoreV2_5*
T0*
validate_shape(*
use_locking(*;
_class1
/-loc:@over_options/q_func/convnet/Conv/weights*&
_output_shapes
: 
П
save/RestoreV2_6/tensor_namesConst*>
value5B3B)over_options/q_func/convnet/Conv_1/biases*
dtype0*
_output_shapes
:
j
!save/RestoreV2_6/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ц
save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
dtypes
2*
_output_shapes
:
а
save/Assign_6Assign)over_options/q_func/convnet/Conv_1/biasessave/RestoreV2_6*
T0*
validate_shape(*
use_locking(*<
_class2
0.loc:@over_options/q_func/convnet/Conv_1/biases*
_output_shapes
:@
Р
save/RestoreV2_7/tensor_namesConst*?
value6B4B*over_options/q_func/convnet/Conv_1/weights*
dtype0*
_output_shapes
:
j
!save/RestoreV2_7/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ц
save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
dtypes
2*
_output_shapes
:
о
save/Assign_7Assign*over_options/q_func/convnet/Conv_1/weightssave/RestoreV2_7*
T0*
validate_shape(*
use_locking(*=
_class3
1/loc:@over_options/q_func/convnet/Conv_1/weights*&
_output_shapes
: @
П
save/RestoreV2_8/tensor_namesConst*>
value5B3B)over_options/q_func/convnet/Conv_2/biases*
dtype0*
_output_shapes
:
j
!save/RestoreV2_8/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ц
save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
dtypes
2*
_output_shapes
:
а
save/Assign_8Assign)over_options/q_func/convnet/Conv_2/biasessave/RestoreV2_8*
T0*
validate_shape(*
use_locking(*<
_class2
0.loc:@over_options/q_func/convnet/Conv_2/biases*
_output_shapes
:@
Р
save/RestoreV2_9/tensor_namesConst*?
value6B4B*over_options/q_func/convnet/Conv_2/weights*
dtype0*
_output_shapes
:
j
!save/RestoreV2_9/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ц
save/RestoreV2_9	RestoreV2
save/Constsave/RestoreV2_9/tensor_names!save/RestoreV2_9/shape_and_slices*
dtypes
2*
_output_shapes
:
о
save/Assign_9Assign*over_options/q_func/convnet/Conv_2/weightssave/RestoreV2_9*
T0*
validate_shape(*
use_locking(*=
_class3
1/loc:@over_options/q_func/convnet/Conv_2/weights*&
_output_shapes
:@@
ґ
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
>
initNoOp0^over_options/q_func/convnet/Conv/weights/Assign
?
init_1NoOp/^over_options/q_func/convnet/Conv/biases/Assign
B
init_2NoOp2^over_options/q_func/convnet/Conv_1/weights/Assign
A
init_3NoOp1^over_options/q_func/convnet/Conv_1/biases/Assign
B
init_4NoOp2^over_options/q_func/convnet/Conv_2/weights/Assign
A
init_5NoOp1^over_options/q_func/convnet/Conv_2/biases/Assign
P
init_6NoOp@^over_options/q_func/action_value/fully_connected/weights/Assign
O
init_7NoOp?^over_options/q_func/action_value/fully_connected/biases/Assign
R
init_8NoOpB^over_options/q_func/action_value/fully_connected_1/weights/Assign
Q
init_9NoOpA^over_options/q_func/action_value/fully_connected_1/biases/Assign
;
init_10NoOp*^target_q_func/convnet/Conv/weights/Assign
:
init_11NoOp)^target_q_func/convnet/Conv/biases/Assign
=
init_12NoOp,^target_q_func/convnet/Conv_1/weights/Assign
<
init_13NoOp+^target_q_func/convnet/Conv_1/biases/Assign
=
init_14NoOp,^target_q_func/convnet/Conv_2/weights/Assign
<
init_15NoOp+^target_q_func/convnet/Conv_2/biases/Assign
K
init_16NoOp:^target_q_func/action_value/fully_connected/weights/Assign
J
init_17NoOp9^target_q_func/action_value/fully_connected/biases/Assign
M
init_18NoOp<^target_q_func/action_value/fully_connected_1/weights/Assign
L
init_19NoOp;^target_q_func/action_value/fully_connected_1/biases/Assign
.
init_20NoOp^Optimizer/beta1_power/Assign
.
init_21NoOp^Optimizer/beta2_power/Assign
P
init_22NoOp?^Optimizer/over_options/q_func/convnet/Conv/weights/Adam/Assign
R
init_23NoOpA^Optimizer/over_options/q_func/convnet/Conv/weights/Adam_1/Assign
O
init_24NoOp>^Optimizer/over_options/q_func/convnet/Conv/biases/Adam/Assign
Q
init_25NoOp@^Optimizer/over_options/q_func/convnet/Conv/biases/Adam_1/Assign
R
init_26NoOpA^Optimizer/over_options/q_func/convnet/Conv_1/weights/Adam/Assign
T
init_27NoOpC^Optimizer/over_options/q_func/convnet/Conv_1/weights/Adam_1/Assign
Q
init_28NoOp@^Optimizer/over_options/q_func/convnet/Conv_1/biases/Adam/Assign
S
init_29NoOpB^Optimizer/over_options/q_func/convnet/Conv_1/biases/Adam_1/Assign
R
init_30NoOpA^Optimizer/over_options/q_func/convnet/Conv_2/weights/Adam/Assign
T
init_31NoOpC^Optimizer/over_options/q_func/convnet/Conv_2/weights/Adam_1/Assign
Q
init_32NoOp@^Optimizer/over_options/q_func/convnet/Conv_2/biases/Adam/Assign
S
init_33NoOpB^Optimizer/over_options/q_func/convnet/Conv_2/biases/Adam_1/Assign
`
init_34NoOpO^Optimizer/over_options/q_func/action_value/fully_connected/weights/Adam/Assign
b
init_35NoOpQ^Optimizer/over_options/q_func/action_value/fully_connected/weights/Adam_1/Assign
_
init_36NoOpN^Optimizer/over_options/q_func/action_value/fully_connected/biases/Adam/Assign
a
init_37NoOpP^Optimizer/over_options/q_func/action_value/fully_connected/biases/Adam_1/Assign
b
init_38NoOpQ^Optimizer/over_options/q_func/action_value/fully_connected_1/weights/Adam/Assign
d
init_39NoOpS^Optimizer/over_options/q_func/action_value/fully_connected_1/weights/Adam_1/Assign
a
init_40NoOpP^Optimizer/over_options/q_func/action_value/fully_connected_1/biases/Adam/Assign
c
init_41NoOpR^Optimizer/over_options/q_func/action_value/fully_connected_1/biases/Adam_1/Assign"їV®й=Є     о«Ѕ	иЊлЂdЊ÷AJ∞р

™&И&
+
Abs
x"T
y"T"
Ttype:	
2	
9
Add
x"T
y"T
z"T"
Ttype:
2	
S
AddN
inputs"T*N
sum"T"
Nint(0"
Ttype:
2	АР
—
	ApplyAdam
var"TА	
m"TА	
v"TА
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"TА"
Ttype:
2	"
use_lockingbool( 
l
ArgMax

input"T
	dimension"Tidx

output	"
Ttype:
2	"
Tidxtype0:
2	
x
Assign
ref"TА

value"T

output_ref"TА"	
Ttype"
validate_shapebool("
use_lockingbool(Ш
{
BiasAdd

value"T	
bias"T
output"T"
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
{
BiasAddGrad
out_backprop"T
output"T"
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
…
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW
п
Conv2DBackpropFilter

input"T
filter_sizes
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW
о
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW
S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
4
Fill
dims

value"T
output"T"	
Ttype
>
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
7
FloorMod
x"T
y"T
z"T"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
7
Less
x"T
y"T
z
"
Ttype:
2		
o
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2
Й
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
:
Maximum
x"T
y"T
z"T"
Ttype:	
2	Р
:
Minimum
x"T
y"T
z"T"
Ttype:	
2	Р
<
Mul
x"T
y"T
z"T"
Ttype:
2	Р
-
Neg
x"T
y"T"
Ttype:
	2	

NoOp
М
OneHot
indices"TI	
depth
on_value"T
	off_value"T
output"T"
axisint€€€€€€€€€"	
Ttype"
TItype0	:
2	
A
Placeholder
output"dtype"
dtypetype"
shapeshape: 
5
Pow
x"T
y"T
z"T"
Ttype:
	2	
К
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
}
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
`
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:
2	
=
RealDiv
x"T
y"T
z"T"
Ttype:
2	
A
Relu
features"T
activations"T"
Ttype:
2		
S
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2		
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
l
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
-
Rsqrt
x"T
y"T"
Ttype:	
2
i
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
.
Sign
x"T
y"T"
Ttype:
	2	
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
0
Square
x"T
y"T"
Ttype:
	2	
2
StopGradient

input"T
output"T"	
Ttype
5
Sub
x"T
y"T
z"T"
Ttype:
	2	
Й
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtypeА"
shapeshape"
dtypetype"
	containerstring "
shared_namestring И
&
	ZerosLike
x"T
y"T"	
Ttype*1.1.02v1.1.0-rc0-61-g1ec6ed5ўн	
o
over_options/obs_t_phPlaceholder*
dtype0*
shape: */
_output_shapes
:€€€€€€€€€	
y
over_options/CastCastover_options/obs_t_ph*

SrcT0*

DstT0*/
_output_shapes
:€€€€€€€€€	
_
over_options/obs_t_float/yConst*
valueB
 *  C*
dtype0*
_output_shapes
: 
М
over_options/obs_t_floatRealDivover_options/Castover_options/obs_t_float/y*
T0*/
_output_shapes
:€€€€€€€€€	
я
Iover_options/q_func/convnet/Conv/weights/Initializer/random_uniform/shapeConst*%
valueB"             *
dtype0*;
_class1
/-loc:@over_options/q_func/convnet/Conv/weights*
_output_shapes
:
…
Gover_options/q_func/convnet/Conv/weights/Initializer/random_uniform/minConst*
valueB
 *чьSљ*
dtype0*;
_class1
/-loc:@over_options/q_func/convnet/Conv/weights*
_output_shapes
: 
…
Gover_options/q_func/convnet/Conv/weights/Initializer/random_uniform/maxConst*
valueB
 *чьS=*
dtype0*;
_class1
/-loc:@over_options/q_func/convnet/Conv/weights*
_output_shapes
: 
Ѕ
Qover_options/q_func/convnet/Conv/weights/Initializer/random_uniform/RandomUniformRandomUniformIover_options/q_func/convnet/Conv/weights/Initializer/random_uniform/shape*

seed *
seed2 *
dtype0*
T0*;
_class1
/-loc:@over_options/q_func/convnet/Conv/weights*&
_output_shapes
: 
Њ
Gover_options/q_func/convnet/Conv/weights/Initializer/random_uniform/subSubGover_options/q_func/convnet/Conv/weights/Initializer/random_uniform/maxGover_options/q_func/convnet/Conv/weights/Initializer/random_uniform/min*
T0*;
_class1
/-loc:@over_options/q_func/convnet/Conv/weights*
_output_shapes
: 
Ў
Gover_options/q_func/convnet/Conv/weights/Initializer/random_uniform/mulMulQover_options/q_func/convnet/Conv/weights/Initializer/random_uniform/RandomUniformGover_options/q_func/convnet/Conv/weights/Initializer/random_uniform/sub*
T0*;
_class1
/-loc:@over_options/q_func/convnet/Conv/weights*&
_output_shapes
: 
 
Cover_options/q_func/convnet/Conv/weights/Initializer/random_uniformAddGover_options/q_func/convnet/Conv/weights/Initializer/random_uniform/mulGover_options/q_func/convnet/Conv/weights/Initializer/random_uniform/min*
T0*;
_class1
/-loc:@over_options/q_func/convnet/Conv/weights*&
_output_shapes
: 
й
(over_options/q_func/convnet/Conv/weights
VariableV2*
shape: *
dtype0*
	container *
shared_name *;
_class1
/-loc:@over_options/q_func/convnet/Conv/weights*&
_output_shapes
: 
њ
/over_options/q_func/convnet/Conv/weights/AssignAssign(over_options/q_func/convnet/Conv/weightsCover_options/q_func/convnet/Conv/weights/Initializer/random_uniform*
T0*
validate_shape(*
use_locking(*;
_class1
/-loc:@over_options/q_func/convnet/Conv/weights*&
_output_shapes
: 
—
-over_options/q_func/convnet/Conv/weights/readIdentity(over_options/q_func/convnet/Conv/weights*
T0*;
_class1
/-loc:@over_options/q_func/convnet/Conv/weights*&
_output_shapes
: 
¬
9over_options/q_func/convnet/Conv/biases/Initializer/ConstConst*
valueB *    *
dtype0*:
_class0
.,loc:@over_options/q_func/convnet/Conv/biases*
_output_shapes
: 
ѕ
'over_options/q_func/convnet/Conv/biases
VariableV2*
shape: *
dtype0*
	container *
shared_name *:
_class0
.,loc:@over_options/q_func/convnet/Conv/biases*
_output_shapes
: 
¶
.over_options/q_func/convnet/Conv/biases/AssignAssign'over_options/q_func/convnet/Conv/biases9over_options/q_func/convnet/Conv/biases/Initializer/Const*
T0*
validate_shape(*
use_locking(*:
_class0
.,loc:@over_options/q_func/convnet/Conv/biases*
_output_shapes
: 
¬
,over_options/q_func/convnet/Conv/biases/readIdentity'over_options/q_func/convnet/Conv/biases*
T0*:
_class0
.,loc:@over_options/q_func/convnet/Conv/biases*
_output_shapes
: 
Л
2over_options/q_func/convnet/Conv/convolution/ShapeConst*%
valueB"             *
dtype0*
_output_shapes
:
Л
:over_options/q_func/convnet/Conv/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
Р
,over_options/q_func/convnet/Conv/convolutionConv2Dover_options/obs_t_float-over_options/q_func/convnet/Conv/weights/read*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*/
_output_shapes
:€€€€€€€€€ 
а
(over_options/q_func/convnet/Conv/BiasAddBiasAdd,over_options/q_func/convnet/Conv/convolution,over_options/q_func/convnet/Conv/biases/read*
T0*
data_formatNHWC*/
_output_shapes
:€€€€€€€€€ 
С
%over_options/q_func/convnet/Conv/ReluRelu(over_options/q_func/convnet/Conv/BiasAdd*
T0*/
_output_shapes
:€€€€€€€€€ 
г
Kover_options/q_func/convnet/Conv_1/weights/Initializer/random_uniform/shapeConst*%
valueB"          @   *
dtype0*=
_class3
1/loc:@over_options/q_func/convnet/Conv_1/weights*
_output_shapes
:
Ќ
Iover_options/q_func/convnet/Conv_1/weights/Initializer/random_uniform/minConst*
valueB
 *  Аљ*
dtype0*=
_class3
1/loc:@over_options/q_func/convnet/Conv_1/weights*
_output_shapes
: 
Ќ
Iover_options/q_func/convnet/Conv_1/weights/Initializer/random_uniform/maxConst*
valueB
 *  А=*
dtype0*=
_class3
1/loc:@over_options/q_func/convnet/Conv_1/weights*
_output_shapes
: 
«
Sover_options/q_func/convnet/Conv_1/weights/Initializer/random_uniform/RandomUniformRandomUniformKover_options/q_func/convnet/Conv_1/weights/Initializer/random_uniform/shape*

seed *
seed2 *
dtype0*
T0*=
_class3
1/loc:@over_options/q_func/convnet/Conv_1/weights*&
_output_shapes
: @
∆
Iover_options/q_func/convnet/Conv_1/weights/Initializer/random_uniform/subSubIover_options/q_func/convnet/Conv_1/weights/Initializer/random_uniform/maxIover_options/q_func/convnet/Conv_1/weights/Initializer/random_uniform/min*
T0*=
_class3
1/loc:@over_options/q_func/convnet/Conv_1/weights*
_output_shapes
: 
а
Iover_options/q_func/convnet/Conv_1/weights/Initializer/random_uniform/mulMulSover_options/q_func/convnet/Conv_1/weights/Initializer/random_uniform/RandomUniformIover_options/q_func/convnet/Conv_1/weights/Initializer/random_uniform/sub*
T0*=
_class3
1/loc:@over_options/q_func/convnet/Conv_1/weights*&
_output_shapes
: @
“
Eover_options/q_func/convnet/Conv_1/weights/Initializer/random_uniformAddIover_options/q_func/convnet/Conv_1/weights/Initializer/random_uniform/mulIover_options/q_func/convnet/Conv_1/weights/Initializer/random_uniform/min*
T0*=
_class3
1/loc:@over_options/q_func/convnet/Conv_1/weights*&
_output_shapes
: @
н
*over_options/q_func/convnet/Conv_1/weights
VariableV2*
shape: @*
dtype0*
	container *
shared_name *=
_class3
1/loc:@over_options/q_func/convnet/Conv_1/weights*&
_output_shapes
: @
«
1over_options/q_func/convnet/Conv_1/weights/AssignAssign*over_options/q_func/convnet/Conv_1/weightsEover_options/q_func/convnet/Conv_1/weights/Initializer/random_uniform*
T0*
validate_shape(*
use_locking(*=
_class3
1/loc:@over_options/q_func/convnet/Conv_1/weights*&
_output_shapes
: @
„
/over_options/q_func/convnet/Conv_1/weights/readIdentity*over_options/q_func/convnet/Conv_1/weights*
T0*=
_class3
1/loc:@over_options/q_func/convnet/Conv_1/weights*&
_output_shapes
: @
∆
;over_options/q_func/convnet/Conv_1/biases/Initializer/ConstConst*
valueB@*    *
dtype0*<
_class2
0.loc:@over_options/q_func/convnet/Conv_1/biases*
_output_shapes
:@
”
)over_options/q_func/convnet/Conv_1/biases
VariableV2*
shape:@*
dtype0*
	container *
shared_name *<
_class2
0.loc:@over_options/q_func/convnet/Conv_1/biases*
_output_shapes
:@
Ѓ
0over_options/q_func/convnet/Conv_1/biases/AssignAssign)over_options/q_func/convnet/Conv_1/biases;over_options/q_func/convnet/Conv_1/biases/Initializer/Const*
T0*
validate_shape(*
use_locking(*<
_class2
0.loc:@over_options/q_func/convnet/Conv_1/biases*
_output_shapes
:@
»
.over_options/q_func/convnet/Conv_1/biases/readIdentity)over_options/q_func/convnet/Conv_1/biases*
T0*<
_class2
0.loc:@over_options/q_func/convnet/Conv_1/biases*
_output_shapes
:@
Н
4over_options/q_func/convnet/Conv_1/convolution/ShapeConst*%
valueB"          @   *
dtype0*
_output_shapes
:
Н
<over_options/q_func/convnet/Conv_1/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
°
.over_options/q_func/convnet/Conv_1/convolutionConv2D%over_options/q_func/convnet/Conv/Relu/over_options/q_func/convnet/Conv_1/weights/read*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*/
_output_shapes
:€€€€€€€€€@
ж
*over_options/q_func/convnet/Conv_1/BiasAddBiasAdd.over_options/q_func/convnet/Conv_1/convolution.over_options/q_func/convnet/Conv_1/biases/read*
T0*
data_formatNHWC*/
_output_shapes
:€€€€€€€€€@
Х
'over_options/q_func/convnet/Conv_1/ReluRelu*over_options/q_func/convnet/Conv_1/BiasAdd*
T0*/
_output_shapes
:€€€€€€€€€@
г
Kover_options/q_func/convnet/Conv_2/weights/Initializer/random_uniform/shapeConst*%
valueB"      @   @   *
dtype0*=
_class3
1/loc:@over_options/q_func/convnet/Conv_2/weights*
_output_shapes
:
Ќ
Iover_options/q_func/convnet/Conv_2/weights/Initializer/random_uniform/minConst*
valueB
 *:ЌУљ*
dtype0*=
_class3
1/loc:@over_options/q_func/convnet/Conv_2/weights*
_output_shapes
: 
Ќ
Iover_options/q_func/convnet/Conv_2/weights/Initializer/random_uniform/maxConst*
valueB
 *:ЌУ=*
dtype0*=
_class3
1/loc:@over_options/q_func/convnet/Conv_2/weights*
_output_shapes
: 
«
Sover_options/q_func/convnet/Conv_2/weights/Initializer/random_uniform/RandomUniformRandomUniformKover_options/q_func/convnet/Conv_2/weights/Initializer/random_uniform/shape*

seed *
seed2 *
dtype0*
T0*=
_class3
1/loc:@over_options/q_func/convnet/Conv_2/weights*&
_output_shapes
:@@
∆
Iover_options/q_func/convnet/Conv_2/weights/Initializer/random_uniform/subSubIover_options/q_func/convnet/Conv_2/weights/Initializer/random_uniform/maxIover_options/q_func/convnet/Conv_2/weights/Initializer/random_uniform/min*
T0*=
_class3
1/loc:@over_options/q_func/convnet/Conv_2/weights*
_output_shapes
: 
а
Iover_options/q_func/convnet/Conv_2/weights/Initializer/random_uniform/mulMulSover_options/q_func/convnet/Conv_2/weights/Initializer/random_uniform/RandomUniformIover_options/q_func/convnet/Conv_2/weights/Initializer/random_uniform/sub*
T0*=
_class3
1/loc:@over_options/q_func/convnet/Conv_2/weights*&
_output_shapes
:@@
“
Eover_options/q_func/convnet/Conv_2/weights/Initializer/random_uniformAddIover_options/q_func/convnet/Conv_2/weights/Initializer/random_uniform/mulIover_options/q_func/convnet/Conv_2/weights/Initializer/random_uniform/min*
T0*=
_class3
1/loc:@over_options/q_func/convnet/Conv_2/weights*&
_output_shapes
:@@
н
*over_options/q_func/convnet/Conv_2/weights
VariableV2*
shape:@@*
dtype0*
	container *
shared_name *=
_class3
1/loc:@over_options/q_func/convnet/Conv_2/weights*&
_output_shapes
:@@
«
1over_options/q_func/convnet/Conv_2/weights/AssignAssign*over_options/q_func/convnet/Conv_2/weightsEover_options/q_func/convnet/Conv_2/weights/Initializer/random_uniform*
T0*
validate_shape(*
use_locking(*=
_class3
1/loc:@over_options/q_func/convnet/Conv_2/weights*&
_output_shapes
:@@
„
/over_options/q_func/convnet/Conv_2/weights/readIdentity*over_options/q_func/convnet/Conv_2/weights*
T0*=
_class3
1/loc:@over_options/q_func/convnet/Conv_2/weights*&
_output_shapes
:@@
∆
;over_options/q_func/convnet/Conv_2/biases/Initializer/ConstConst*
valueB@*    *
dtype0*<
_class2
0.loc:@over_options/q_func/convnet/Conv_2/biases*
_output_shapes
:@
”
)over_options/q_func/convnet/Conv_2/biases
VariableV2*
shape:@*
dtype0*
	container *
shared_name *<
_class2
0.loc:@over_options/q_func/convnet/Conv_2/biases*
_output_shapes
:@
Ѓ
0over_options/q_func/convnet/Conv_2/biases/AssignAssign)over_options/q_func/convnet/Conv_2/biases;over_options/q_func/convnet/Conv_2/biases/Initializer/Const*
T0*
validate_shape(*
use_locking(*<
_class2
0.loc:@over_options/q_func/convnet/Conv_2/biases*
_output_shapes
:@
»
.over_options/q_func/convnet/Conv_2/biases/readIdentity)over_options/q_func/convnet/Conv_2/biases*
T0*<
_class2
0.loc:@over_options/q_func/convnet/Conv_2/biases*
_output_shapes
:@
Н
4over_options/q_func/convnet/Conv_2/convolution/ShapeConst*%
valueB"      @   @   *
dtype0*
_output_shapes
:
Н
<over_options/q_func/convnet/Conv_2/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
£
.over_options/q_func/convnet/Conv_2/convolutionConv2D'over_options/q_func/convnet/Conv_1/Relu/over_options/q_func/convnet/Conv_2/weights/read*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*/
_output_shapes
:€€€€€€€€€@
ж
*over_options/q_func/convnet/Conv_2/BiasAddBiasAdd.over_options/q_func/convnet/Conv_2/convolution.over_options/q_func/convnet/Conv_2/biases/read*
T0*
data_formatNHWC*/
_output_shapes
:€€€€€€€€€@
Х
'over_options/q_func/convnet/Conv_2/ReluRelu*over_options/q_func/convnet/Conv_2/BiasAdd*
T0*/
_output_shapes
:€€€€€€€€€@
И
!over_options/q_func/Flatten/ShapeShape'over_options/q_func/convnet/Conv_2/Relu*
T0*
out_type0*
_output_shapes
:
q
'over_options/q_func/Flatten/Slice/beginConst*
valueB: *
dtype0*
_output_shapes
:
p
&over_options/q_func/Flatten/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
–
!over_options/q_func/Flatten/SliceSlice!over_options/q_func/Flatten/Shape'over_options/q_func/Flatten/Slice/begin&over_options/q_func/Flatten/Slice/size*
T0*
Index0*
_output_shapes
:
s
)over_options/q_func/Flatten/Slice_1/beginConst*
valueB:*
dtype0*
_output_shapes
:
r
(over_options/q_func/Flatten/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
÷
#over_options/q_func/Flatten/Slice_1Slice!over_options/q_func/Flatten/Shape)over_options/q_func/Flatten/Slice_1/begin(over_options/q_func/Flatten/Slice_1/size*
T0*
Index0*
_output_shapes
:
k
!over_options/q_func/Flatten/ConstConst*
valueB: *
dtype0*
_output_shapes
:
Ѓ
 over_options/q_func/Flatten/ProdProd#over_options/q_func/Flatten/Slice_1!over_options/q_func/Flatten/Const*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
l
*over_options/q_func/Flatten/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
≥
&over_options/q_func/Flatten/ExpandDims
ExpandDims over_options/q_func/Flatten/Prod*over_options/q_func/Flatten/ExpandDims/dim*
T0*

Tdim0*
_output_shapes
:
i
'over_options/q_func/Flatten/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
№
"over_options/q_func/Flatten/concatConcatV2!over_options/q_func/Flatten/Slice&over_options/q_func/Flatten/ExpandDims'over_options/q_func/Flatten/concat/axis*
N*
T0*

Tidx0*
_output_shapes
:
Љ
#over_options/q_func/Flatten/ReshapeReshape'over_options/q_func/convnet/Conv_2/Relu"over_options/q_func/Flatten/concat*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
ч
Yover_options/q_func/action_value/fully_connected/weights/Initializer/random_uniform/shapeConst*
valueB"      *
dtype0*K
_classA
?=loc:@over_options/q_func/action_value/fully_connected/weights*
_output_shapes
:
й
Wover_options/q_func/action_value/fully_connected/weights/Initializer/random_uniform/minConst*
valueB
 *„≥Ёљ*
dtype0*K
_classA
?=loc:@over_options/q_func/action_value/fully_connected/weights*
_output_shapes
: 
й
Wover_options/q_func/action_value/fully_connected/weights/Initializer/random_uniform/maxConst*
valueB
 *„≥Ё=*
dtype0*K
_classA
?=loc:@over_options/q_func/action_value/fully_connected/weights*
_output_shapes
: 
л
aover_options/q_func/action_value/fully_connected/weights/Initializer/random_uniform/RandomUniformRandomUniformYover_options/q_func/action_value/fully_connected/weights/Initializer/random_uniform/shape*

seed *
seed2 *
dtype0*
T0*K
_classA
?=loc:@over_options/q_func/action_value/fully_connected/weights* 
_output_shapes
:
АА
ю
Wover_options/q_func/action_value/fully_connected/weights/Initializer/random_uniform/subSubWover_options/q_func/action_value/fully_connected/weights/Initializer/random_uniform/maxWover_options/q_func/action_value/fully_connected/weights/Initializer/random_uniform/min*
T0*K
_classA
?=loc:@over_options/q_func/action_value/fully_connected/weights*
_output_shapes
: 
Т
Wover_options/q_func/action_value/fully_connected/weights/Initializer/random_uniform/mulMulaover_options/q_func/action_value/fully_connected/weights/Initializer/random_uniform/RandomUniformWover_options/q_func/action_value/fully_connected/weights/Initializer/random_uniform/sub*
T0*K
_classA
?=loc:@over_options/q_func/action_value/fully_connected/weights* 
_output_shapes
:
АА
Д
Sover_options/q_func/action_value/fully_connected/weights/Initializer/random_uniformAddWover_options/q_func/action_value/fully_connected/weights/Initializer/random_uniform/mulWover_options/q_func/action_value/fully_connected/weights/Initializer/random_uniform/min*
T0*K
_classA
?=loc:@over_options/q_func/action_value/fully_connected/weights* 
_output_shapes
:
АА
э
8over_options/q_func/action_value/fully_connected/weights
VariableV2*
shape:
АА*
dtype0*
	container *
shared_name *K
_classA
?=loc:@over_options/q_func/action_value/fully_connected/weights* 
_output_shapes
:
АА
щ
?over_options/q_func/action_value/fully_connected/weights/AssignAssign8over_options/q_func/action_value/fully_connected/weightsSover_options/q_func/action_value/fully_connected/weights/Initializer/random_uniform*
T0*
validate_shape(*
use_locking(*K
_classA
?=loc:@over_options/q_func/action_value/fully_connected/weights* 
_output_shapes
:
АА
ы
=over_options/q_func/action_value/fully_connected/weights/readIdentity8over_options/q_func/action_value/fully_connected/weights*
T0*K
_classA
?=loc:@over_options/q_func/action_value/fully_connected/weights* 
_output_shapes
:
АА
д
Iover_options/q_func/action_value/fully_connected/biases/Initializer/ConstConst*
valueBА*    *
dtype0*J
_class@
><loc:@over_options/q_func/action_value/fully_connected/biases*
_output_shapes	
:А
с
7over_options/q_func/action_value/fully_connected/biases
VariableV2*
shape:А*
dtype0*
	container *
shared_name *J
_class@
><loc:@over_options/q_func/action_value/fully_connected/biases*
_output_shapes	
:А
з
>over_options/q_func/action_value/fully_connected/biases/AssignAssign7over_options/q_func/action_value/fully_connected/biasesIover_options/q_func/action_value/fully_connected/biases/Initializer/Const*
T0*
validate_shape(*
use_locking(*J
_class@
><loc:@over_options/q_func/action_value/fully_connected/biases*
_output_shapes	
:А
у
<over_options/q_func/action_value/fully_connected/biases/readIdentity7over_options/q_func/action_value/fully_connected/biases*
T0*J
_class@
><loc:@over_options/q_func/action_value/fully_connected/biases*
_output_shapes	
:А
ю
7over_options/q_func/action_value/fully_connected/MatMulMatMul#over_options/q_func/Flatten/Reshape=over_options/q_func/action_value/fully_connected/weights/read*
transpose_a( *
transpose_b( *
T0*(
_output_shapes
:€€€€€€€€€А
Д
8over_options/q_func/action_value/fully_connected/BiasAddBiasAdd7over_options/q_func/action_value/fully_connected/MatMul<over_options/q_func/action_value/fully_connected/biases/read*
T0*
data_formatNHWC*(
_output_shapes
:€€€€€€€€€А
™
5over_options/q_func/action_value/fully_connected/ReluRelu8over_options/q_func/action_value/fully_connected/BiasAdd*
T0*(
_output_shapes
:€€€€€€€€€А
ы
[over_options/q_func/action_value/fully_connected_1/weights/Initializer/random_uniform/shapeConst*
valueB"      *
dtype0*M
_classC
A?loc:@over_options/q_func/action_value/fully_connected_1/weights*
_output_shapes
:
н
Yover_options/q_func/action_value/fully_connected_1/weights/Initializer/random_uniform/minConst*
valueB
 *≤_Њ*
dtype0*M
_classC
A?loc:@over_options/q_func/action_value/fully_connected_1/weights*
_output_shapes
: 
н
Yover_options/q_func/action_value/fully_connected_1/weights/Initializer/random_uniform/maxConst*
valueB
 *≤_>*
dtype0*M
_classC
A?loc:@over_options/q_func/action_value/fully_connected_1/weights*
_output_shapes
: 
р
cover_options/q_func/action_value/fully_connected_1/weights/Initializer/random_uniform/RandomUniformRandomUniform[over_options/q_func/action_value/fully_connected_1/weights/Initializer/random_uniform/shape*

seed *
seed2 *
dtype0*
T0*M
_classC
A?loc:@over_options/q_func/action_value/fully_connected_1/weights*
_output_shapes
:	А
Ж
Yover_options/q_func/action_value/fully_connected_1/weights/Initializer/random_uniform/subSubYover_options/q_func/action_value/fully_connected_1/weights/Initializer/random_uniform/maxYover_options/q_func/action_value/fully_connected_1/weights/Initializer/random_uniform/min*
T0*M
_classC
A?loc:@over_options/q_func/action_value/fully_connected_1/weights*
_output_shapes
: 
Щ
Yover_options/q_func/action_value/fully_connected_1/weights/Initializer/random_uniform/mulMulcover_options/q_func/action_value/fully_connected_1/weights/Initializer/random_uniform/RandomUniformYover_options/q_func/action_value/fully_connected_1/weights/Initializer/random_uniform/sub*
T0*M
_classC
A?loc:@over_options/q_func/action_value/fully_connected_1/weights*
_output_shapes
:	А
Л
Uover_options/q_func/action_value/fully_connected_1/weights/Initializer/random_uniformAddYover_options/q_func/action_value/fully_connected_1/weights/Initializer/random_uniform/mulYover_options/q_func/action_value/fully_connected_1/weights/Initializer/random_uniform/min*
T0*M
_classC
A?loc:@over_options/q_func/action_value/fully_connected_1/weights*
_output_shapes
:	А
€
:over_options/q_func/action_value/fully_connected_1/weights
VariableV2*
shape:	А*
dtype0*
	container *
shared_name *M
_classC
A?loc:@over_options/q_func/action_value/fully_connected_1/weights*
_output_shapes
:	А
А
Aover_options/q_func/action_value/fully_connected_1/weights/AssignAssign:over_options/q_func/action_value/fully_connected_1/weightsUover_options/q_func/action_value/fully_connected_1/weights/Initializer/random_uniform*
T0*
validate_shape(*
use_locking(*M
_classC
A?loc:@over_options/q_func/action_value/fully_connected_1/weights*
_output_shapes
:	А
А
?over_options/q_func/action_value/fully_connected_1/weights/readIdentity:over_options/q_func/action_value/fully_connected_1/weights*
T0*M
_classC
A?loc:@over_options/q_func/action_value/fully_connected_1/weights*
_output_shapes
:	А
ж
Kover_options/q_func/action_value/fully_connected_1/biases/Initializer/ConstConst*
valueB*    *
dtype0*L
_classB
@>loc:@over_options/q_func/action_value/fully_connected_1/biases*
_output_shapes
:
у
9over_options/q_func/action_value/fully_connected_1/biases
VariableV2*
shape:*
dtype0*
	container *
shared_name *L
_classB
@>loc:@over_options/q_func/action_value/fully_connected_1/biases*
_output_shapes
:
о
@over_options/q_func/action_value/fully_connected_1/biases/AssignAssign9over_options/q_func/action_value/fully_connected_1/biasesKover_options/q_func/action_value/fully_connected_1/biases/Initializer/Const*
T0*
validate_shape(*
use_locking(*L
_classB
@>loc:@over_options/q_func/action_value/fully_connected_1/biases*
_output_shapes
:
ш
>over_options/q_func/action_value/fully_connected_1/biases/readIdentity9over_options/q_func/action_value/fully_connected_1/biases*
T0*L
_classB
@>loc:@over_options/q_func/action_value/fully_connected_1/biases*
_output_shapes
:
У
9over_options/q_func/action_value/fully_connected_1/MatMulMatMul5over_options/q_func/action_value/fully_connected/Relu?over_options/q_func/action_value/fully_connected_1/weights/read*
transpose_a( *
transpose_b( *
T0*'
_output_shapes
:€€€€€€€€€
Й
:over_options/q_func/action_value/fully_connected_1/BiasAddBiasAdd9over_options/q_func/action_value/fully_connected_1/MatMul>over_options/q_func/action_value/fully_connected_1/biases/read*
T0*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€
`
over_options/pred_ac/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
і
over_options/pred_acArgMax:over_options/q_func/action_value/fully_connected_1/BiasAddover_options/pred_ac/dimension*
T0*

Tidx0*#
_output_shapes
:€€€€€€€€€
V
act_t_phPlaceholder*
dtype0*
shape: *#
_output_shapes
:€€€€€€€€€
V
rew_t_phPlaceholder*
dtype0*
shape: *#
_output_shapes
:€€€€€€€€€
o
obs_tp1_ph/obs_tp1_phPlaceholder*
dtype0*
shape: */
_output_shapes
:€€€€€€€€€	
w
obs_tp1_ph/CastCastobs_tp1_ph/obs_tp1_ph*

SrcT0*

DstT0*/
_output_shapes
:€€€€€€€€€	
Y
obs_tp1_ph/truediv/yConst*
valueB
 *  C*
dtype0*
_output_shapes
: 
~
obs_tp1_ph/truedivRealDivobs_tp1_ph/Castobs_tp1_ph/truediv/y*
T0*/
_output_shapes
:€€€€€€€€€	
Z
done_mask_phPlaceholder*
dtype0*
shape: *#
_output_shapes
:€€€€€€€€€
W
	opt_stepsPlaceholder*
dtype0*
shape: *#
_output_shapes
:€€€€€€€€€
^
pred_q_a/one_hot/on_valueConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
_
pred_q_a/one_hot/off_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
X
pred_q_a/one_hot/depthConst*
value	B :*
dtype0*
_output_shapes
: 
ƒ
pred_q_a/one_hotOneHotact_t_phpred_q_a/one_hot/depthpred_q_a/one_hot/on_valuepred_q_a/one_hot/off_value*
axis€€€€€€€€€*
T0*
TI0*'
_output_shapes
:€€€€€€€€€
У
pred_q_a/mulMul:over_options/q_func/action_value/fully_connected_1/BiasAddpred_q_a/one_hot*
T0*'
_output_shapes
:€€€€€€€€€
e
#pred_q_a/pred_q_a/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
Ц
pred_q_a/pred_q_aSumpred_q_a/mul#pred_q_a/pred_q_a/reduction_indices*
	keep_dims( *
T0*

Tidx0*#
_output_shapes
:€€€€€€€€€
”
Ctarget_q_func/convnet/Conv/weights/Initializer/random_uniform/shapeConst*%
valueB"             *
dtype0*5
_class+
)'loc:@target_q_func/convnet/Conv/weights*
_output_shapes
:
љ
Atarget_q_func/convnet/Conv/weights/Initializer/random_uniform/minConst*
valueB
 *чьSљ*
dtype0*5
_class+
)'loc:@target_q_func/convnet/Conv/weights*
_output_shapes
: 
љ
Atarget_q_func/convnet/Conv/weights/Initializer/random_uniform/maxConst*
valueB
 *чьS=*
dtype0*5
_class+
)'loc:@target_q_func/convnet/Conv/weights*
_output_shapes
: 
ѓ
Ktarget_q_func/convnet/Conv/weights/Initializer/random_uniform/RandomUniformRandomUniformCtarget_q_func/convnet/Conv/weights/Initializer/random_uniform/shape*

seed *
seed2 *
dtype0*
T0*5
_class+
)'loc:@target_q_func/convnet/Conv/weights*&
_output_shapes
: 
¶
Atarget_q_func/convnet/Conv/weights/Initializer/random_uniform/subSubAtarget_q_func/convnet/Conv/weights/Initializer/random_uniform/maxAtarget_q_func/convnet/Conv/weights/Initializer/random_uniform/min*
T0*5
_class+
)'loc:@target_q_func/convnet/Conv/weights*
_output_shapes
: 
ј
Atarget_q_func/convnet/Conv/weights/Initializer/random_uniform/mulMulKtarget_q_func/convnet/Conv/weights/Initializer/random_uniform/RandomUniformAtarget_q_func/convnet/Conv/weights/Initializer/random_uniform/sub*
T0*5
_class+
)'loc:@target_q_func/convnet/Conv/weights*&
_output_shapes
: 
≤
=target_q_func/convnet/Conv/weights/Initializer/random_uniformAddAtarget_q_func/convnet/Conv/weights/Initializer/random_uniform/mulAtarget_q_func/convnet/Conv/weights/Initializer/random_uniform/min*
T0*5
_class+
)'loc:@target_q_func/convnet/Conv/weights*&
_output_shapes
: 
Ё
"target_q_func/convnet/Conv/weights
VariableV2*
shape: *
dtype0*
	container *
shared_name *5
_class+
)'loc:@target_q_func/convnet/Conv/weights*&
_output_shapes
: 
І
)target_q_func/convnet/Conv/weights/AssignAssign"target_q_func/convnet/Conv/weights=target_q_func/convnet/Conv/weights/Initializer/random_uniform*
T0*
validate_shape(*
use_locking(*5
_class+
)'loc:@target_q_func/convnet/Conv/weights*&
_output_shapes
: 
њ
'target_q_func/convnet/Conv/weights/readIdentity"target_q_func/convnet/Conv/weights*
T0*5
_class+
)'loc:@target_q_func/convnet/Conv/weights*&
_output_shapes
: 
ґ
3target_q_func/convnet/Conv/biases/Initializer/ConstConst*
valueB *    *
dtype0*4
_class*
(&loc:@target_q_func/convnet/Conv/biases*
_output_shapes
: 
√
!target_q_func/convnet/Conv/biases
VariableV2*
shape: *
dtype0*
	container *
shared_name *4
_class*
(&loc:@target_q_func/convnet/Conv/biases*
_output_shapes
: 
О
(target_q_func/convnet/Conv/biases/AssignAssign!target_q_func/convnet/Conv/biases3target_q_func/convnet/Conv/biases/Initializer/Const*
T0*
validate_shape(*
use_locking(*4
_class*
(&loc:@target_q_func/convnet/Conv/biases*
_output_shapes
: 
∞
&target_q_func/convnet/Conv/biases/readIdentity!target_q_func/convnet/Conv/biases*
T0*4
_class*
(&loc:@target_q_func/convnet/Conv/biases*
_output_shapes
: 
Е
,target_q_func/convnet/Conv/convolution/ShapeConst*%
valueB"             *
dtype0*
_output_shapes
:
Е
4target_q_func/convnet/Conv/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
ю
&target_q_func/convnet/Conv/convolutionConv2Dobs_tp1_ph/truediv'target_q_func/convnet/Conv/weights/read*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*/
_output_shapes
:€€€€€€€€€ 
ќ
"target_q_func/convnet/Conv/BiasAddBiasAdd&target_q_func/convnet/Conv/convolution&target_q_func/convnet/Conv/biases/read*
T0*
data_formatNHWC*/
_output_shapes
:€€€€€€€€€ 
Е
target_q_func/convnet/Conv/ReluRelu"target_q_func/convnet/Conv/BiasAdd*
T0*/
_output_shapes
:€€€€€€€€€ 
„
Etarget_q_func/convnet/Conv_1/weights/Initializer/random_uniform/shapeConst*%
valueB"          @   *
dtype0*7
_class-
+)loc:@target_q_func/convnet/Conv_1/weights*
_output_shapes
:
Ѕ
Ctarget_q_func/convnet/Conv_1/weights/Initializer/random_uniform/minConst*
valueB
 *  Аљ*
dtype0*7
_class-
+)loc:@target_q_func/convnet/Conv_1/weights*
_output_shapes
: 
Ѕ
Ctarget_q_func/convnet/Conv_1/weights/Initializer/random_uniform/maxConst*
valueB
 *  А=*
dtype0*7
_class-
+)loc:@target_q_func/convnet/Conv_1/weights*
_output_shapes
: 
µ
Mtarget_q_func/convnet/Conv_1/weights/Initializer/random_uniform/RandomUniformRandomUniformEtarget_q_func/convnet/Conv_1/weights/Initializer/random_uniform/shape*

seed *
seed2 *
dtype0*
T0*7
_class-
+)loc:@target_q_func/convnet/Conv_1/weights*&
_output_shapes
: @
Ѓ
Ctarget_q_func/convnet/Conv_1/weights/Initializer/random_uniform/subSubCtarget_q_func/convnet/Conv_1/weights/Initializer/random_uniform/maxCtarget_q_func/convnet/Conv_1/weights/Initializer/random_uniform/min*
T0*7
_class-
+)loc:@target_q_func/convnet/Conv_1/weights*
_output_shapes
: 
»
Ctarget_q_func/convnet/Conv_1/weights/Initializer/random_uniform/mulMulMtarget_q_func/convnet/Conv_1/weights/Initializer/random_uniform/RandomUniformCtarget_q_func/convnet/Conv_1/weights/Initializer/random_uniform/sub*
T0*7
_class-
+)loc:@target_q_func/convnet/Conv_1/weights*&
_output_shapes
: @
Ї
?target_q_func/convnet/Conv_1/weights/Initializer/random_uniformAddCtarget_q_func/convnet/Conv_1/weights/Initializer/random_uniform/mulCtarget_q_func/convnet/Conv_1/weights/Initializer/random_uniform/min*
T0*7
_class-
+)loc:@target_q_func/convnet/Conv_1/weights*&
_output_shapes
: @
б
$target_q_func/convnet/Conv_1/weights
VariableV2*
shape: @*
dtype0*
	container *
shared_name *7
_class-
+)loc:@target_q_func/convnet/Conv_1/weights*&
_output_shapes
: @
ѓ
+target_q_func/convnet/Conv_1/weights/AssignAssign$target_q_func/convnet/Conv_1/weights?target_q_func/convnet/Conv_1/weights/Initializer/random_uniform*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@target_q_func/convnet/Conv_1/weights*&
_output_shapes
: @
≈
)target_q_func/convnet/Conv_1/weights/readIdentity$target_q_func/convnet/Conv_1/weights*
T0*7
_class-
+)loc:@target_q_func/convnet/Conv_1/weights*&
_output_shapes
: @
Ї
5target_q_func/convnet/Conv_1/biases/Initializer/ConstConst*
valueB@*    *
dtype0*6
_class,
*(loc:@target_q_func/convnet/Conv_1/biases*
_output_shapes
:@
«
#target_q_func/convnet/Conv_1/biases
VariableV2*
shape:@*
dtype0*
	container *
shared_name *6
_class,
*(loc:@target_q_func/convnet/Conv_1/biases*
_output_shapes
:@
Ц
*target_q_func/convnet/Conv_1/biases/AssignAssign#target_q_func/convnet/Conv_1/biases5target_q_func/convnet/Conv_1/biases/Initializer/Const*
T0*
validate_shape(*
use_locking(*6
_class,
*(loc:@target_q_func/convnet/Conv_1/biases*
_output_shapes
:@
ґ
(target_q_func/convnet/Conv_1/biases/readIdentity#target_q_func/convnet/Conv_1/biases*
T0*6
_class,
*(loc:@target_q_func/convnet/Conv_1/biases*
_output_shapes
:@
З
.target_q_func/convnet/Conv_1/convolution/ShapeConst*%
valueB"          @   *
dtype0*
_output_shapes
:
З
6target_q_func/convnet/Conv_1/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
П
(target_q_func/convnet/Conv_1/convolutionConv2Dtarget_q_func/convnet/Conv/Relu)target_q_func/convnet/Conv_1/weights/read*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*/
_output_shapes
:€€€€€€€€€@
‘
$target_q_func/convnet/Conv_1/BiasAddBiasAdd(target_q_func/convnet/Conv_1/convolution(target_q_func/convnet/Conv_1/biases/read*
T0*
data_formatNHWC*/
_output_shapes
:€€€€€€€€€@
Й
!target_q_func/convnet/Conv_1/ReluRelu$target_q_func/convnet/Conv_1/BiasAdd*
T0*/
_output_shapes
:€€€€€€€€€@
„
Etarget_q_func/convnet/Conv_2/weights/Initializer/random_uniform/shapeConst*%
valueB"      @   @   *
dtype0*7
_class-
+)loc:@target_q_func/convnet/Conv_2/weights*
_output_shapes
:
Ѕ
Ctarget_q_func/convnet/Conv_2/weights/Initializer/random_uniform/minConst*
valueB
 *:ЌУљ*
dtype0*7
_class-
+)loc:@target_q_func/convnet/Conv_2/weights*
_output_shapes
: 
Ѕ
Ctarget_q_func/convnet/Conv_2/weights/Initializer/random_uniform/maxConst*
valueB
 *:ЌУ=*
dtype0*7
_class-
+)loc:@target_q_func/convnet/Conv_2/weights*
_output_shapes
: 
µ
Mtarget_q_func/convnet/Conv_2/weights/Initializer/random_uniform/RandomUniformRandomUniformEtarget_q_func/convnet/Conv_2/weights/Initializer/random_uniform/shape*

seed *
seed2 *
dtype0*
T0*7
_class-
+)loc:@target_q_func/convnet/Conv_2/weights*&
_output_shapes
:@@
Ѓ
Ctarget_q_func/convnet/Conv_2/weights/Initializer/random_uniform/subSubCtarget_q_func/convnet/Conv_2/weights/Initializer/random_uniform/maxCtarget_q_func/convnet/Conv_2/weights/Initializer/random_uniform/min*
T0*7
_class-
+)loc:@target_q_func/convnet/Conv_2/weights*
_output_shapes
: 
»
Ctarget_q_func/convnet/Conv_2/weights/Initializer/random_uniform/mulMulMtarget_q_func/convnet/Conv_2/weights/Initializer/random_uniform/RandomUniformCtarget_q_func/convnet/Conv_2/weights/Initializer/random_uniform/sub*
T0*7
_class-
+)loc:@target_q_func/convnet/Conv_2/weights*&
_output_shapes
:@@
Ї
?target_q_func/convnet/Conv_2/weights/Initializer/random_uniformAddCtarget_q_func/convnet/Conv_2/weights/Initializer/random_uniform/mulCtarget_q_func/convnet/Conv_2/weights/Initializer/random_uniform/min*
T0*7
_class-
+)loc:@target_q_func/convnet/Conv_2/weights*&
_output_shapes
:@@
б
$target_q_func/convnet/Conv_2/weights
VariableV2*
shape:@@*
dtype0*
	container *
shared_name *7
_class-
+)loc:@target_q_func/convnet/Conv_2/weights*&
_output_shapes
:@@
ѓ
+target_q_func/convnet/Conv_2/weights/AssignAssign$target_q_func/convnet/Conv_2/weights?target_q_func/convnet/Conv_2/weights/Initializer/random_uniform*
T0*
validate_shape(*
use_locking(*7
_class-
+)loc:@target_q_func/convnet/Conv_2/weights*&
_output_shapes
:@@
≈
)target_q_func/convnet/Conv_2/weights/readIdentity$target_q_func/convnet/Conv_2/weights*
T0*7
_class-
+)loc:@target_q_func/convnet/Conv_2/weights*&
_output_shapes
:@@
Ї
5target_q_func/convnet/Conv_2/biases/Initializer/ConstConst*
valueB@*    *
dtype0*6
_class,
*(loc:@target_q_func/convnet/Conv_2/biases*
_output_shapes
:@
«
#target_q_func/convnet/Conv_2/biases
VariableV2*
shape:@*
dtype0*
	container *
shared_name *6
_class,
*(loc:@target_q_func/convnet/Conv_2/biases*
_output_shapes
:@
Ц
*target_q_func/convnet/Conv_2/biases/AssignAssign#target_q_func/convnet/Conv_2/biases5target_q_func/convnet/Conv_2/biases/Initializer/Const*
T0*
validate_shape(*
use_locking(*6
_class,
*(loc:@target_q_func/convnet/Conv_2/biases*
_output_shapes
:@
ґ
(target_q_func/convnet/Conv_2/biases/readIdentity#target_q_func/convnet/Conv_2/biases*
T0*6
_class,
*(loc:@target_q_func/convnet/Conv_2/biases*
_output_shapes
:@
З
.target_q_func/convnet/Conv_2/convolution/ShapeConst*%
valueB"      @   @   *
dtype0*
_output_shapes
:
З
6target_q_func/convnet/Conv_2/convolution/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
С
(target_q_func/convnet/Conv_2/convolutionConv2D!target_q_func/convnet/Conv_1/Relu)target_q_func/convnet/Conv_2/weights/read*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*/
_output_shapes
:€€€€€€€€€@
‘
$target_q_func/convnet/Conv_2/BiasAddBiasAdd(target_q_func/convnet/Conv_2/convolution(target_q_func/convnet/Conv_2/biases/read*
T0*
data_formatNHWC*/
_output_shapes
:€€€€€€€€€@
Й
!target_q_func/convnet/Conv_2/ReluRelu$target_q_func/convnet/Conv_2/BiasAdd*
T0*/
_output_shapes
:€€€€€€€€€@
|
target_q_func/Flatten/ShapeShape!target_q_func/convnet/Conv_2/Relu*
T0*
out_type0*
_output_shapes
:
k
!target_q_func/Flatten/Slice/beginConst*
valueB: *
dtype0*
_output_shapes
:
j
 target_q_func/Flatten/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
Є
target_q_func/Flatten/SliceSlicetarget_q_func/Flatten/Shape!target_q_func/Flatten/Slice/begin target_q_func/Flatten/Slice/size*
T0*
Index0*
_output_shapes
:
m
#target_q_func/Flatten/Slice_1/beginConst*
valueB:*
dtype0*
_output_shapes
:
l
"target_q_func/Flatten/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
Њ
target_q_func/Flatten/Slice_1Slicetarget_q_func/Flatten/Shape#target_q_func/Flatten/Slice_1/begin"target_q_func/Flatten/Slice_1/size*
T0*
Index0*
_output_shapes
:
e
target_q_func/Flatten/ConstConst*
valueB: *
dtype0*
_output_shapes
:
Ь
target_q_func/Flatten/ProdProdtarget_q_func/Flatten/Slice_1target_q_func/Flatten/Const*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
f
$target_q_func/Flatten/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
°
 target_q_func/Flatten/ExpandDims
ExpandDimstarget_q_func/Flatten/Prod$target_q_func/Flatten/ExpandDims/dim*
T0*

Tdim0*
_output_shapes
:
c
!target_q_func/Flatten/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ƒ
target_q_func/Flatten/concatConcatV2target_q_func/Flatten/Slice target_q_func/Flatten/ExpandDims!target_q_func/Flatten/concat/axis*
N*
T0*

Tidx0*
_output_shapes
:
™
target_q_func/Flatten/ReshapeReshape!target_q_func/convnet/Conv_2/Relutarget_q_func/Flatten/concat*
T0*
Tshape0*(
_output_shapes
:€€€€€€€€€А
л
Starget_q_func/action_value/fully_connected/weights/Initializer/random_uniform/shapeConst*
valueB"      *
dtype0*E
_class;
97loc:@target_q_func/action_value/fully_connected/weights*
_output_shapes
:
Ё
Qtarget_q_func/action_value/fully_connected/weights/Initializer/random_uniform/minConst*
valueB
 *„≥Ёљ*
dtype0*E
_class;
97loc:@target_q_func/action_value/fully_connected/weights*
_output_shapes
: 
Ё
Qtarget_q_func/action_value/fully_connected/weights/Initializer/random_uniform/maxConst*
valueB
 *„≥Ё=*
dtype0*E
_class;
97loc:@target_q_func/action_value/fully_connected/weights*
_output_shapes
: 
ў
[target_q_func/action_value/fully_connected/weights/Initializer/random_uniform/RandomUniformRandomUniformStarget_q_func/action_value/fully_connected/weights/Initializer/random_uniform/shape*

seed *
seed2 *
dtype0*
T0*E
_class;
97loc:@target_q_func/action_value/fully_connected/weights* 
_output_shapes
:
АА
ж
Qtarget_q_func/action_value/fully_connected/weights/Initializer/random_uniform/subSubQtarget_q_func/action_value/fully_connected/weights/Initializer/random_uniform/maxQtarget_q_func/action_value/fully_connected/weights/Initializer/random_uniform/min*
T0*E
_class;
97loc:@target_q_func/action_value/fully_connected/weights*
_output_shapes
: 
ъ
Qtarget_q_func/action_value/fully_connected/weights/Initializer/random_uniform/mulMul[target_q_func/action_value/fully_connected/weights/Initializer/random_uniform/RandomUniformQtarget_q_func/action_value/fully_connected/weights/Initializer/random_uniform/sub*
T0*E
_class;
97loc:@target_q_func/action_value/fully_connected/weights* 
_output_shapes
:
АА
м
Mtarget_q_func/action_value/fully_connected/weights/Initializer/random_uniformAddQtarget_q_func/action_value/fully_connected/weights/Initializer/random_uniform/mulQtarget_q_func/action_value/fully_connected/weights/Initializer/random_uniform/min*
T0*E
_class;
97loc:@target_q_func/action_value/fully_connected/weights* 
_output_shapes
:
АА
с
2target_q_func/action_value/fully_connected/weights
VariableV2*
shape:
АА*
dtype0*
	container *
shared_name *E
_class;
97loc:@target_q_func/action_value/fully_connected/weights* 
_output_shapes
:
АА
б
9target_q_func/action_value/fully_connected/weights/AssignAssign2target_q_func/action_value/fully_connected/weightsMtarget_q_func/action_value/fully_connected/weights/Initializer/random_uniform*
T0*
validate_shape(*
use_locking(*E
_class;
97loc:@target_q_func/action_value/fully_connected/weights* 
_output_shapes
:
АА
й
7target_q_func/action_value/fully_connected/weights/readIdentity2target_q_func/action_value/fully_connected/weights*
T0*E
_class;
97loc:@target_q_func/action_value/fully_connected/weights* 
_output_shapes
:
АА
Ў
Ctarget_q_func/action_value/fully_connected/biases/Initializer/ConstConst*
valueBА*    *
dtype0*D
_class:
86loc:@target_q_func/action_value/fully_connected/biases*
_output_shapes	
:А
е
1target_q_func/action_value/fully_connected/biases
VariableV2*
shape:А*
dtype0*
	container *
shared_name *D
_class:
86loc:@target_q_func/action_value/fully_connected/biases*
_output_shapes	
:А
ѕ
8target_q_func/action_value/fully_connected/biases/AssignAssign1target_q_func/action_value/fully_connected/biasesCtarget_q_func/action_value/fully_connected/biases/Initializer/Const*
T0*
validate_shape(*
use_locking(*D
_class:
86loc:@target_q_func/action_value/fully_connected/biases*
_output_shapes	
:А
б
6target_q_func/action_value/fully_connected/biases/readIdentity1target_q_func/action_value/fully_connected/biases*
T0*D
_class:
86loc:@target_q_func/action_value/fully_connected/biases*
_output_shapes	
:А
м
1target_q_func/action_value/fully_connected/MatMulMatMultarget_q_func/Flatten/Reshape7target_q_func/action_value/fully_connected/weights/read*
transpose_a( *
transpose_b( *
T0*(
_output_shapes
:€€€€€€€€€А
т
2target_q_func/action_value/fully_connected/BiasAddBiasAdd1target_q_func/action_value/fully_connected/MatMul6target_q_func/action_value/fully_connected/biases/read*
T0*
data_formatNHWC*(
_output_shapes
:€€€€€€€€€А
Ю
/target_q_func/action_value/fully_connected/ReluRelu2target_q_func/action_value/fully_connected/BiasAdd*
T0*(
_output_shapes
:€€€€€€€€€А
п
Utarget_q_func/action_value/fully_connected_1/weights/Initializer/random_uniform/shapeConst*
valueB"      *
dtype0*G
_class=
;9loc:@target_q_func/action_value/fully_connected_1/weights*
_output_shapes
:
б
Starget_q_func/action_value/fully_connected_1/weights/Initializer/random_uniform/minConst*
valueB
 *≤_Њ*
dtype0*G
_class=
;9loc:@target_q_func/action_value/fully_connected_1/weights*
_output_shapes
: 
б
Starget_q_func/action_value/fully_connected_1/weights/Initializer/random_uniform/maxConst*
valueB
 *≤_>*
dtype0*G
_class=
;9loc:@target_q_func/action_value/fully_connected_1/weights*
_output_shapes
: 
ё
]target_q_func/action_value/fully_connected_1/weights/Initializer/random_uniform/RandomUniformRandomUniformUtarget_q_func/action_value/fully_connected_1/weights/Initializer/random_uniform/shape*

seed *
seed2 *
dtype0*
T0*G
_class=
;9loc:@target_q_func/action_value/fully_connected_1/weights*
_output_shapes
:	А
о
Starget_q_func/action_value/fully_connected_1/weights/Initializer/random_uniform/subSubStarget_q_func/action_value/fully_connected_1/weights/Initializer/random_uniform/maxStarget_q_func/action_value/fully_connected_1/weights/Initializer/random_uniform/min*
T0*G
_class=
;9loc:@target_q_func/action_value/fully_connected_1/weights*
_output_shapes
: 
Б
Starget_q_func/action_value/fully_connected_1/weights/Initializer/random_uniform/mulMul]target_q_func/action_value/fully_connected_1/weights/Initializer/random_uniform/RandomUniformStarget_q_func/action_value/fully_connected_1/weights/Initializer/random_uniform/sub*
T0*G
_class=
;9loc:@target_q_func/action_value/fully_connected_1/weights*
_output_shapes
:	А
у
Otarget_q_func/action_value/fully_connected_1/weights/Initializer/random_uniformAddStarget_q_func/action_value/fully_connected_1/weights/Initializer/random_uniform/mulStarget_q_func/action_value/fully_connected_1/weights/Initializer/random_uniform/min*
T0*G
_class=
;9loc:@target_q_func/action_value/fully_connected_1/weights*
_output_shapes
:	А
у
4target_q_func/action_value/fully_connected_1/weights
VariableV2*
shape:	А*
dtype0*
	container *
shared_name *G
_class=
;9loc:@target_q_func/action_value/fully_connected_1/weights*
_output_shapes
:	А
и
;target_q_func/action_value/fully_connected_1/weights/AssignAssign4target_q_func/action_value/fully_connected_1/weightsOtarget_q_func/action_value/fully_connected_1/weights/Initializer/random_uniform*
T0*
validate_shape(*
use_locking(*G
_class=
;9loc:@target_q_func/action_value/fully_connected_1/weights*
_output_shapes
:	А
о
9target_q_func/action_value/fully_connected_1/weights/readIdentity4target_q_func/action_value/fully_connected_1/weights*
T0*G
_class=
;9loc:@target_q_func/action_value/fully_connected_1/weights*
_output_shapes
:	А
Џ
Etarget_q_func/action_value/fully_connected_1/biases/Initializer/ConstConst*
valueB*    *
dtype0*F
_class<
:8loc:@target_q_func/action_value/fully_connected_1/biases*
_output_shapes
:
з
3target_q_func/action_value/fully_connected_1/biases
VariableV2*
shape:*
dtype0*
	container *
shared_name *F
_class<
:8loc:@target_q_func/action_value/fully_connected_1/biases*
_output_shapes
:
÷
:target_q_func/action_value/fully_connected_1/biases/AssignAssign3target_q_func/action_value/fully_connected_1/biasesEtarget_q_func/action_value/fully_connected_1/biases/Initializer/Const*
T0*
validate_shape(*
use_locking(*F
_class<
:8loc:@target_q_func/action_value/fully_connected_1/biases*
_output_shapes
:
ж
8target_q_func/action_value/fully_connected_1/biases/readIdentity3target_q_func/action_value/fully_connected_1/biases*
T0*F
_class<
:8loc:@target_q_func/action_value/fully_connected_1/biases*
_output_shapes
:
Б
3target_q_func/action_value/fully_connected_1/MatMulMatMul/target_q_func/action_value/fully_connected/Relu9target_q_func/action_value/fully_connected_1/weights/read*
transpose_a( *
transpose_b( *
T0*'
_output_shapes
:€€€€€€€€€
ч
4target_q_func/action_value/fully_connected_1/BiasAddBiasAdd3target_q_func/action_value/fully_connected_1/MatMul8target_q_func/action_value/fully_connected_1/biases/read*
T0*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€
U
target_q_a/sub/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
c
target_q_a/subSubtarget_q_a/sub/xdone_mask_ph*
T0*#
_output_shapes
:€€€€€€€€€
U
target_q_a/Pow/xConst*
valueB
 *§p}?*
dtype0*
_output_shapes
: 
`
target_q_a/PowPowtarget_q_a/Pow/x	opt_steps*
T0*#
_output_shapes
:€€€€€€€€€
c
target_q_a/mulMultarget_q_a/subtarget_q_a/Pow*
T0*#
_output_shapes
:€€€€€€€€€
b
 target_q_a/Max/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
Є
target_q_a/MaxMax4target_q_func/action_value/fully_connected_1/BiasAdd target_q_a/Max/reduction_indices*
	keep_dims( *
T0*

Tidx0*#
_output_shapes
:€€€€€€€€€
e
target_q_a/mul_1Multarget_q_a/multarget_q_a/Max*
T0*#
_output_shapes
:€€€€€€€€€
_
target_q_a/addAddrew_t_phtarget_q_a/mul_1*
T0*#
_output_shapes
:€€€€€€€€€
p
"Compute_bellman_error/StopGradientStopGradienttarget_q_a/add*
T0*#
_output_shapes
:€€€€€€€€€
Е
Compute_bellman_error/subSubpred_q_a/pred_q_a"Compute_bellman_error/StopGradient*
T0*#
_output_shapes
:€€€€€€€€€
i
Compute_bellman_error/AbsAbsCompute_bellman_error/sub*
T0*#
_output_shapes
:€€€€€€€€€
a
Compute_bellman_error/Less/yConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Й
Compute_bellman_error/LessLessCompute_bellman_error/AbsCompute_bellman_error/Less/y*
T0*#
_output_shapes
:€€€€€€€€€
o
Compute_bellman_error/SquareSquareCompute_bellman_error/sub*
T0*#
_output_shapes
:€€€€€€€€€
`
Compute_bellman_error/mul/yConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
Й
Compute_bellman_error/mulMulCompute_bellman_error/SquareCompute_bellman_error/mul/y*
T0*#
_output_shapes
:€€€€€€€€€
k
Compute_bellman_error/Abs_1AbsCompute_bellman_error/sub*
T0*#
_output_shapes
:€€€€€€€€€
b
Compute_bellman_error/sub_1/yConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
М
Compute_bellman_error/sub_1SubCompute_bellman_error/Abs_1Compute_bellman_error/sub_1/y*
T0*#
_output_shapes
:€€€€€€€€€
b
Compute_bellman_error/mul_1/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
М
Compute_bellman_error/mul_1MulCompute_bellman_error/mul_1/xCompute_bellman_error/sub_1*
T0*#
_output_shapes
:€€€€€€€€€
®
Compute_bellman_error/SelectSelectCompute_bellman_error/LessCompute_bellman_error/mulCompute_bellman_error/mul_1*
T0*#
_output_shapes
:€€€€€€€€€
e
Compute_bellman_error/ConstConst*
valueB: *
dtype0*
_output_shapes
:
°
!Compute_bellman_error/total_errorSumCompute_bellman_error/SelectCompute_bellman_error/Const*
	keep_dims( *
T0*

Tidx0*
_output_shapes
: 
N
learning_ratePlaceholder*
dtype0*
shape: *
_output_shapes
: 
\
Optimizer/gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
^
Optimizer/gradients/ConstConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
w
Optimizer/gradients/FillFillOptimizer/gradients/ShapeOptimizer/gradients/Const*
T0*
_output_shapes
: 
Т
HOptimizer/gradients/Compute_bellman_error/total_error_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
д
BOptimizer/gradients/Compute_bellman_error/total_error_grad/ReshapeReshapeOptimizer/gradients/FillHOptimizer/gradients/Compute_bellman_error/total_error_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
Ь
@Optimizer/gradients/Compute_bellman_error/total_error_grad/ShapeShapeCompute_bellman_error/Select*
T0*
out_type0*
_output_shapes
:
Н
?Optimizer/gradients/Compute_bellman_error/total_error_grad/TileTileBOptimizer/gradients/Compute_bellman_error/total_error_grad/Reshape@Optimizer/gradients/Compute_bellman_error/total_error_grad/Shape*
T0*

Tmultiples0*#
_output_shapes
:€€€€€€€€€
Ц
@Optimizer/gradients/Compute_bellman_error/Select_grad/zeros_like	ZerosLikeCompute_bellman_error/mul*
T0*#
_output_shapes
:€€€€€€€€€
У
<Optimizer/gradients/Compute_bellman_error/Select_grad/SelectSelectCompute_bellman_error/Less?Optimizer/gradients/Compute_bellman_error/total_error_grad/Tile@Optimizer/gradients/Compute_bellman_error/Select_grad/zeros_like*
T0*#
_output_shapes
:€€€€€€€€€
Х
>Optimizer/gradients/Compute_bellman_error/Select_grad/Select_1SelectCompute_bellman_error/Less@Optimizer/gradients/Compute_bellman_error/Select_grad/zeros_like?Optimizer/gradients/Compute_bellman_error/total_error_grad/Tile*
T0*#
_output_shapes
:€€€€€€€€€
ќ
FOptimizer/gradients/Compute_bellman_error/Select_grad/tuple/group_depsNoOp=^Optimizer/gradients/Compute_bellman_error/Select_grad/Select?^Optimizer/gradients/Compute_bellman_error/Select_grad/Select_1
а
NOptimizer/gradients/Compute_bellman_error/Select_grad/tuple/control_dependencyIdentity<Optimizer/gradients/Compute_bellman_error/Select_grad/SelectG^Optimizer/gradients/Compute_bellman_error/Select_grad/tuple/group_deps*
T0*O
_classE
CAloc:@Optimizer/gradients/Compute_bellman_error/Select_grad/Select*#
_output_shapes
:€€€€€€€€€
ж
POptimizer/gradients/Compute_bellman_error/Select_grad/tuple/control_dependency_1Identity>Optimizer/gradients/Compute_bellman_error/Select_grad/Select_1G^Optimizer/gradients/Compute_bellman_error/Select_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@Optimizer/gradients/Compute_bellman_error/Select_grad/Select_1*#
_output_shapes
:€€€€€€€€€
Ф
8Optimizer/gradients/Compute_bellman_error/mul_grad/ShapeShapeCompute_bellman_error/Square*
T0*
out_type0*
_output_shapes
:
}
:Optimizer/gradients/Compute_bellman_error/mul_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
Ф
HOptimizer/gradients/Compute_bellman_error/mul_grad/BroadcastGradientArgsBroadcastGradientArgs8Optimizer/gradients/Compute_bellman_error/mul_grad/Shape:Optimizer/gradients/Compute_bellman_error/mul_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Ў
6Optimizer/gradients/Compute_bellman_error/mul_grad/mulMulNOptimizer/gradients/Compute_bellman_error/Select_grad/tuple/control_dependencyCompute_bellman_error/mul/y*
T0*#
_output_shapes
:€€€€€€€€€
€
6Optimizer/gradients/Compute_bellman_error/mul_grad/SumSum6Optimizer/gradients/Compute_bellman_error/mul_grad/mulHOptimizer/gradients/Compute_bellman_error/mul_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
у
:Optimizer/gradients/Compute_bellman_error/mul_grad/ReshapeReshape6Optimizer/gradients/Compute_bellman_error/mul_grad/Sum8Optimizer/gradients/Compute_bellman_error/mul_grad/Shape*
T0*
Tshape0*#
_output_shapes
:€€€€€€€€€
џ
8Optimizer/gradients/Compute_bellman_error/mul_grad/mul_1MulCompute_bellman_error/SquareNOptimizer/gradients/Compute_bellman_error/Select_grad/tuple/control_dependency*
T0*#
_output_shapes
:€€€€€€€€€
Е
8Optimizer/gradients/Compute_bellman_error/mul_grad/Sum_1Sum8Optimizer/gradients/Compute_bellman_error/mul_grad/mul_1JOptimizer/gradients/Compute_bellman_error/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
м
<Optimizer/gradients/Compute_bellman_error/mul_grad/Reshape_1Reshape8Optimizer/gradients/Compute_bellman_error/mul_grad/Sum_1:Optimizer/gradients/Compute_bellman_error/mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
«
COptimizer/gradients/Compute_bellman_error/mul_grad/tuple/group_depsNoOp;^Optimizer/gradients/Compute_bellman_error/mul_grad/Reshape=^Optimizer/gradients/Compute_bellman_error/mul_grad/Reshape_1
÷
KOptimizer/gradients/Compute_bellman_error/mul_grad/tuple/control_dependencyIdentity:Optimizer/gradients/Compute_bellman_error/mul_grad/ReshapeD^Optimizer/gradients/Compute_bellman_error/mul_grad/tuple/group_deps*
T0*M
_classC
A?loc:@Optimizer/gradients/Compute_bellman_error/mul_grad/Reshape*#
_output_shapes
:€€€€€€€€€
ѕ
MOptimizer/gradients/Compute_bellman_error/mul_grad/tuple/control_dependency_1Identity<Optimizer/gradients/Compute_bellman_error/mul_grad/Reshape_1D^Optimizer/gradients/Compute_bellman_error/mul_grad/tuple/group_deps*
T0*O
_classE
CAloc:@Optimizer/gradients/Compute_bellman_error/mul_grad/Reshape_1*
_output_shapes
: 
}
:Optimizer/gradients/Compute_bellman_error/mul_1_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Ч
<Optimizer/gradients/Compute_bellman_error/mul_1_grad/Shape_1ShapeCompute_bellman_error/sub_1*
T0*
out_type0*
_output_shapes
:
Ъ
JOptimizer/gradients/Compute_bellman_error/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs:Optimizer/gradients/Compute_bellman_error/mul_1_grad/Shape<Optimizer/gradients/Compute_bellman_error/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
№
8Optimizer/gradients/Compute_bellman_error/mul_1_grad/mulMulPOptimizer/gradients/Compute_bellman_error/Select_grad/tuple/control_dependency_1Compute_bellman_error/sub_1*
T0*#
_output_shapes
:€€€€€€€€€
Е
8Optimizer/gradients/Compute_bellman_error/mul_1_grad/SumSum8Optimizer/gradients/Compute_bellman_error/mul_1_grad/mulJOptimizer/gradients/Compute_bellman_error/mul_1_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
м
<Optimizer/gradients/Compute_bellman_error/mul_1_grad/ReshapeReshape8Optimizer/gradients/Compute_bellman_error/mul_1_grad/Sum:Optimizer/gradients/Compute_bellman_error/mul_1_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
а
:Optimizer/gradients/Compute_bellman_error/mul_1_grad/mul_1MulCompute_bellman_error/mul_1/xPOptimizer/gradients/Compute_bellman_error/Select_grad/tuple/control_dependency_1*
T0*#
_output_shapes
:€€€€€€€€€
Л
:Optimizer/gradients/Compute_bellman_error/mul_1_grad/Sum_1Sum:Optimizer/gradients/Compute_bellman_error/mul_1_grad/mul_1LOptimizer/gradients/Compute_bellman_error/mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
€
>Optimizer/gradients/Compute_bellman_error/mul_1_grad/Reshape_1Reshape:Optimizer/gradients/Compute_bellman_error/mul_1_grad/Sum_1<Optimizer/gradients/Compute_bellman_error/mul_1_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:€€€€€€€€€
Ќ
EOptimizer/gradients/Compute_bellman_error/mul_1_grad/tuple/group_depsNoOp=^Optimizer/gradients/Compute_bellman_error/mul_1_grad/Reshape?^Optimizer/gradients/Compute_bellman_error/mul_1_grad/Reshape_1
—
MOptimizer/gradients/Compute_bellman_error/mul_1_grad/tuple/control_dependencyIdentity<Optimizer/gradients/Compute_bellman_error/mul_1_grad/ReshapeF^Optimizer/gradients/Compute_bellman_error/mul_1_grad/tuple/group_deps*
T0*O
_classE
CAloc:@Optimizer/gradients/Compute_bellman_error/mul_1_grad/Reshape*
_output_shapes
: 
д
OOptimizer/gradients/Compute_bellman_error/mul_1_grad/tuple/control_dependency_1Identity>Optimizer/gradients/Compute_bellman_error/mul_1_grad/Reshape_1F^Optimizer/gradients/Compute_bellman_error/mul_1_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@Optimizer/gradients/Compute_bellman_error/mul_1_grad/Reshape_1*#
_output_shapes
:€€€€€€€€€
ќ
;Optimizer/gradients/Compute_bellman_error/Square_grad/mul/xConstL^Optimizer/gradients/Compute_bellman_error/mul_grad/tuple/control_dependency*
valueB
 *   @*
dtype0*
_output_shapes
: 
∆
9Optimizer/gradients/Compute_bellman_error/Square_grad/mulMul;Optimizer/gradients/Compute_bellman_error/Square_grad/mul/xCompute_bellman_error/sub*
T0*#
_output_shapes
:€€€€€€€€€
ш
;Optimizer/gradients/Compute_bellman_error/Square_grad/mul_1MulKOptimizer/gradients/Compute_bellman_error/mul_grad/tuple/control_dependency9Optimizer/gradients/Compute_bellman_error/Square_grad/mul*
T0*#
_output_shapes
:€€€€€€€€€
Х
:Optimizer/gradients/Compute_bellman_error/sub_1_grad/ShapeShapeCompute_bellman_error/Abs_1*
T0*
out_type0*
_output_shapes
:

<Optimizer/gradients/Compute_bellman_error/sub_1_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
Ъ
JOptimizer/gradients/Compute_bellman_error/sub_1_grad/BroadcastGradientArgsBroadcastGradientArgs:Optimizer/gradients/Compute_bellman_error/sub_1_grad/Shape<Optimizer/gradients/Compute_bellman_error/sub_1_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
Ь
8Optimizer/gradients/Compute_bellman_error/sub_1_grad/SumSumOOptimizer/gradients/Compute_bellman_error/mul_1_grad/tuple/control_dependency_1JOptimizer/gradients/Compute_bellman_error/sub_1_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
щ
<Optimizer/gradients/Compute_bellman_error/sub_1_grad/ReshapeReshape8Optimizer/gradients/Compute_bellman_error/sub_1_grad/Sum:Optimizer/gradients/Compute_bellman_error/sub_1_grad/Shape*
T0*
Tshape0*#
_output_shapes
:€€€€€€€€€
†
:Optimizer/gradients/Compute_bellman_error/sub_1_grad/Sum_1SumOOptimizer/gradients/Compute_bellman_error/mul_1_grad/tuple/control_dependency_1LOptimizer/gradients/Compute_bellman_error/sub_1_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
Ю
8Optimizer/gradients/Compute_bellman_error/sub_1_grad/NegNeg:Optimizer/gradients/Compute_bellman_error/sub_1_grad/Sum_1*
T0*
_output_shapes
:
р
>Optimizer/gradients/Compute_bellman_error/sub_1_grad/Reshape_1Reshape8Optimizer/gradients/Compute_bellman_error/sub_1_grad/Neg<Optimizer/gradients/Compute_bellman_error/sub_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
Ќ
EOptimizer/gradients/Compute_bellman_error/sub_1_grad/tuple/group_depsNoOp=^Optimizer/gradients/Compute_bellman_error/sub_1_grad/Reshape?^Optimizer/gradients/Compute_bellman_error/sub_1_grad/Reshape_1
ё
MOptimizer/gradients/Compute_bellman_error/sub_1_grad/tuple/control_dependencyIdentity<Optimizer/gradients/Compute_bellman_error/sub_1_grad/ReshapeF^Optimizer/gradients/Compute_bellman_error/sub_1_grad/tuple/group_deps*
T0*O
_classE
CAloc:@Optimizer/gradients/Compute_bellman_error/sub_1_grad/Reshape*#
_output_shapes
:€€€€€€€€€
„
OOptimizer/gradients/Compute_bellman_error/sub_1_grad/tuple/control_dependency_1Identity>Optimizer/gradients/Compute_bellman_error/sub_1_grad/Reshape_1F^Optimizer/gradients/Compute_bellman_error/sub_1_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@Optimizer/gradients/Compute_bellman_error/sub_1_grad/Reshape_1*
_output_shapes
: 
К
9Optimizer/gradients/Compute_bellman_error/Abs_1_grad/SignSignCompute_bellman_error/sub*
T0*#
_output_shapes
:€€€€€€€€€
ч
8Optimizer/gradients/Compute_bellman_error/Abs_1_grad/mulMulMOptimizer/gradients/Compute_bellman_error/sub_1_grad/tuple/control_dependency9Optimizer/gradients/Compute_bellman_error/Abs_1_grad/Sign*
T0*#
_output_shapes
:€€€€€€€€€
Ю
Optimizer/gradients/AddNAddN;Optimizer/gradients/Compute_bellman_error/Square_grad/mul_18Optimizer/gradients/Compute_bellman_error/Abs_1_grad/mul*
N*
T0*N
_classD
B@loc:@Optimizer/gradients/Compute_bellman_error/Square_grad/mul_1*#
_output_shapes
:€€€€€€€€€
Й
8Optimizer/gradients/Compute_bellman_error/sub_grad/ShapeShapepred_q_a/pred_q_a*
T0*
out_type0*
_output_shapes
:
Ь
:Optimizer/gradients/Compute_bellman_error/sub_grad/Shape_1Shape"Compute_bellman_error/StopGradient*
T0*
out_type0*
_output_shapes
:
Ф
HOptimizer/gradients/Compute_bellman_error/sub_grad/BroadcastGradientArgsBroadcastGradientArgs8Optimizer/gradients/Compute_bellman_error/sub_grad/Shape:Optimizer/gradients/Compute_bellman_error/sub_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
б
6Optimizer/gradients/Compute_bellman_error/sub_grad/SumSumOptimizer/gradients/AddNHOptimizer/gradients/Compute_bellman_error/sub_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
у
:Optimizer/gradients/Compute_bellman_error/sub_grad/ReshapeReshape6Optimizer/gradients/Compute_bellman_error/sub_grad/Sum8Optimizer/gradients/Compute_bellman_error/sub_grad/Shape*
T0*
Tshape0*#
_output_shapes
:€€€€€€€€€
е
8Optimizer/gradients/Compute_bellman_error/sub_grad/Sum_1SumOptimizer/gradients/AddNJOptimizer/gradients/Compute_bellman_error/sub_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
Ъ
6Optimizer/gradients/Compute_bellman_error/sub_grad/NegNeg8Optimizer/gradients/Compute_bellman_error/sub_grad/Sum_1*
T0*
_output_shapes
:
ч
<Optimizer/gradients/Compute_bellman_error/sub_grad/Reshape_1Reshape6Optimizer/gradients/Compute_bellman_error/sub_grad/Neg:Optimizer/gradients/Compute_bellman_error/sub_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:€€€€€€€€€
«
COptimizer/gradients/Compute_bellman_error/sub_grad/tuple/group_depsNoOp;^Optimizer/gradients/Compute_bellman_error/sub_grad/Reshape=^Optimizer/gradients/Compute_bellman_error/sub_grad/Reshape_1
÷
KOptimizer/gradients/Compute_bellman_error/sub_grad/tuple/control_dependencyIdentity:Optimizer/gradients/Compute_bellman_error/sub_grad/ReshapeD^Optimizer/gradients/Compute_bellman_error/sub_grad/tuple/group_deps*
T0*M
_classC
A?loc:@Optimizer/gradients/Compute_bellman_error/sub_grad/Reshape*#
_output_shapes
:€€€€€€€€€
№
MOptimizer/gradients/Compute_bellman_error/sub_grad/tuple/control_dependency_1Identity<Optimizer/gradients/Compute_bellman_error/sub_grad/Reshape_1D^Optimizer/gradients/Compute_bellman_error/sub_grad/tuple/group_deps*
T0*O
_classE
CAloc:@Optimizer/gradients/Compute_bellman_error/sub_grad/Reshape_1*#
_output_shapes
:€€€€€€€€€
|
0Optimizer/gradients/pred_q_a/pred_q_a_grad/ShapeShapepred_q_a/mul*
T0*
out_type0*
_output_shapes
:
q
/Optimizer/gradients/pred_q_a/pred_q_a_grad/SizeConst*
value	B :*
dtype0*
_output_shapes
: 
ђ
.Optimizer/gradients/pred_q_a/pred_q_a_grad/addAdd#pred_q_a/pred_q_a/reduction_indices/Optimizer/gradients/pred_q_a/pred_q_a_grad/Size*
T0*
_output_shapes
: 
Љ
.Optimizer/gradients/pred_q_a/pred_q_a_grad/modFloorMod.Optimizer/gradients/pred_q_a/pred_q_a_grad/add/Optimizer/gradients/pred_q_a/pred_q_a_grad/Size*
T0*
_output_shapes
: 
u
2Optimizer/gradients/pred_q_a/pred_q_a_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
x
6Optimizer/gradients/pred_q_a/pred_q_a_grad/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
x
6Optimizer/gradients/pred_q_a/pred_q_a_grad/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
В
0Optimizer/gradients/pred_q_a/pred_q_a_grad/rangeRange6Optimizer/gradients/pred_q_a/pred_q_a_grad/range/start/Optimizer/gradients/pred_q_a/pred_q_a_grad/Size6Optimizer/gradients/pred_q_a/pred_q_a_grad/range/delta*

Tidx0*
_output_shapes
:
w
5Optimizer/gradients/pred_q_a/pred_q_a_grad/Fill/valueConst*
value	B :*
dtype0*
_output_shapes
: 
√
/Optimizer/gradients/pred_q_a/pred_q_a_grad/FillFill2Optimizer/gradients/pred_q_a/pred_q_a_grad/Shape_15Optimizer/gradients/pred_q_a/pred_q_a_grad/Fill/value*
T0*
_output_shapes
: 
≈
8Optimizer/gradients/pred_q_a/pred_q_a_grad/DynamicStitchDynamicStitch0Optimizer/gradients/pred_q_a/pred_q_a_grad/range.Optimizer/gradients/pred_q_a/pred_q_a_grad/mod0Optimizer/gradients/pred_q_a/pred_q_a_grad/Shape/Optimizer/gradients/pred_q_a/pred_q_a_grad/Fill*
N*
T0*#
_output_shapes
:€€€€€€€€€
v
4Optimizer/gradients/pred_q_a/pred_q_a_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
џ
2Optimizer/gradients/pred_q_a/pred_q_a_grad/MaximumMaximum8Optimizer/gradients/pred_q_a/pred_q_a_grad/DynamicStitch4Optimizer/gradients/pred_q_a/pred_q_a_grad/Maximum/y*
T0*#
_output_shapes
:€€€€€€€€€
 
3Optimizer/gradients/pred_q_a/pred_q_a_grad/floordivFloorDiv0Optimizer/gradients/pred_q_a/pred_q_a_grad/Shape2Optimizer/gradients/pred_q_a/pred_q_a_grad/Maximum*
T0*
_output_shapes
:
х
2Optimizer/gradients/pred_q_a/pred_q_a_grad/ReshapeReshapeKOptimizer/gradients/Compute_bellman_error/sub_grad/tuple/control_dependency8Optimizer/gradients/pred_q_a/pred_q_a_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
д
/Optimizer/gradients/pred_q_a/pred_q_a_grad/TileTile2Optimizer/gradients/pred_q_a/pred_q_a_grad/Reshape3Optimizer/gradients/pred_q_a/pred_q_a_grad/floordiv*
T0*

Tmultiples0*'
_output_shapes
:€€€€€€€€€
•
+Optimizer/gradients/pred_q_a/mul_grad/ShapeShape:over_options/q_func/action_value/fully_connected_1/BiasAdd*
T0*
out_type0*
_output_shapes
:
}
-Optimizer/gradients/pred_q_a/mul_grad/Shape_1Shapepred_q_a/one_hot*
T0*
out_type0*
_output_shapes
:
н
;Optimizer/gradients/pred_q_a/mul_grad/BroadcastGradientArgsBroadcastGradientArgs+Optimizer/gradients/pred_q_a/mul_grad/Shape-Optimizer/gradients/pred_q_a/mul_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
•
)Optimizer/gradients/pred_q_a/mul_grad/mulMul/Optimizer/gradients/pred_q_a/pred_q_a_grad/Tilepred_q_a/one_hot*
T0*'
_output_shapes
:€€€€€€€€€
Ў
)Optimizer/gradients/pred_q_a/mul_grad/SumSum)Optimizer/gradients/pred_q_a/mul_grad/mul;Optimizer/gradients/pred_q_a/mul_grad/BroadcastGradientArgs*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
–
-Optimizer/gradients/pred_q_a/mul_grad/ReshapeReshape)Optimizer/gradients/pred_q_a/mul_grad/Sum+Optimizer/gradients/pred_q_a/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
—
+Optimizer/gradients/pred_q_a/mul_grad/mul_1Mul:over_options/q_func/action_value/fully_connected_1/BiasAdd/Optimizer/gradients/pred_q_a/pred_q_a_grad/Tile*
T0*'
_output_shapes
:€€€€€€€€€
ё
+Optimizer/gradients/pred_q_a/mul_grad/Sum_1Sum+Optimizer/gradients/pred_q_a/mul_grad/mul_1=Optimizer/gradients/pred_q_a/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
÷
/Optimizer/gradients/pred_q_a/mul_grad/Reshape_1Reshape+Optimizer/gradients/pred_q_a/mul_grad/Sum_1-Optimizer/gradients/pred_q_a/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
†
6Optimizer/gradients/pred_q_a/mul_grad/tuple/group_depsNoOp.^Optimizer/gradients/pred_q_a/mul_grad/Reshape0^Optimizer/gradients/pred_q_a/mul_grad/Reshape_1
¶
>Optimizer/gradients/pred_q_a/mul_grad/tuple/control_dependencyIdentity-Optimizer/gradients/pred_q_a/mul_grad/Reshape7^Optimizer/gradients/pred_q_a/mul_grad/tuple/group_deps*
T0*@
_class6
42loc:@Optimizer/gradients/pred_q_a/mul_grad/Reshape*'
_output_shapes
:€€€€€€€€€
ђ
@Optimizer/gradients/pred_q_a/mul_grad/tuple/control_dependency_1Identity/Optimizer/gradients/pred_q_a/mul_grad/Reshape_17^Optimizer/gradients/pred_q_a/mul_grad/tuple/group_deps*
T0*B
_class8
64loc:@Optimizer/gradients/pred_q_a/mul_grad/Reshape_1*'
_output_shapes
:€€€€€€€€€
к
_Optimizer/gradients/over_options/q_func/action_value/fully_connected_1/BiasAdd_grad/BiasAddGradBiasAddGrad>Optimizer/gradients/pred_q_a/mul_grad/tuple/control_dependency*
T0*
data_formatNHWC*
_output_shapes
:
П
dOptimizer/gradients/over_options/q_func/action_value/fully_connected_1/BiasAdd_grad/tuple/group_depsNoOp?^Optimizer/gradients/pred_q_a/mul_grad/tuple/control_dependency`^Optimizer/gradients/over_options/q_func/action_value/fully_connected_1/BiasAdd_grad/BiasAddGrad
У
lOptimizer/gradients/over_options/q_func/action_value/fully_connected_1/BiasAdd_grad/tuple/control_dependencyIdentity>Optimizer/gradients/pred_q_a/mul_grad/tuple/control_dependencye^Optimizer/gradients/over_options/q_func/action_value/fully_connected_1/BiasAdd_grad/tuple/group_deps*
T0*@
_class6
42loc:@Optimizer/gradients/pred_q_a/mul_grad/Reshape*'
_output_shapes
:€€€€€€€€€
џ
nOptimizer/gradients/over_options/q_func/action_value/fully_connected_1/BiasAdd_grad/tuple/control_dependency_1Identity_Optimizer/gradients/over_options/q_func/action_value/fully_connected_1/BiasAdd_grad/BiasAddGrade^Optimizer/gradients/over_options/q_func/action_value/fully_connected_1/BiasAdd_grad/tuple/group_deps*
T0*r
_classh
fdloc:@Optimizer/gradients/over_options/q_func/action_value/fully_connected_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
л
YOptimizer/gradients/over_options/q_func/action_value/fully_connected_1/MatMul_grad/MatMulMatMullOptimizer/gradients/over_options/q_func/action_value/fully_connected_1/BiasAdd_grad/tuple/control_dependency?over_options/q_func/action_value/fully_connected_1/weights/read*
transpose_a( *
transpose_b(*
T0*(
_output_shapes
:€€€€€€€€€А
Џ
[Optimizer/gradients/over_options/q_func/action_value/fully_connected_1/MatMul_grad/MatMul_1MatMul5over_options/q_func/action_value/fully_connected/RelulOptimizer/gradients/over_options/q_func/action_value/fully_connected_1/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
T0*
_output_shapes
:	А
•
cOptimizer/gradients/over_options/q_func/action_value/fully_connected_1/MatMul_grad/tuple/group_depsNoOpZ^Optimizer/gradients/over_options/q_func/action_value/fully_connected_1/MatMul_grad/MatMul\^Optimizer/gradients/over_options/q_func/action_value/fully_connected_1/MatMul_grad/MatMul_1
ў
kOptimizer/gradients/over_options/q_func/action_value/fully_connected_1/MatMul_grad/tuple/control_dependencyIdentityYOptimizer/gradients/over_options/q_func/action_value/fully_connected_1/MatMul_grad/MatMuld^Optimizer/gradients/over_options/q_func/action_value/fully_connected_1/MatMul_grad/tuple/group_deps*
T0*l
_classb
`^loc:@Optimizer/gradients/over_options/q_func/action_value/fully_connected_1/MatMul_grad/MatMul*(
_output_shapes
:€€€€€€€€€А
÷
mOptimizer/gradients/over_options/q_func/action_value/fully_connected_1/MatMul_grad/tuple/control_dependency_1Identity[Optimizer/gradients/over_options/q_func/action_value/fully_connected_1/MatMul_grad/MatMul_1d^Optimizer/gradients/over_options/q_func/action_value/fully_connected_1/MatMul_grad/tuple/group_deps*
T0*n
_classd
b`loc:@Optimizer/gradients/over_options/q_func/action_value/fully_connected_1/MatMul_grad/MatMul_1*
_output_shapes
:	А
Ї
WOptimizer/gradients/over_options/q_func/action_value/fully_connected/Relu_grad/ReluGradReluGradkOptimizer/gradients/over_options/q_func/action_value/fully_connected_1/MatMul_grad/tuple/control_dependency5over_options/q_func/action_value/fully_connected/Relu*
T0*(
_output_shapes
:€€€€€€€€€А
В
]Optimizer/gradients/over_options/q_func/action_value/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGradWOptimizer/gradients/over_options/q_func/action_value/fully_connected/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:А
§
bOptimizer/gradients/over_options/q_func/action_value/fully_connected/BiasAdd_grad/tuple/group_depsNoOpX^Optimizer/gradients/over_options/q_func/action_value/fully_connected/Relu_grad/ReluGrad^^Optimizer/gradients/over_options/q_func/action_value/fully_connected/BiasAdd_grad/BiasAddGrad
”
jOptimizer/gradients/over_options/q_func/action_value/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentityWOptimizer/gradients/over_options/q_func/action_value/fully_connected/Relu_grad/ReluGradc^Optimizer/gradients/over_options/q_func/action_value/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*j
_class`
^\loc:@Optimizer/gradients/over_options/q_func/action_value/fully_connected/Relu_grad/ReluGrad*(
_output_shapes
:€€€€€€€€€А
‘
lOptimizer/gradients/over_options/q_func/action_value/fully_connected/BiasAdd_grad/tuple/control_dependency_1Identity]Optimizer/gradients/over_options/q_func/action_value/fully_connected/BiasAdd_grad/BiasAddGradc^Optimizer/gradients/over_options/q_func/action_value/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*p
_classf
dbloc:@Optimizer/gradients/over_options/q_func/action_value/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:А
е
WOptimizer/gradients/over_options/q_func/action_value/fully_connected/MatMul_grad/MatMulMatMuljOptimizer/gradients/over_options/q_func/action_value/fully_connected/BiasAdd_grad/tuple/control_dependency=over_options/q_func/action_value/fully_connected/weights/read*
transpose_a( *
transpose_b(*
T0*(
_output_shapes
:€€€€€€€€€А
≈
YOptimizer/gradients/over_options/q_func/action_value/fully_connected/MatMul_grad/MatMul_1MatMul#over_options/q_func/Flatten/ReshapejOptimizer/gradients/over_options/q_func/action_value/fully_connected/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
transpose_b( *
T0* 
_output_shapes
:
АА
Я
aOptimizer/gradients/over_options/q_func/action_value/fully_connected/MatMul_grad/tuple/group_depsNoOpX^Optimizer/gradients/over_options/q_func/action_value/fully_connected/MatMul_grad/MatMulZ^Optimizer/gradients/over_options/q_func/action_value/fully_connected/MatMul_grad/MatMul_1
—
iOptimizer/gradients/over_options/q_func/action_value/fully_connected/MatMul_grad/tuple/control_dependencyIdentityWOptimizer/gradients/over_options/q_func/action_value/fully_connected/MatMul_grad/MatMulb^Optimizer/gradients/over_options/q_func/action_value/fully_connected/MatMul_grad/tuple/group_deps*
T0*j
_class`
^\loc:@Optimizer/gradients/over_options/q_func/action_value/fully_connected/MatMul_grad/MatMul*(
_output_shapes
:€€€€€€€€€А
ѕ
kOptimizer/gradients/over_options/q_func/action_value/fully_connected/MatMul_grad/tuple/control_dependency_1IdentityYOptimizer/gradients/over_options/q_func/action_value/fully_connected/MatMul_grad/MatMul_1b^Optimizer/gradients/over_options/q_func/action_value/fully_connected/MatMul_grad/tuple/group_deps*
T0*l
_classb
`^loc:@Optimizer/gradients/over_options/q_func/action_value/fully_connected/MatMul_grad/MatMul_1* 
_output_shapes
:
АА
©
BOptimizer/gradients/over_options/q_func/Flatten/Reshape_grad/ShapeShape'over_options/q_func/convnet/Conv_2/Relu*
T0*
out_type0*
_output_shapes
:
∆
DOptimizer/gradients/over_options/q_func/Flatten/Reshape_grad/ReshapeReshapeiOptimizer/gradients/over_options/q_func/action_value/fully_connected/MatMul_grad/tuple/control_dependencyBOptimizer/gradients/over_options/q_func/Flatten/Reshape_grad/Shape*
T0*
Tshape0*/
_output_shapes
:€€€€€€€€€@
ю
IOptimizer/gradients/over_options/q_func/convnet/Conv_2/Relu_grad/ReluGradReluGradDOptimizer/gradients/over_options/q_func/Flatten/Reshape_grad/Reshape'over_options/q_func/convnet/Conv_2/Relu*
T0*/
_output_shapes
:€€€€€€€€€@
е
OOptimizer/gradients/over_options/q_func/convnet/Conv_2/BiasAdd_grad/BiasAddGradBiasAddGradIOptimizer/gradients/over_options/q_func/convnet/Conv_2/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@
ъ
TOptimizer/gradients/over_options/q_func/convnet/Conv_2/BiasAdd_grad/tuple/group_depsNoOpJ^Optimizer/gradients/over_options/q_func/convnet/Conv_2/Relu_grad/ReluGradP^Optimizer/gradients/over_options/q_func/convnet/Conv_2/BiasAdd_grad/BiasAddGrad
Ґ
\Optimizer/gradients/over_options/q_func/convnet/Conv_2/BiasAdd_grad/tuple/control_dependencyIdentityIOptimizer/gradients/over_options/q_func/convnet/Conv_2/Relu_grad/ReluGradU^Optimizer/gradients/over_options/q_func/convnet/Conv_2/BiasAdd_grad/tuple/group_deps*
T0*\
_classR
PNloc:@Optimizer/gradients/over_options/q_func/convnet/Conv_2/Relu_grad/ReluGrad*/
_output_shapes
:€€€€€€€€€@
Ы
^Optimizer/gradients/over_options/q_func/convnet/Conv_2/BiasAdd_grad/tuple/control_dependency_1IdentityOOptimizer/gradients/over_options/q_func/convnet/Conv_2/BiasAdd_grad/BiasAddGradU^Optimizer/gradients/over_options/q_func/convnet/Conv_2/BiasAdd_grad/tuple/group_deps*
T0*b
_classX
VTloc:@Optimizer/gradients/over_options/q_func/convnet/Conv_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
і
MOptimizer/gradients/over_options/q_func/convnet/Conv_2/convolution_grad/ShapeShape'over_options/q_func/convnet/Conv_1/Relu*
T0*
out_type0*
_output_shapes
:
ь
[Optimizer/gradients/over_options/q_func/convnet/Conv_2/convolution_grad/Conv2DBackpropInputConv2DBackpropInputMOptimizer/gradients/over_options/q_func/convnet/Conv_2/convolution_grad/Shape/over_options/q_func/convnet/Conv_2/weights/read\Optimizer/gradients/over_options/q_func/convnet/Conv_2/BiasAdd_grad/tuple/control_dependency*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
®
OOptimizer/gradients/over_options/q_func/convnet/Conv_2/convolution_grad/Shape_1Const*%
valueB"      @   @   *
dtype0*
_output_shapes
:
‘
\Optimizer/gradients/over_options/q_func/convnet/Conv_2/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilter'over_options/q_func/convnet/Conv_1/ReluOOptimizer/gradients/over_options/q_func/convnet/Conv_2/convolution_grad/Shape_1\Optimizer/gradients/over_options/q_func/convnet/Conv_2/BiasAdd_grad/tuple/control_dependency*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*&
_output_shapes
:@@
Э
XOptimizer/gradients/over_options/q_func/convnet/Conv_2/convolution_grad/tuple/group_depsNoOp\^Optimizer/gradients/over_options/q_func/convnet/Conv_2/convolution_grad/Conv2DBackpropInput]^Optimizer/gradients/over_options/q_func/convnet/Conv_2/convolution_grad/Conv2DBackpropFilter
ќ
`Optimizer/gradients/over_options/q_func/convnet/Conv_2/convolution_grad/tuple/control_dependencyIdentity[Optimizer/gradients/over_options/q_func/convnet/Conv_2/convolution_grad/Conv2DBackpropInputY^Optimizer/gradients/over_options/q_func/convnet/Conv_2/convolution_grad/tuple/group_deps*
T0*n
_classd
b`loc:@Optimizer/gradients/over_options/q_func/convnet/Conv_2/convolution_grad/Conv2DBackpropInput*/
_output_shapes
:€€€€€€€€€@
…
bOptimizer/gradients/over_options/q_func/convnet/Conv_2/convolution_grad/tuple/control_dependency_1Identity\Optimizer/gradients/over_options/q_func/convnet/Conv_2/convolution_grad/Conv2DBackpropFilterY^Optimizer/gradients/over_options/q_func/convnet/Conv_2/convolution_grad/tuple/group_deps*
T0*o
_classe
caloc:@Optimizer/gradients/over_options/q_func/convnet/Conv_2/convolution_grad/Conv2DBackpropFilter*&
_output_shapes
:@@
Ъ
IOptimizer/gradients/over_options/q_func/convnet/Conv_1/Relu_grad/ReluGradReluGrad`Optimizer/gradients/over_options/q_func/convnet/Conv_2/convolution_grad/tuple/control_dependency'over_options/q_func/convnet/Conv_1/Relu*
T0*/
_output_shapes
:€€€€€€€€€@
е
OOptimizer/gradients/over_options/q_func/convnet/Conv_1/BiasAdd_grad/BiasAddGradBiasAddGradIOptimizer/gradients/over_options/q_func/convnet/Conv_1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@
ъ
TOptimizer/gradients/over_options/q_func/convnet/Conv_1/BiasAdd_grad/tuple/group_depsNoOpJ^Optimizer/gradients/over_options/q_func/convnet/Conv_1/Relu_grad/ReluGradP^Optimizer/gradients/over_options/q_func/convnet/Conv_1/BiasAdd_grad/BiasAddGrad
Ґ
\Optimizer/gradients/over_options/q_func/convnet/Conv_1/BiasAdd_grad/tuple/control_dependencyIdentityIOptimizer/gradients/over_options/q_func/convnet/Conv_1/Relu_grad/ReluGradU^Optimizer/gradients/over_options/q_func/convnet/Conv_1/BiasAdd_grad/tuple/group_deps*
T0*\
_classR
PNloc:@Optimizer/gradients/over_options/q_func/convnet/Conv_1/Relu_grad/ReluGrad*/
_output_shapes
:€€€€€€€€€@
Ы
^Optimizer/gradients/over_options/q_func/convnet/Conv_1/BiasAdd_grad/tuple/control_dependency_1IdentityOOptimizer/gradients/over_options/q_func/convnet/Conv_1/BiasAdd_grad/BiasAddGradU^Optimizer/gradients/over_options/q_func/convnet/Conv_1/BiasAdd_grad/tuple/group_deps*
T0*b
_classX
VTloc:@Optimizer/gradients/over_options/q_func/convnet/Conv_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
≤
MOptimizer/gradients/over_options/q_func/convnet/Conv_1/convolution_grad/ShapeShape%over_options/q_func/convnet/Conv/Relu*
T0*
out_type0*
_output_shapes
:
ь
[Optimizer/gradients/over_options/q_func/convnet/Conv_1/convolution_grad/Conv2DBackpropInputConv2DBackpropInputMOptimizer/gradients/over_options/q_func/convnet/Conv_1/convolution_grad/Shape/over_options/q_func/convnet/Conv_1/weights/read\Optimizer/gradients/over_options/q_func/convnet/Conv_1/BiasAdd_grad/tuple/control_dependency*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
®
OOptimizer/gradients/over_options/q_func/convnet/Conv_1/convolution_grad/Shape_1Const*%
valueB"          @   *
dtype0*
_output_shapes
:
“
\Optimizer/gradients/over_options/q_func/convnet/Conv_1/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilter%over_options/q_func/convnet/Conv/ReluOOptimizer/gradients/over_options/q_func/convnet/Conv_1/convolution_grad/Shape_1\Optimizer/gradients/over_options/q_func/convnet/Conv_1/BiasAdd_grad/tuple/control_dependency*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*&
_output_shapes
: @
Э
XOptimizer/gradients/over_options/q_func/convnet/Conv_1/convolution_grad/tuple/group_depsNoOp\^Optimizer/gradients/over_options/q_func/convnet/Conv_1/convolution_grad/Conv2DBackpropInput]^Optimizer/gradients/over_options/q_func/convnet/Conv_1/convolution_grad/Conv2DBackpropFilter
ќ
`Optimizer/gradients/over_options/q_func/convnet/Conv_1/convolution_grad/tuple/control_dependencyIdentity[Optimizer/gradients/over_options/q_func/convnet/Conv_1/convolution_grad/Conv2DBackpropInputY^Optimizer/gradients/over_options/q_func/convnet/Conv_1/convolution_grad/tuple/group_deps*
T0*n
_classd
b`loc:@Optimizer/gradients/over_options/q_func/convnet/Conv_1/convolution_grad/Conv2DBackpropInput*/
_output_shapes
:€€€€€€€€€ 
…
bOptimizer/gradients/over_options/q_func/convnet/Conv_1/convolution_grad/tuple/control_dependency_1Identity\Optimizer/gradients/over_options/q_func/convnet/Conv_1/convolution_grad/Conv2DBackpropFilterY^Optimizer/gradients/over_options/q_func/convnet/Conv_1/convolution_grad/tuple/group_deps*
T0*o
_classe
caloc:@Optimizer/gradients/over_options/q_func/convnet/Conv_1/convolution_grad/Conv2DBackpropFilter*&
_output_shapes
: @
Ц
GOptimizer/gradients/over_options/q_func/convnet/Conv/Relu_grad/ReluGradReluGrad`Optimizer/gradients/over_options/q_func/convnet/Conv_1/convolution_grad/tuple/control_dependency%over_options/q_func/convnet/Conv/Relu*
T0*/
_output_shapes
:€€€€€€€€€ 
б
MOptimizer/gradients/over_options/q_func/convnet/Conv/BiasAdd_grad/BiasAddGradBiasAddGradGOptimizer/gradients/over_options/q_func/convnet/Conv/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
: 
ф
ROptimizer/gradients/over_options/q_func/convnet/Conv/BiasAdd_grad/tuple/group_depsNoOpH^Optimizer/gradients/over_options/q_func/convnet/Conv/Relu_grad/ReluGradN^Optimizer/gradients/over_options/q_func/convnet/Conv/BiasAdd_grad/BiasAddGrad
Ъ
ZOptimizer/gradients/over_options/q_func/convnet/Conv/BiasAdd_grad/tuple/control_dependencyIdentityGOptimizer/gradients/over_options/q_func/convnet/Conv/Relu_grad/ReluGradS^Optimizer/gradients/over_options/q_func/convnet/Conv/BiasAdd_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@Optimizer/gradients/over_options/q_func/convnet/Conv/Relu_grad/ReluGrad*/
_output_shapes
:€€€€€€€€€ 
У
\Optimizer/gradients/over_options/q_func/convnet/Conv/BiasAdd_grad/tuple/control_dependency_1IdentityMOptimizer/gradients/over_options/q_func/convnet/Conv/BiasAdd_grad/BiasAddGradS^Optimizer/gradients/over_options/q_func/convnet/Conv/BiasAdd_grad/tuple/group_deps*
T0*`
_classV
TRloc:@Optimizer/gradients/over_options/q_func/convnet/Conv/BiasAdd_grad/BiasAddGrad*
_output_shapes
: 
£
KOptimizer/gradients/over_options/q_func/convnet/Conv/convolution_grad/ShapeShapeover_options/obs_t_float*
T0*
out_type0*
_output_shapes
:
ф
YOptimizer/gradients/over_options/q_func/convnet/Conv/convolution_grad/Conv2DBackpropInputConv2DBackpropInputKOptimizer/gradients/over_options/q_func/convnet/Conv/convolution_grad/Shape-over_options/q_func/convnet/Conv/weights/readZOptimizer/gradients/over_options/q_func/convnet/Conv/BiasAdd_grad/tuple/control_dependency*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
¶
MOptimizer/gradients/over_options/q_func/convnet/Conv/convolution_grad/Shape_1Const*%
valueB"             *
dtype0*
_output_shapes
:
њ
ZOptimizer/gradients/over_options/q_func/convnet/Conv/convolution_grad/Conv2DBackpropFilterConv2DBackpropFilterover_options/obs_t_floatMOptimizer/gradients/over_options/q_func/convnet/Conv/convolution_grad/Shape_1ZOptimizer/gradients/over_options/q_func/convnet/Conv/BiasAdd_grad/tuple/control_dependency*
T0*
strides
*
use_cudnn_on_gpu(*
paddingSAME*
data_formatNHWC*&
_output_shapes
: 
Ч
VOptimizer/gradients/over_options/q_func/convnet/Conv/convolution_grad/tuple/group_depsNoOpZ^Optimizer/gradients/over_options/q_func/convnet/Conv/convolution_grad/Conv2DBackpropInput[^Optimizer/gradients/over_options/q_func/convnet/Conv/convolution_grad/Conv2DBackpropFilter
∆
^Optimizer/gradients/over_options/q_func/convnet/Conv/convolution_grad/tuple/control_dependencyIdentityYOptimizer/gradients/over_options/q_func/convnet/Conv/convolution_grad/Conv2DBackpropInputW^Optimizer/gradients/over_options/q_func/convnet/Conv/convolution_grad/tuple/group_deps*
T0*l
_classb
`^loc:@Optimizer/gradients/over_options/q_func/convnet/Conv/convolution_grad/Conv2DBackpropInput*/
_output_shapes
:€€€€€€€€€	
Ѕ
`Optimizer/gradients/over_options/q_func/convnet/Conv/convolution_grad/tuple/control_dependency_1IdentityZOptimizer/gradients/over_options/q_func/convnet/Conv/convolution_grad/Conv2DBackpropFilterW^Optimizer/gradients/over_options/q_func/convnet/Conv/convolution_grad/tuple/group_deps*
T0*m
_classc
a_loc:@Optimizer/gradients/over_options/q_func/convnet/Conv/convolution_grad/Conv2DBackpropFilter*&
_output_shapes
: 
Ц
Optimizer/clip_by_norm/mulMul`Optimizer/gradients/over_options/q_func/convnet/Conv/convolution_grad/tuple/control_dependency_1`Optimizer/gradients/over_options/q_func/convnet/Conv/convolution_grad/tuple/control_dependency_1*
T0*&
_output_shapes
: 
u
Optimizer/clip_by_norm/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:
©
Optimizer/clip_by_norm/SumSumOptimizer/clip_by_norm/mulOptimizer/clip_by_norm/Const*
	keep_dims(*
T0*

Tidx0*&
_output_shapes
:
r
Optimizer/clip_by_norm/RsqrtRsqrtOptimizer/clip_by_norm/Sum*
T0*&
_output_shapes
:
c
Optimizer/clip_by_norm/mul_1/yConst*
valueB
 *   A*
dtype0*
_output_shapes
: 
÷
Optimizer/clip_by_norm/mul_1Mul`Optimizer/gradients/over_options/q_func/convnet/Conv/convolution_grad/tuple/control_dependency_1Optimizer/clip_by_norm/mul_1/y*
T0*&
_output_shapes
: 
c
Optimizer/clip_by_norm/Const_1Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
e
 Optimizer/clip_by_norm/truediv/yConst*
valueB
 *   A*
dtype0*
_output_shapes
: 
М
Optimizer/clip_by_norm/truedivRealDivOptimizer/clip_by_norm/Const_1 Optimizer/clip_by_norm/truediv/y*
T0*
_output_shapes
: 
Ш
Optimizer/clip_by_norm/MinimumMinimumOptimizer/clip_by_norm/RsqrtOptimizer/clip_by_norm/truediv*
T0*&
_output_shapes
:
Т
Optimizer/clip_by_norm/mul_2MulOptimizer/clip_by_norm/mul_1Optimizer/clip_by_norm/Minimum*
T0*&
_output_shapes
: 
q
Optimizer/clip_by_normIdentityOptimizer/clip_by_norm/mul_2*
T0*&
_output_shapes
: 
Д
Optimizer/clip_by_norm_1/mulMul\Optimizer/gradients/over_options/q_func/convnet/Conv/BiasAdd_grad/tuple/control_dependency_1\Optimizer/gradients/over_options/q_func/convnet/Conv/BiasAdd_grad/tuple/control_dependency_1*
T0*
_output_shapes
: 
h
Optimizer/clip_by_norm_1/ConstConst*
valueB: *
dtype0*
_output_shapes
:
£
Optimizer/clip_by_norm_1/SumSumOptimizer/clip_by_norm_1/mulOptimizer/clip_by_norm_1/Const*
	keep_dims(*
T0*

Tidx0*
_output_shapes
:
j
Optimizer/clip_by_norm_1/RsqrtRsqrtOptimizer/clip_by_norm_1/Sum*
T0*
_output_shapes
:
e
 Optimizer/clip_by_norm_1/mul_1/yConst*
valueB
 *   A*
dtype0*
_output_shapes
: 
 
Optimizer/clip_by_norm_1/mul_1Mul\Optimizer/gradients/over_options/q_func/convnet/Conv/BiasAdd_grad/tuple/control_dependency_1 Optimizer/clip_by_norm_1/mul_1/y*
T0*
_output_shapes
: 
e
 Optimizer/clip_by_norm_1/Const_1Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
g
"Optimizer/clip_by_norm_1/truediv/yConst*
valueB
 *   A*
dtype0*
_output_shapes
: 
Т
 Optimizer/clip_by_norm_1/truedivRealDiv Optimizer/clip_by_norm_1/Const_1"Optimizer/clip_by_norm_1/truediv/y*
T0*
_output_shapes
: 
Т
 Optimizer/clip_by_norm_1/MinimumMinimumOptimizer/clip_by_norm_1/Rsqrt Optimizer/clip_by_norm_1/truediv*
T0*
_output_shapes
:
М
Optimizer/clip_by_norm_1/mul_2MulOptimizer/clip_by_norm_1/mul_1 Optimizer/clip_by_norm_1/Minimum*
T0*
_output_shapes
: 
i
Optimizer/clip_by_norm_1IdentityOptimizer/clip_by_norm_1/mul_2*
T0*
_output_shapes
: 
Ь
Optimizer/clip_by_norm_2/mulMulbOptimizer/gradients/over_options/q_func/convnet/Conv_1/convolution_grad/tuple/control_dependency_1bOptimizer/gradients/over_options/q_func/convnet/Conv_1/convolution_grad/tuple/control_dependency_1*
T0*&
_output_shapes
: @
w
Optimizer/clip_by_norm_2/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:
ѓ
Optimizer/clip_by_norm_2/SumSumOptimizer/clip_by_norm_2/mulOptimizer/clip_by_norm_2/Const*
	keep_dims(*
T0*

Tidx0*&
_output_shapes
:
v
Optimizer/clip_by_norm_2/RsqrtRsqrtOptimizer/clip_by_norm_2/Sum*
T0*&
_output_shapes
:
e
 Optimizer/clip_by_norm_2/mul_1/yConst*
valueB
 *   A*
dtype0*
_output_shapes
: 
№
Optimizer/clip_by_norm_2/mul_1MulbOptimizer/gradients/over_options/q_func/convnet/Conv_1/convolution_grad/tuple/control_dependency_1 Optimizer/clip_by_norm_2/mul_1/y*
T0*&
_output_shapes
: @
e
 Optimizer/clip_by_norm_2/Const_1Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
g
"Optimizer/clip_by_norm_2/truediv/yConst*
valueB
 *   A*
dtype0*
_output_shapes
: 
Т
 Optimizer/clip_by_norm_2/truedivRealDiv Optimizer/clip_by_norm_2/Const_1"Optimizer/clip_by_norm_2/truediv/y*
T0*
_output_shapes
: 
Ю
 Optimizer/clip_by_norm_2/MinimumMinimumOptimizer/clip_by_norm_2/Rsqrt Optimizer/clip_by_norm_2/truediv*
T0*&
_output_shapes
:
Ш
Optimizer/clip_by_norm_2/mul_2MulOptimizer/clip_by_norm_2/mul_1 Optimizer/clip_by_norm_2/Minimum*
T0*&
_output_shapes
: @
u
Optimizer/clip_by_norm_2IdentityOptimizer/clip_by_norm_2/mul_2*
T0*&
_output_shapes
: @
И
Optimizer/clip_by_norm_3/mulMul^Optimizer/gradients/over_options/q_func/convnet/Conv_1/BiasAdd_grad/tuple/control_dependency_1^Optimizer/gradients/over_options/q_func/convnet/Conv_1/BiasAdd_grad/tuple/control_dependency_1*
T0*
_output_shapes
:@
h
Optimizer/clip_by_norm_3/ConstConst*
valueB: *
dtype0*
_output_shapes
:
£
Optimizer/clip_by_norm_3/SumSumOptimizer/clip_by_norm_3/mulOptimizer/clip_by_norm_3/Const*
	keep_dims(*
T0*

Tidx0*
_output_shapes
:
j
Optimizer/clip_by_norm_3/RsqrtRsqrtOptimizer/clip_by_norm_3/Sum*
T0*
_output_shapes
:
e
 Optimizer/clip_by_norm_3/mul_1/yConst*
valueB
 *   A*
dtype0*
_output_shapes
: 
ћ
Optimizer/clip_by_norm_3/mul_1Mul^Optimizer/gradients/over_options/q_func/convnet/Conv_1/BiasAdd_grad/tuple/control_dependency_1 Optimizer/clip_by_norm_3/mul_1/y*
T0*
_output_shapes
:@
e
 Optimizer/clip_by_norm_3/Const_1Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
g
"Optimizer/clip_by_norm_3/truediv/yConst*
valueB
 *   A*
dtype0*
_output_shapes
: 
Т
 Optimizer/clip_by_norm_3/truedivRealDiv Optimizer/clip_by_norm_3/Const_1"Optimizer/clip_by_norm_3/truediv/y*
T0*
_output_shapes
: 
Т
 Optimizer/clip_by_norm_3/MinimumMinimumOptimizer/clip_by_norm_3/Rsqrt Optimizer/clip_by_norm_3/truediv*
T0*
_output_shapes
:
М
Optimizer/clip_by_norm_3/mul_2MulOptimizer/clip_by_norm_3/mul_1 Optimizer/clip_by_norm_3/Minimum*
T0*
_output_shapes
:@
i
Optimizer/clip_by_norm_3IdentityOptimizer/clip_by_norm_3/mul_2*
T0*
_output_shapes
:@
Ь
Optimizer/clip_by_norm_4/mulMulbOptimizer/gradients/over_options/q_func/convnet/Conv_2/convolution_grad/tuple/control_dependency_1bOptimizer/gradients/over_options/q_func/convnet/Conv_2/convolution_grad/tuple/control_dependency_1*
T0*&
_output_shapes
:@@
w
Optimizer/clip_by_norm_4/ConstConst*%
valueB"             *
dtype0*
_output_shapes
:
ѓ
Optimizer/clip_by_norm_4/SumSumOptimizer/clip_by_norm_4/mulOptimizer/clip_by_norm_4/Const*
	keep_dims(*
T0*

Tidx0*&
_output_shapes
:
v
Optimizer/clip_by_norm_4/RsqrtRsqrtOptimizer/clip_by_norm_4/Sum*
T0*&
_output_shapes
:
e
 Optimizer/clip_by_norm_4/mul_1/yConst*
valueB
 *   A*
dtype0*
_output_shapes
: 
№
Optimizer/clip_by_norm_4/mul_1MulbOptimizer/gradients/over_options/q_func/convnet/Conv_2/convolution_grad/tuple/control_dependency_1 Optimizer/clip_by_norm_4/mul_1/y*
T0*&
_output_shapes
:@@
e
 Optimizer/clip_by_norm_4/Const_1Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
g
"Optimizer/clip_by_norm_4/truediv/yConst*
valueB
 *   A*
dtype0*
_output_shapes
: 
Т
 Optimizer/clip_by_norm_4/truedivRealDiv Optimizer/clip_by_norm_4/Const_1"Optimizer/clip_by_norm_4/truediv/y*
T0*
_output_shapes
: 
Ю
 Optimizer/clip_by_norm_4/MinimumMinimumOptimizer/clip_by_norm_4/Rsqrt Optimizer/clip_by_norm_4/truediv*
T0*&
_output_shapes
:
Ш
Optimizer/clip_by_norm_4/mul_2MulOptimizer/clip_by_norm_4/mul_1 Optimizer/clip_by_norm_4/Minimum*
T0*&
_output_shapes
:@@
u
Optimizer/clip_by_norm_4IdentityOptimizer/clip_by_norm_4/mul_2*
T0*&
_output_shapes
:@@
И
Optimizer/clip_by_norm_5/mulMul^Optimizer/gradients/over_options/q_func/convnet/Conv_2/BiasAdd_grad/tuple/control_dependency_1^Optimizer/gradients/over_options/q_func/convnet/Conv_2/BiasAdd_grad/tuple/control_dependency_1*
T0*
_output_shapes
:@
h
Optimizer/clip_by_norm_5/ConstConst*
valueB: *
dtype0*
_output_shapes
:
£
Optimizer/clip_by_norm_5/SumSumOptimizer/clip_by_norm_5/mulOptimizer/clip_by_norm_5/Const*
	keep_dims(*
T0*

Tidx0*
_output_shapes
:
j
Optimizer/clip_by_norm_5/RsqrtRsqrtOptimizer/clip_by_norm_5/Sum*
T0*
_output_shapes
:
e
 Optimizer/clip_by_norm_5/mul_1/yConst*
valueB
 *   A*
dtype0*
_output_shapes
: 
ћ
Optimizer/clip_by_norm_5/mul_1Mul^Optimizer/gradients/over_options/q_func/convnet/Conv_2/BiasAdd_grad/tuple/control_dependency_1 Optimizer/clip_by_norm_5/mul_1/y*
T0*
_output_shapes
:@
e
 Optimizer/clip_by_norm_5/Const_1Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
g
"Optimizer/clip_by_norm_5/truediv/yConst*
valueB
 *   A*
dtype0*
_output_shapes
: 
Т
 Optimizer/clip_by_norm_5/truedivRealDiv Optimizer/clip_by_norm_5/Const_1"Optimizer/clip_by_norm_5/truediv/y*
T0*
_output_shapes
: 
Т
 Optimizer/clip_by_norm_5/MinimumMinimumOptimizer/clip_by_norm_5/Rsqrt Optimizer/clip_by_norm_5/truediv*
T0*
_output_shapes
:
М
Optimizer/clip_by_norm_5/mul_2MulOptimizer/clip_by_norm_5/mul_1 Optimizer/clip_by_norm_5/Minimum*
T0*
_output_shapes
:@
i
Optimizer/clip_by_norm_5IdentityOptimizer/clip_by_norm_5/mul_2*
T0*
_output_shapes
:@
®
Optimizer/clip_by_norm_6/mulMulkOptimizer/gradients/over_options/q_func/action_value/fully_connected/MatMul_grad/tuple/control_dependency_1kOptimizer/gradients/over_options/q_func/action_value/fully_connected/MatMul_grad/tuple/control_dependency_1*
T0* 
_output_shapes
:
АА
o
Optimizer/clip_by_norm_6/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
І
Optimizer/clip_by_norm_6/SumSumOptimizer/clip_by_norm_6/mulOptimizer/clip_by_norm_6/Const*
	keep_dims(*
T0*

Tidx0*
_output_shapes

:
n
Optimizer/clip_by_norm_6/RsqrtRsqrtOptimizer/clip_by_norm_6/Sum*
T0*
_output_shapes

:
e
 Optimizer/clip_by_norm_6/mul_1/yConst*
valueB
 *   A*
dtype0*
_output_shapes
: 
я
Optimizer/clip_by_norm_6/mul_1MulkOptimizer/gradients/over_options/q_func/action_value/fully_connected/MatMul_grad/tuple/control_dependency_1 Optimizer/clip_by_norm_6/mul_1/y*
T0* 
_output_shapes
:
АА
e
 Optimizer/clip_by_norm_6/Const_1Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
g
"Optimizer/clip_by_norm_6/truediv/yConst*
valueB
 *   A*
dtype0*
_output_shapes
: 
Т
 Optimizer/clip_by_norm_6/truedivRealDiv Optimizer/clip_by_norm_6/Const_1"Optimizer/clip_by_norm_6/truediv/y*
T0*
_output_shapes
: 
Ц
 Optimizer/clip_by_norm_6/MinimumMinimumOptimizer/clip_by_norm_6/Rsqrt Optimizer/clip_by_norm_6/truediv*
T0*
_output_shapes

:
Т
Optimizer/clip_by_norm_6/mul_2MulOptimizer/clip_by_norm_6/mul_1 Optimizer/clip_by_norm_6/Minimum*
T0* 
_output_shapes
:
АА
o
Optimizer/clip_by_norm_6IdentityOptimizer/clip_by_norm_6/mul_2*
T0* 
_output_shapes
:
АА
•
Optimizer/clip_by_norm_7/mulMullOptimizer/gradients/over_options/q_func/action_value/fully_connected/BiasAdd_grad/tuple/control_dependency_1lOptimizer/gradients/over_options/q_func/action_value/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
T0*
_output_shapes	
:А
h
Optimizer/clip_by_norm_7/ConstConst*
valueB: *
dtype0*
_output_shapes
:
£
Optimizer/clip_by_norm_7/SumSumOptimizer/clip_by_norm_7/mulOptimizer/clip_by_norm_7/Const*
	keep_dims(*
T0*

Tidx0*
_output_shapes
:
j
Optimizer/clip_by_norm_7/RsqrtRsqrtOptimizer/clip_by_norm_7/Sum*
T0*
_output_shapes
:
e
 Optimizer/clip_by_norm_7/mul_1/yConst*
valueB
 *   A*
dtype0*
_output_shapes
: 
џ
Optimizer/clip_by_norm_7/mul_1MullOptimizer/gradients/over_options/q_func/action_value/fully_connected/BiasAdd_grad/tuple/control_dependency_1 Optimizer/clip_by_norm_7/mul_1/y*
T0*
_output_shapes	
:А
e
 Optimizer/clip_by_norm_7/Const_1Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
g
"Optimizer/clip_by_norm_7/truediv/yConst*
valueB
 *   A*
dtype0*
_output_shapes
: 
Т
 Optimizer/clip_by_norm_7/truedivRealDiv Optimizer/clip_by_norm_7/Const_1"Optimizer/clip_by_norm_7/truediv/y*
T0*
_output_shapes
: 
Т
 Optimizer/clip_by_norm_7/MinimumMinimumOptimizer/clip_by_norm_7/Rsqrt Optimizer/clip_by_norm_7/truediv*
T0*
_output_shapes
:
Н
Optimizer/clip_by_norm_7/mul_2MulOptimizer/clip_by_norm_7/mul_1 Optimizer/clip_by_norm_7/Minimum*
T0*
_output_shapes	
:А
j
Optimizer/clip_by_norm_7IdentityOptimizer/clip_by_norm_7/mul_2*
T0*
_output_shapes	
:А
Ђ
Optimizer/clip_by_norm_8/mulMulmOptimizer/gradients/over_options/q_func/action_value/fully_connected_1/MatMul_grad/tuple/control_dependency_1mOptimizer/gradients/over_options/q_func/action_value/fully_connected_1/MatMul_grad/tuple/control_dependency_1*
T0*
_output_shapes
:	А
o
Optimizer/clip_by_norm_8/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
І
Optimizer/clip_by_norm_8/SumSumOptimizer/clip_by_norm_8/mulOptimizer/clip_by_norm_8/Const*
	keep_dims(*
T0*

Tidx0*
_output_shapes

:
n
Optimizer/clip_by_norm_8/RsqrtRsqrtOptimizer/clip_by_norm_8/Sum*
T0*
_output_shapes

:
e
 Optimizer/clip_by_norm_8/mul_1/yConst*
valueB
 *   A*
dtype0*
_output_shapes
: 
а
Optimizer/clip_by_norm_8/mul_1MulmOptimizer/gradients/over_options/q_func/action_value/fully_connected_1/MatMul_grad/tuple/control_dependency_1 Optimizer/clip_by_norm_8/mul_1/y*
T0*
_output_shapes
:	А
e
 Optimizer/clip_by_norm_8/Const_1Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
g
"Optimizer/clip_by_norm_8/truediv/yConst*
valueB
 *   A*
dtype0*
_output_shapes
: 
Т
 Optimizer/clip_by_norm_8/truedivRealDiv Optimizer/clip_by_norm_8/Const_1"Optimizer/clip_by_norm_8/truediv/y*
T0*
_output_shapes
: 
Ц
 Optimizer/clip_by_norm_8/MinimumMinimumOptimizer/clip_by_norm_8/Rsqrt Optimizer/clip_by_norm_8/truediv*
T0*
_output_shapes

:
С
Optimizer/clip_by_norm_8/mul_2MulOptimizer/clip_by_norm_8/mul_1 Optimizer/clip_by_norm_8/Minimum*
T0*
_output_shapes
:	А
n
Optimizer/clip_by_norm_8IdentityOptimizer/clip_by_norm_8/mul_2*
T0*
_output_shapes
:	А
®
Optimizer/clip_by_norm_9/mulMulnOptimizer/gradients/over_options/q_func/action_value/fully_connected_1/BiasAdd_grad/tuple/control_dependency_1nOptimizer/gradients/over_options/q_func/action_value/fully_connected_1/BiasAdd_grad/tuple/control_dependency_1*
T0*
_output_shapes
:
h
Optimizer/clip_by_norm_9/ConstConst*
valueB: *
dtype0*
_output_shapes
:
£
Optimizer/clip_by_norm_9/SumSumOptimizer/clip_by_norm_9/mulOptimizer/clip_by_norm_9/Const*
	keep_dims(*
T0*

Tidx0*
_output_shapes
:
j
Optimizer/clip_by_norm_9/RsqrtRsqrtOptimizer/clip_by_norm_9/Sum*
T0*
_output_shapes
:
e
 Optimizer/clip_by_norm_9/mul_1/yConst*
valueB
 *   A*
dtype0*
_output_shapes
: 
№
Optimizer/clip_by_norm_9/mul_1MulnOptimizer/gradients/over_options/q_func/action_value/fully_connected_1/BiasAdd_grad/tuple/control_dependency_1 Optimizer/clip_by_norm_9/mul_1/y*
T0*
_output_shapes
:
e
 Optimizer/clip_by_norm_9/Const_1Const*
valueB
 *  А?*
dtype0*
_output_shapes
: 
g
"Optimizer/clip_by_norm_9/truediv/yConst*
valueB
 *   A*
dtype0*
_output_shapes
: 
Т
 Optimizer/clip_by_norm_9/truedivRealDiv Optimizer/clip_by_norm_9/Const_1"Optimizer/clip_by_norm_9/truediv/y*
T0*
_output_shapes
: 
Т
 Optimizer/clip_by_norm_9/MinimumMinimumOptimizer/clip_by_norm_9/Rsqrt Optimizer/clip_by_norm_9/truediv*
T0*
_output_shapes
:
М
Optimizer/clip_by_norm_9/mul_2MulOptimizer/clip_by_norm_9/mul_1 Optimizer/clip_by_norm_9/Minimum*
T0*
_output_shapes
:
i
Optimizer/clip_by_norm_9IdentityOptimizer/clip_by_norm_9/mul_2*
T0*
_output_shapes
:
•
#Optimizer/beta1_power/initial_valueConst*
valueB
 *fff?*
dtype0*;
_class1
/-loc:@over_options/q_func/convnet/Conv/weights*
_output_shapes
: 
ґ
Optimizer/beta1_power
VariableV2*
shape: *
dtype0*
	container *
shared_name *;
_class1
/-loc:@over_options/q_func/convnet/Conv/weights*
_output_shapes
: 
й
Optimizer/beta1_power/AssignAssignOptimizer/beta1_power#Optimizer/beta1_power/initial_value*
T0*
validate_shape(*
use_locking(*;
_class1
/-loc:@over_options/q_func/convnet/Conv/weights*
_output_shapes
: 
Ы
Optimizer/beta1_power/readIdentityOptimizer/beta1_power*
T0*;
_class1
/-loc:@over_options/q_func/convnet/Conv/weights*
_output_shapes
: 
•
#Optimizer/beta2_power/initial_valueConst*
valueB
 *wЊ?*
dtype0*;
_class1
/-loc:@over_options/q_func/convnet/Conv/weights*
_output_shapes
: 
ґ
Optimizer/beta2_power
VariableV2*
shape: *
dtype0*
	container *
shared_name *;
_class1
/-loc:@over_options/q_func/convnet/Conv/weights*
_output_shapes
: 
й
Optimizer/beta2_power/AssignAssignOptimizer/beta2_power#Optimizer/beta2_power/initial_value*
T0*
validate_shape(*
use_locking(*;
_class1
/-loc:@over_options/q_func/convnet/Conv/weights*
_output_shapes
: 
Ы
Optimizer/beta2_power/readIdentityOptimizer/beta2_power*
T0*;
_class1
/-loc:@over_options/q_func/convnet/Conv/weights*
_output_shapes
: 
л
IOptimizer/over_options/q_func/convnet/Conv/weights/Adam/Initializer/ConstConst*%
valueB *    *
dtype0*;
_class1
/-loc:@over_options/q_func/convnet/Conv/weights*&
_output_shapes
: 
ш
7Optimizer/over_options/q_func/convnet/Conv/weights/Adam
VariableV2*
shape: *
dtype0*
	container *
shared_name *;
_class1
/-loc:@over_options/q_func/convnet/Conv/weights*&
_output_shapes
: 
г
>Optimizer/over_options/q_func/convnet/Conv/weights/Adam/AssignAssign7Optimizer/over_options/q_func/convnet/Conv/weights/AdamIOptimizer/over_options/q_func/convnet/Conv/weights/Adam/Initializer/Const*
T0*
validate_shape(*
use_locking(*;
_class1
/-loc:@over_options/q_func/convnet/Conv/weights*&
_output_shapes
: 
п
<Optimizer/over_options/q_func/convnet/Conv/weights/Adam/readIdentity7Optimizer/over_options/q_func/convnet/Conv/weights/Adam*
T0*;
_class1
/-loc:@over_options/q_func/convnet/Conv/weights*&
_output_shapes
: 
н
KOptimizer/over_options/q_func/convnet/Conv/weights/Adam_1/Initializer/ConstConst*%
valueB *    *
dtype0*;
_class1
/-loc:@over_options/q_func/convnet/Conv/weights*&
_output_shapes
: 
ъ
9Optimizer/over_options/q_func/convnet/Conv/weights/Adam_1
VariableV2*
shape: *
dtype0*
	container *
shared_name *;
_class1
/-loc:@over_options/q_func/convnet/Conv/weights*&
_output_shapes
: 
й
@Optimizer/over_options/q_func/convnet/Conv/weights/Adam_1/AssignAssign9Optimizer/over_options/q_func/convnet/Conv/weights/Adam_1KOptimizer/over_options/q_func/convnet/Conv/weights/Adam_1/Initializer/Const*
T0*
validate_shape(*
use_locking(*;
_class1
/-loc:@over_options/q_func/convnet/Conv/weights*&
_output_shapes
: 
у
>Optimizer/over_options/q_func/convnet/Conv/weights/Adam_1/readIdentity9Optimizer/over_options/q_func/convnet/Conv/weights/Adam_1*
T0*;
_class1
/-loc:@over_options/q_func/convnet/Conv/weights*&
_output_shapes
: 
—
HOptimizer/over_options/q_func/convnet/Conv/biases/Adam/Initializer/ConstConst*
valueB *    *
dtype0*:
_class0
.,loc:@over_options/q_func/convnet/Conv/biases*
_output_shapes
: 
ё
6Optimizer/over_options/q_func/convnet/Conv/biases/Adam
VariableV2*
shape: *
dtype0*
	container *
shared_name *:
_class0
.,loc:@over_options/q_func/convnet/Conv/biases*
_output_shapes
: 
”
=Optimizer/over_options/q_func/convnet/Conv/biases/Adam/AssignAssign6Optimizer/over_options/q_func/convnet/Conv/biases/AdamHOptimizer/over_options/q_func/convnet/Conv/biases/Adam/Initializer/Const*
T0*
validate_shape(*
use_locking(*:
_class0
.,loc:@over_options/q_func/convnet/Conv/biases*
_output_shapes
: 
а
;Optimizer/over_options/q_func/convnet/Conv/biases/Adam/readIdentity6Optimizer/over_options/q_func/convnet/Conv/biases/Adam*
T0*:
_class0
.,loc:@over_options/q_func/convnet/Conv/biases*
_output_shapes
: 
”
JOptimizer/over_options/q_func/convnet/Conv/biases/Adam_1/Initializer/ConstConst*
valueB *    *
dtype0*:
_class0
.,loc:@over_options/q_func/convnet/Conv/biases*
_output_shapes
: 
а
8Optimizer/over_options/q_func/convnet/Conv/biases/Adam_1
VariableV2*
shape: *
dtype0*
	container *
shared_name *:
_class0
.,loc:@over_options/q_func/convnet/Conv/biases*
_output_shapes
: 
ў
?Optimizer/over_options/q_func/convnet/Conv/biases/Adam_1/AssignAssign8Optimizer/over_options/q_func/convnet/Conv/biases/Adam_1JOptimizer/over_options/q_func/convnet/Conv/biases/Adam_1/Initializer/Const*
T0*
validate_shape(*
use_locking(*:
_class0
.,loc:@over_options/q_func/convnet/Conv/biases*
_output_shapes
: 
д
=Optimizer/over_options/q_func/convnet/Conv/biases/Adam_1/readIdentity8Optimizer/over_options/q_func/convnet/Conv/biases/Adam_1*
T0*:
_class0
.,loc:@over_options/q_func/convnet/Conv/biases*
_output_shapes
: 
п
KOptimizer/over_options/q_func/convnet/Conv_1/weights/Adam/Initializer/ConstConst*%
valueB @*    *
dtype0*=
_class3
1/loc:@over_options/q_func/convnet/Conv_1/weights*&
_output_shapes
: @
ь
9Optimizer/over_options/q_func/convnet/Conv_1/weights/Adam
VariableV2*
shape: @*
dtype0*
	container *
shared_name *=
_class3
1/loc:@over_options/q_func/convnet/Conv_1/weights*&
_output_shapes
: @
л
@Optimizer/over_options/q_func/convnet/Conv_1/weights/Adam/AssignAssign9Optimizer/over_options/q_func/convnet/Conv_1/weights/AdamKOptimizer/over_options/q_func/convnet/Conv_1/weights/Adam/Initializer/Const*
T0*
validate_shape(*
use_locking(*=
_class3
1/loc:@over_options/q_func/convnet/Conv_1/weights*&
_output_shapes
: @
х
>Optimizer/over_options/q_func/convnet/Conv_1/weights/Adam/readIdentity9Optimizer/over_options/q_func/convnet/Conv_1/weights/Adam*
T0*=
_class3
1/loc:@over_options/q_func/convnet/Conv_1/weights*&
_output_shapes
: @
с
MOptimizer/over_options/q_func/convnet/Conv_1/weights/Adam_1/Initializer/ConstConst*%
valueB @*    *
dtype0*=
_class3
1/loc:@over_options/q_func/convnet/Conv_1/weights*&
_output_shapes
: @
ю
;Optimizer/over_options/q_func/convnet/Conv_1/weights/Adam_1
VariableV2*
shape: @*
dtype0*
	container *
shared_name *=
_class3
1/loc:@over_options/q_func/convnet/Conv_1/weights*&
_output_shapes
: @
с
BOptimizer/over_options/q_func/convnet/Conv_1/weights/Adam_1/AssignAssign;Optimizer/over_options/q_func/convnet/Conv_1/weights/Adam_1MOptimizer/over_options/q_func/convnet/Conv_1/weights/Adam_1/Initializer/Const*
T0*
validate_shape(*
use_locking(*=
_class3
1/loc:@over_options/q_func/convnet/Conv_1/weights*&
_output_shapes
: @
щ
@Optimizer/over_options/q_func/convnet/Conv_1/weights/Adam_1/readIdentity;Optimizer/over_options/q_func/convnet/Conv_1/weights/Adam_1*
T0*=
_class3
1/loc:@over_options/q_func/convnet/Conv_1/weights*&
_output_shapes
: @
’
JOptimizer/over_options/q_func/convnet/Conv_1/biases/Adam/Initializer/ConstConst*
valueB@*    *
dtype0*<
_class2
0.loc:@over_options/q_func/convnet/Conv_1/biases*
_output_shapes
:@
в
8Optimizer/over_options/q_func/convnet/Conv_1/biases/Adam
VariableV2*
shape:@*
dtype0*
	container *
shared_name *<
_class2
0.loc:@over_options/q_func/convnet/Conv_1/biases*
_output_shapes
:@
џ
?Optimizer/over_options/q_func/convnet/Conv_1/biases/Adam/AssignAssign8Optimizer/over_options/q_func/convnet/Conv_1/biases/AdamJOptimizer/over_options/q_func/convnet/Conv_1/biases/Adam/Initializer/Const*
T0*
validate_shape(*
use_locking(*<
_class2
0.loc:@over_options/q_func/convnet/Conv_1/biases*
_output_shapes
:@
ж
=Optimizer/over_options/q_func/convnet/Conv_1/biases/Adam/readIdentity8Optimizer/over_options/q_func/convnet/Conv_1/biases/Adam*
T0*<
_class2
0.loc:@over_options/q_func/convnet/Conv_1/biases*
_output_shapes
:@
„
LOptimizer/over_options/q_func/convnet/Conv_1/biases/Adam_1/Initializer/ConstConst*
valueB@*    *
dtype0*<
_class2
0.loc:@over_options/q_func/convnet/Conv_1/biases*
_output_shapes
:@
д
:Optimizer/over_options/q_func/convnet/Conv_1/biases/Adam_1
VariableV2*
shape:@*
dtype0*
	container *
shared_name *<
_class2
0.loc:@over_options/q_func/convnet/Conv_1/biases*
_output_shapes
:@
б
AOptimizer/over_options/q_func/convnet/Conv_1/biases/Adam_1/AssignAssign:Optimizer/over_options/q_func/convnet/Conv_1/biases/Adam_1LOptimizer/over_options/q_func/convnet/Conv_1/biases/Adam_1/Initializer/Const*
T0*
validate_shape(*
use_locking(*<
_class2
0.loc:@over_options/q_func/convnet/Conv_1/biases*
_output_shapes
:@
к
?Optimizer/over_options/q_func/convnet/Conv_1/biases/Adam_1/readIdentity:Optimizer/over_options/q_func/convnet/Conv_1/biases/Adam_1*
T0*<
_class2
0.loc:@over_options/q_func/convnet/Conv_1/biases*
_output_shapes
:@
п
KOptimizer/over_options/q_func/convnet/Conv_2/weights/Adam/Initializer/ConstConst*%
valueB@@*    *
dtype0*=
_class3
1/loc:@over_options/q_func/convnet/Conv_2/weights*&
_output_shapes
:@@
ь
9Optimizer/over_options/q_func/convnet/Conv_2/weights/Adam
VariableV2*
shape:@@*
dtype0*
	container *
shared_name *=
_class3
1/loc:@over_options/q_func/convnet/Conv_2/weights*&
_output_shapes
:@@
л
@Optimizer/over_options/q_func/convnet/Conv_2/weights/Adam/AssignAssign9Optimizer/over_options/q_func/convnet/Conv_2/weights/AdamKOptimizer/over_options/q_func/convnet/Conv_2/weights/Adam/Initializer/Const*
T0*
validate_shape(*
use_locking(*=
_class3
1/loc:@over_options/q_func/convnet/Conv_2/weights*&
_output_shapes
:@@
х
>Optimizer/over_options/q_func/convnet/Conv_2/weights/Adam/readIdentity9Optimizer/over_options/q_func/convnet/Conv_2/weights/Adam*
T0*=
_class3
1/loc:@over_options/q_func/convnet/Conv_2/weights*&
_output_shapes
:@@
с
MOptimizer/over_options/q_func/convnet/Conv_2/weights/Adam_1/Initializer/ConstConst*%
valueB@@*    *
dtype0*=
_class3
1/loc:@over_options/q_func/convnet/Conv_2/weights*&
_output_shapes
:@@
ю
;Optimizer/over_options/q_func/convnet/Conv_2/weights/Adam_1
VariableV2*
shape:@@*
dtype0*
	container *
shared_name *=
_class3
1/loc:@over_options/q_func/convnet/Conv_2/weights*&
_output_shapes
:@@
с
BOptimizer/over_options/q_func/convnet/Conv_2/weights/Adam_1/AssignAssign;Optimizer/over_options/q_func/convnet/Conv_2/weights/Adam_1MOptimizer/over_options/q_func/convnet/Conv_2/weights/Adam_1/Initializer/Const*
T0*
validate_shape(*
use_locking(*=
_class3
1/loc:@over_options/q_func/convnet/Conv_2/weights*&
_output_shapes
:@@
щ
@Optimizer/over_options/q_func/convnet/Conv_2/weights/Adam_1/readIdentity;Optimizer/over_options/q_func/convnet/Conv_2/weights/Adam_1*
T0*=
_class3
1/loc:@over_options/q_func/convnet/Conv_2/weights*&
_output_shapes
:@@
’
JOptimizer/over_options/q_func/convnet/Conv_2/biases/Adam/Initializer/ConstConst*
valueB@*    *
dtype0*<
_class2
0.loc:@over_options/q_func/convnet/Conv_2/biases*
_output_shapes
:@
в
8Optimizer/over_options/q_func/convnet/Conv_2/biases/Adam
VariableV2*
shape:@*
dtype0*
	container *
shared_name *<
_class2
0.loc:@over_options/q_func/convnet/Conv_2/biases*
_output_shapes
:@
џ
?Optimizer/over_options/q_func/convnet/Conv_2/biases/Adam/AssignAssign8Optimizer/over_options/q_func/convnet/Conv_2/biases/AdamJOptimizer/over_options/q_func/convnet/Conv_2/biases/Adam/Initializer/Const*
T0*
validate_shape(*
use_locking(*<
_class2
0.loc:@over_options/q_func/convnet/Conv_2/biases*
_output_shapes
:@
ж
=Optimizer/over_options/q_func/convnet/Conv_2/biases/Adam/readIdentity8Optimizer/over_options/q_func/convnet/Conv_2/biases/Adam*
T0*<
_class2
0.loc:@over_options/q_func/convnet/Conv_2/biases*
_output_shapes
:@
„
LOptimizer/over_options/q_func/convnet/Conv_2/biases/Adam_1/Initializer/ConstConst*
valueB@*    *
dtype0*<
_class2
0.loc:@over_options/q_func/convnet/Conv_2/biases*
_output_shapes
:@
д
:Optimizer/over_options/q_func/convnet/Conv_2/biases/Adam_1
VariableV2*
shape:@*
dtype0*
	container *
shared_name *<
_class2
0.loc:@over_options/q_func/convnet/Conv_2/biases*
_output_shapes
:@
б
AOptimizer/over_options/q_func/convnet/Conv_2/biases/Adam_1/AssignAssign:Optimizer/over_options/q_func/convnet/Conv_2/biases/Adam_1LOptimizer/over_options/q_func/convnet/Conv_2/biases/Adam_1/Initializer/Const*
T0*
validate_shape(*
use_locking(*<
_class2
0.loc:@over_options/q_func/convnet/Conv_2/biases*
_output_shapes
:@
к
?Optimizer/over_options/q_func/convnet/Conv_2/biases/Adam_1/readIdentity:Optimizer/over_options/q_func/convnet/Conv_2/biases/Adam_1*
T0*<
_class2
0.loc:@over_options/q_func/convnet/Conv_2/biases*
_output_shapes
:@
€
YOptimizer/over_options/q_func/action_value/fully_connected/weights/Adam/Initializer/ConstConst*
valueB
АА*    *
dtype0*K
_classA
?=loc:@over_options/q_func/action_value/fully_connected/weights* 
_output_shapes
:
АА
М
GOptimizer/over_options/q_func/action_value/fully_connected/weights/Adam
VariableV2*
shape:
АА*
dtype0*
	container *
shared_name *K
_classA
?=loc:@over_options/q_func/action_value/fully_connected/weights* 
_output_shapes
:
АА
Э
NOptimizer/over_options/q_func/action_value/fully_connected/weights/Adam/AssignAssignGOptimizer/over_options/q_func/action_value/fully_connected/weights/AdamYOptimizer/over_options/q_func/action_value/fully_connected/weights/Adam/Initializer/Const*
T0*
validate_shape(*
use_locking(*K
_classA
?=loc:@over_options/q_func/action_value/fully_connected/weights* 
_output_shapes
:
АА
Щ
LOptimizer/over_options/q_func/action_value/fully_connected/weights/Adam/readIdentityGOptimizer/over_options/q_func/action_value/fully_connected/weights/Adam*
T0*K
_classA
?=loc:@over_options/q_func/action_value/fully_connected/weights* 
_output_shapes
:
АА
Б
[Optimizer/over_options/q_func/action_value/fully_connected/weights/Adam_1/Initializer/ConstConst*
valueB
АА*    *
dtype0*K
_classA
?=loc:@over_options/q_func/action_value/fully_connected/weights* 
_output_shapes
:
АА
О
IOptimizer/over_options/q_func/action_value/fully_connected/weights/Adam_1
VariableV2*
shape:
АА*
dtype0*
	container *
shared_name *K
_classA
?=loc:@over_options/q_func/action_value/fully_connected/weights* 
_output_shapes
:
АА
£
POptimizer/over_options/q_func/action_value/fully_connected/weights/Adam_1/AssignAssignIOptimizer/over_options/q_func/action_value/fully_connected/weights/Adam_1[Optimizer/over_options/q_func/action_value/fully_connected/weights/Adam_1/Initializer/Const*
T0*
validate_shape(*
use_locking(*K
_classA
?=loc:@over_options/q_func/action_value/fully_connected/weights* 
_output_shapes
:
АА
Э
NOptimizer/over_options/q_func/action_value/fully_connected/weights/Adam_1/readIdentityIOptimizer/over_options/q_func/action_value/fully_connected/weights/Adam_1*
T0*K
_classA
?=loc:@over_options/q_func/action_value/fully_connected/weights* 
_output_shapes
:
АА
у
XOptimizer/over_options/q_func/action_value/fully_connected/biases/Adam/Initializer/ConstConst*
valueBА*    *
dtype0*J
_class@
><loc:@over_options/q_func/action_value/fully_connected/biases*
_output_shapes	
:А
А
FOptimizer/over_options/q_func/action_value/fully_connected/biases/Adam
VariableV2*
shape:А*
dtype0*
	container *
shared_name *J
_class@
><loc:@over_options/q_func/action_value/fully_connected/biases*
_output_shapes	
:А
Ф
MOptimizer/over_options/q_func/action_value/fully_connected/biases/Adam/AssignAssignFOptimizer/over_options/q_func/action_value/fully_connected/biases/AdamXOptimizer/over_options/q_func/action_value/fully_connected/biases/Adam/Initializer/Const*
T0*
validate_shape(*
use_locking(*J
_class@
><loc:@over_options/q_func/action_value/fully_connected/biases*
_output_shapes	
:А
С
KOptimizer/over_options/q_func/action_value/fully_connected/biases/Adam/readIdentityFOptimizer/over_options/q_func/action_value/fully_connected/biases/Adam*
T0*J
_class@
><loc:@over_options/q_func/action_value/fully_connected/biases*
_output_shapes	
:А
х
ZOptimizer/over_options/q_func/action_value/fully_connected/biases/Adam_1/Initializer/ConstConst*
valueBА*    *
dtype0*J
_class@
><loc:@over_options/q_func/action_value/fully_connected/biases*
_output_shapes	
:А
В
HOptimizer/over_options/q_func/action_value/fully_connected/biases/Adam_1
VariableV2*
shape:А*
dtype0*
	container *
shared_name *J
_class@
><loc:@over_options/q_func/action_value/fully_connected/biases*
_output_shapes	
:А
Ъ
OOptimizer/over_options/q_func/action_value/fully_connected/biases/Adam_1/AssignAssignHOptimizer/over_options/q_func/action_value/fully_connected/biases/Adam_1ZOptimizer/over_options/q_func/action_value/fully_connected/biases/Adam_1/Initializer/Const*
T0*
validate_shape(*
use_locking(*J
_class@
><loc:@over_options/q_func/action_value/fully_connected/biases*
_output_shapes	
:А
Х
MOptimizer/over_options/q_func/action_value/fully_connected/biases/Adam_1/readIdentityHOptimizer/over_options/q_func/action_value/fully_connected/biases/Adam_1*
T0*J
_class@
><loc:@over_options/q_func/action_value/fully_connected/biases*
_output_shapes	
:А
Б
[Optimizer/over_options/q_func/action_value/fully_connected_1/weights/Adam/Initializer/ConstConst*
valueB	А*    *
dtype0*M
_classC
A?loc:@over_options/q_func/action_value/fully_connected_1/weights*
_output_shapes
:	А
О
IOptimizer/over_options/q_func/action_value/fully_connected_1/weights/Adam
VariableV2*
shape:	А*
dtype0*
	container *
shared_name *M
_classC
A?loc:@over_options/q_func/action_value/fully_connected_1/weights*
_output_shapes
:	А
§
POptimizer/over_options/q_func/action_value/fully_connected_1/weights/Adam/AssignAssignIOptimizer/over_options/q_func/action_value/fully_connected_1/weights/Adam[Optimizer/over_options/q_func/action_value/fully_connected_1/weights/Adam/Initializer/Const*
T0*
validate_shape(*
use_locking(*M
_classC
A?loc:@over_options/q_func/action_value/fully_connected_1/weights*
_output_shapes
:	А
Ю
NOptimizer/over_options/q_func/action_value/fully_connected_1/weights/Adam/readIdentityIOptimizer/over_options/q_func/action_value/fully_connected_1/weights/Adam*
T0*M
_classC
A?loc:@over_options/q_func/action_value/fully_connected_1/weights*
_output_shapes
:	А
Г
]Optimizer/over_options/q_func/action_value/fully_connected_1/weights/Adam_1/Initializer/ConstConst*
valueB	А*    *
dtype0*M
_classC
A?loc:@over_options/q_func/action_value/fully_connected_1/weights*
_output_shapes
:	А
Р
KOptimizer/over_options/q_func/action_value/fully_connected_1/weights/Adam_1
VariableV2*
shape:	А*
dtype0*
	container *
shared_name *M
_classC
A?loc:@over_options/q_func/action_value/fully_connected_1/weights*
_output_shapes
:	А
™
ROptimizer/over_options/q_func/action_value/fully_connected_1/weights/Adam_1/AssignAssignKOptimizer/over_options/q_func/action_value/fully_connected_1/weights/Adam_1]Optimizer/over_options/q_func/action_value/fully_connected_1/weights/Adam_1/Initializer/Const*
T0*
validate_shape(*
use_locking(*M
_classC
A?loc:@over_options/q_func/action_value/fully_connected_1/weights*
_output_shapes
:	А
Ґ
POptimizer/over_options/q_func/action_value/fully_connected_1/weights/Adam_1/readIdentityKOptimizer/over_options/q_func/action_value/fully_connected_1/weights/Adam_1*
T0*M
_classC
A?loc:@over_options/q_func/action_value/fully_connected_1/weights*
_output_shapes
:	А
х
ZOptimizer/over_options/q_func/action_value/fully_connected_1/biases/Adam/Initializer/ConstConst*
valueB*    *
dtype0*L
_classB
@>loc:@over_options/q_func/action_value/fully_connected_1/biases*
_output_shapes
:
В
HOptimizer/over_options/q_func/action_value/fully_connected_1/biases/Adam
VariableV2*
shape:*
dtype0*
	container *
shared_name *L
_classB
@>loc:@over_options/q_func/action_value/fully_connected_1/biases*
_output_shapes
:
Ы
OOptimizer/over_options/q_func/action_value/fully_connected_1/biases/Adam/AssignAssignHOptimizer/over_options/q_func/action_value/fully_connected_1/biases/AdamZOptimizer/over_options/q_func/action_value/fully_connected_1/biases/Adam/Initializer/Const*
T0*
validate_shape(*
use_locking(*L
_classB
@>loc:@over_options/q_func/action_value/fully_connected_1/biases*
_output_shapes
:
Ц
MOptimizer/over_options/q_func/action_value/fully_connected_1/biases/Adam/readIdentityHOptimizer/over_options/q_func/action_value/fully_connected_1/biases/Adam*
T0*L
_classB
@>loc:@over_options/q_func/action_value/fully_connected_1/biases*
_output_shapes
:
ч
\Optimizer/over_options/q_func/action_value/fully_connected_1/biases/Adam_1/Initializer/ConstConst*
valueB*    *
dtype0*L
_classB
@>loc:@over_options/q_func/action_value/fully_connected_1/biases*
_output_shapes
:
Д
JOptimizer/over_options/q_func/action_value/fully_connected_1/biases/Adam_1
VariableV2*
shape:*
dtype0*
	container *
shared_name *L
_classB
@>loc:@over_options/q_func/action_value/fully_connected_1/biases*
_output_shapes
:
°
QOptimizer/over_options/q_func/action_value/fully_connected_1/biases/Adam_1/AssignAssignJOptimizer/over_options/q_func/action_value/fully_connected_1/biases/Adam_1\Optimizer/over_options/q_func/action_value/fully_connected_1/biases/Adam_1/Initializer/Const*
T0*
validate_shape(*
use_locking(*L
_classB
@>loc:@over_options/q_func/action_value/fully_connected_1/biases*
_output_shapes
:
Ъ
OOptimizer/over_options/q_func/action_value/fully_connected_1/biases/Adam_1/readIdentityJOptimizer/over_options/q_func/action_value/fully_connected_1/biases/Adam_1*
T0*L
_classB
@>loc:@over_options/q_func/action_value/fully_connected_1/biases*
_output_shapes
:
Y
Optimizer/Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
Y
Optimizer/Adam/beta2Const*
valueB
 *wЊ?*
dtype0*
_output_shapes
: 
[
Optimizer/Adam/epsilonConst*
valueB
 *Ј—8*
dtype0*
_output_shapes
: 
Ч
HOptimizer/Adam/update_over_options/q_func/convnet/Conv/weights/ApplyAdam	ApplyAdam(over_options/q_func/convnet/Conv/weights7Optimizer/over_options/q_func/convnet/Conv/weights/Adam9Optimizer/over_options/q_func/convnet/Conv/weights/Adam_1Optimizer/beta1_power/readOptimizer/beta2_power/readlearning_rateOptimizer/Adam/beta1Optimizer/Adam/beta2Optimizer/Adam/epsilonOptimizer/clip_by_norm*
T0*
use_locking( *;
_class1
/-loc:@over_options/q_func/convnet/Conv/weights*&
_output_shapes
: 
И
GOptimizer/Adam/update_over_options/q_func/convnet/Conv/biases/ApplyAdam	ApplyAdam'over_options/q_func/convnet/Conv/biases6Optimizer/over_options/q_func/convnet/Conv/biases/Adam8Optimizer/over_options/q_func/convnet/Conv/biases/Adam_1Optimizer/beta1_power/readOptimizer/beta2_power/readlearning_rateOptimizer/Adam/beta1Optimizer/Adam/beta2Optimizer/Adam/epsilonOptimizer/clip_by_norm_1*
T0*
use_locking( *:
_class0
.,loc:@over_options/q_func/convnet/Conv/biases*
_output_shapes
: 
£
JOptimizer/Adam/update_over_options/q_func/convnet/Conv_1/weights/ApplyAdam	ApplyAdam*over_options/q_func/convnet/Conv_1/weights9Optimizer/over_options/q_func/convnet/Conv_1/weights/Adam;Optimizer/over_options/q_func/convnet/Conv_1/weights/Adam_1Optimizer/beta1_power/readOptimizer/beta2_power/readlearning_rateOptimizer/Adam/beta1Optimizer/Adam/beta2Optimizer/Adam/epsilonOptimizer/clip_by_norm_2*
T0*
use_locking( *=
_class3
1/loc:@over_options/q_func/convnet/Conv_1/weights*&
_output_shapes
: @
Т
IOptimizer/Adam/update_over_options/q_func/convnet/Conv_1/biases/ApplyAdam	ApplyAdam)over_options/q_func/convnet/Conv_1/biases8Optimizer/over_options/q_func/convnet/Conv_1/biases/Adam:Optimizer/over_options/q_func/convnet/Conv_1/biases/Adam_1Optimizer/beta1_power/readOptimizer/beta2_power/readlearning_rateOptimizer/Adam/beta1Optimizer/Adam/beta2Optimizer/Adam/epsilonOptimizer/clip_by_norm_3*
T0*
use_locking( *<
_class2
0.loc:@over_options/q_func/convnet/Conv_1/biases*
_output_shapes
:@
£
JOptimizer/Adam/update_over_options/q_func/convnet/Conv_2/weights/ApplyAdam	ApplyAdam*over_options/q_func/convnet/Conv_2/weights9Optimizer/over_options/q_func/convnet/Conv_2/weights/Adam;Optimizer/over_options/q_func/convnet/Conv_2/weights/Adam_1Optimizer/beta1_power/readOptimizer/beta2_power/readlearning_rateOptimizer/Adam/beta1Optimizer/Adam/beta2Optimizer/Adam/epsilonOptimizer/clip_by_norm_4*
T0*
use_locking( *=
_class3
1/loc:@over_options/q_func/convnet/Conv_2/weights*&
_output_shapes
:@@
Т
IOptimizer/Adam/update_over_options/q_func/convnet/Conv_2/biases/ApplyAdam	ApplyAdam)over_options/q_func/convnet/Conv_2/biases8Optimizer/over_options/q_func/convnet/Conv_2/biases/Adam:Optimizer/over_options/q_func/convnet/Conv_2/biases/Adam_1Optimizer/beta1_power/readOptimizer/beta2_power/readlearning_rateOptimizer/Adam/beta1Optimizer/Adam/beta2Optimizer/Adam/epsilonOptimizer/clip_by_norm_5*
T0*
use_locking( *<
_class2
0.loc:@over_options/q_func/convnet/Conv_2/biases*
_output_shapes
:@
г
XOptimizer/Adam/update_over_options/q_func/action_value/fully_connected/weights/ApplyAdam	ApplyAdam8over_options/q_func/action_value/fully_connected/weightsGOptimizer/over_options/q_func/action_value/fully_connected/weights/AdamIOptimizer/over_options/q_func/action_value/fully_connected/weights/Adam_1Optimizer/beta1_power/readOptimizer/beta2_power/readlearning_rateOptimizer/Adam/beta1Optimizer/Adam/beta2Optimizer/Adam/epsilonOptimizer/clip_by_norm_6*
T0*
use_locking( *K
_classA
?=loc:@over_options/q_func/action_value/fully_connected/weights* 
_output_shapes
:
АА
ў
WOptimizer/Adam/update_over_options/q_func/action_value/fully_connected/biases/ApplyAdam	ApplyAdam7over_options/q_func/action_value/fully_connected/biasesFOptimizer/over_options/q_func/action_value/fully_connected/biases/AdamHOptimizer/over_options/q_func/action_value/fully_connected/biases/Adam_1Optimizer/beta1_power/readOptimizer/beta2_power/readlearning_rateOptimizer/Adam/beta1Optimizer/Adam/beta2Optimizer/Adam/epsilonOptimizer/clip_by_norm_7*
T0*
use_locking( *J
_class@
><loc:@over_options/q_func/action_value/fully_connected/biases*
_output_shapes	
:А
м
ZOptimizer/Adam/update_over_options/q_func/action_value/fully_connected_1/weights/ApplyAdam	ApplyAdam:over_options/q_func/action_value/fully_connected_1/weightsIOptimizer/over_options/q_func/action_value/fully_connected_1/weights/AdamKOptimizer/over_options/q_func/action_value/fully_connected_1/weights/Adam_1Optimizer/beta1_power/readOptimizer/beta2_power/readlearning_rateOptimizer/Adam/beta1Optimizer/Adam/beta2Optimizer/Adam/epsilonOptimizer/clip_by_norm_8*
T0*
use_locking( *M
_classC
A?loc:@over_options/q_func/action_value/fully_connected_1/weights*
_output_shapes
:	А
в
YOptimizer/Adam/update_over_options/q_func/action_value/fully_connected_1/biases/ApplyAdam	ApplyAdam9over_options/q_func/action_value/fully_connected_1/biasesHOptimizer/over_options/q_func/action_value/fully_connected_1/biases/AdamJOptimizer/over_options/q_func/action_value/fully_connected_1/biases/Adam_1Optimizer/beta1_power/readOptimizer/beta2_power/readlearning_rateOptimizer/Adam/beta1Optimizer/Adam/beta2Optimizer/Adam/epsilonOptimizer/clip_by_norm_9*
T0*
use_locking( *L
_classB
@>loc:@over_options/q_func/action_value/fully_connected_1/biases*
_output_shapes
:
ё
Optimizer/Adam/mulMulOptimizer/beta1_power/readOptimizer/Adam/beta1I^Optimizer/Adam/update_over_options/q_func/convnet/Conv/weights/ApplyAdamH^Optimizer/Adam/update_over_options/q_func/convnet/Conv/biases/ApplyAdamK^Optimizer/Adam/update_over_options/q_func/convnet/Conv_1/weights/ApplyAdamJ^Optimizer/Adam/update_over_options/q_func/convnet/Conv_1/biases/ApplyAdamK^Optimizer/Adam/update_over_options/q_func/convnet/Conv_2/weights/ApplyAdamJ^Optimizer/Adam/update_over_options/q_func/convnet/Conv_2/biases/ApplyAdamY^Optimizer/Adam/update_over_options/q_func/action_value/fully_connected/weights/ApplyAdamX^Optimizer/Adam/update_over_options/q_func/action_value/fully_connected/biases/ApplyAdam[^Optimizer/Adam/update_over_options/q_func/action_value/fully_connected_1/weights/ApplyAdamZ^Optimizer/Adam/update_over_options/q_func/action_value/fully_connected_1/biases/ApplyAdam*
T0*;
_class1
/-loc:@over_options/q_func/convnet/Conv/weights*
_output_shapes
: 
—
Optimizer/Adam/AssignAssignOptimizer/beta1_powerOptimizer/Adam/mul*
T0*
validate_shape(*
use_locking( *;
_class1
/-loc:@over_options/q_func/convnet/Conv/weights*
_output_shapes
: 
а
Optimizer/Adam/mul_1MulOptimizer/beta2_power/readOptimizer/Adam/beta2I^Optimizer/Adam/update_over_options/q_func/convnet/Conv/weights/ApplyAdamH^Optimizer/Adam/update_over_options/q_func/convnet/Conv/biases/ApplyAdamK^Optimizer/Adam/update_over_options/q_func/convnet/Conv_1/weights/ApplyAdamJ^Optimizer/Adam/update_over_options/q_func/convnet/Conv_1/biases/ApplyAdamK^Optimizer/Adam/update_over_options/q_func/convnet/Conv_2/weights/ApplyAdamJ^Optimizer/Adam/update_over_options/q_func/convnet/Conv_2/biases/ApplyAdamY^Optimizer/Adam/update_over_options/q_func/action_value/fully_connected/weights/ApplyAdamX^Optimizer/Adam/update_over_options/q_func/action_value/fully_connected/biases/ApplyAdam[^Optimizer/Adam/update_over_options/q_func/action_value/fully_connected_1/weights/ApplyAdamZ^Optimizer/Adam/update_over_options/q_func/action_value/fully_connected_1/biases/ApplyAdam*
T0*;
_class1
/-loc:@over_options/q_func/convnet/Conv/weights*
_output_shapes
: 
’
Optimizer/Adam/Assign_1AssignOptimizer/beta2_powerOptimizer/Adam/mul_1*
T0*
validate_shape(*
use_locking( *;
_class1
/-loc:@over_options/q_func/convnet/Conv/weights*
_output_shapes
: 
э
Optimizer/AdamNoOpI^Optimizer/Adam/update_over_options/q_func/convnet/Conv/weights/ApplyAdamH^Optimizer/Adam/update_over_options/q_func/convnet/Conv/biases/ApplyAdamK^Optimizer/Adam/update_over_options/q_func/convnet/Conv_1/weights/ApplyAdamJ^Optimizer/Adam/update_over_options/q_func/convnet/Conv_1/biases/ApplyAdamK^Optimizer/Adam/update_over_options/q_func/convnet/Conv_2/weights/ApplyAdamJ^Optimizer/Adam/update_over_options/q_func/convnet/Conv_2/biases/ApplyAdamY^Optimizer/Adam/update_over_options/q_func/action_value/fully_connected/weights/ApplyAdamX^Optimizer/Adam/update_over_options/q_func/action_value/fully_connected/biases/ApplyAdam[^Optimizer/Adam/update_over_options/q_func/action_value/fully_connected_1/weights/ApplyAdamZ^Optimizer/Adam/update_over_options/q_func/action_value/fully_connected_1/biases/ApplyAdam^Optimizer/Adam/Assign^Optimizer/Adam/Assign_1
Ц
AssignAssign1target_q_func/action_value/fully_connected/biases<over_options/q_func/action_value/fully_connected/biases/read*
T0*
validate_shape(*
use_locking( *D
_class:
86loc:@target_q_func/action_value/fully_connected/biases*
_output_shapes	
:А
†
Assign_1Assign2target_q_func/action_value/fully_connected/weights=over_options/q_func/action_value/fully_connected/weights/read*
T0*
validate_shape(*
use_locking( *E
_class;
97loc:@target_q_func/action_value/fully_connected/weights* 
_output_shapes
:
АА
Э
Assign_2Assign3target_q_func/action_value/fully_connected_1/biases>over_options/q_func/action_value/fully_connected_1/biases/read*
T0*
validate_shape(*
use_locking( *F
_class<
:8loc:@target_q_func/action_value/fully_connected_1/biases*
_output_shapes
:
•
Assign_3Assign4target_q_func/action_value/fully_connected_1/weights?over_options/q_func/action_value/fully_connected_1/weights/read*
T0*
validate_shape(*
use_locking( *G
_class=
;9loc:@target_q_func/action_value/fully_connected_1/weights*
_output_shapes
:	А
з
Assign_4Assign!target_q_func/convnet/Conv/biases,over_options/q_func/convnet/Conv/biases/read*
T0*
validate_shape(*
use_locking( *4
_class*
(&loc:@target_q_func/convnet/Conv/biases*
_output_shapes
: 
ц
Assign_5Assign"target_q_func/convnet/Conv/weights-over_options/q_func/convnet/Conv/weights/read*
T0*
validate_shape(*
use_locking( *5
_class+
)'loc:@target_q_func/convnet/Conv/weights*&
_output_shapes
: 
н
Assign_6Assign#target_q_func/convnet/Conv_1/biases.over_options/q_func/convnet/Conv_1/biases/read*
T0*
validate_shape(*
use_locking( *6
_class,
*(loc:@target_q_func/convnet/Conv_1/biases*
_output_shapes
:@
ь
Assign_7Assign$target_q_func/convnet/Conv_1/weights/over_options/q_func/convnet/Conv_1/weights/read*
T0*
validate_shape(*
use_locking( *7
_class-
+)loc:@target_q_func/convnet/Conv_1/weights*&
_output_shapes
: @
н
Assign_8Assign#target_q_func/convnet/Conv_2/biases.over_options/q_func/convnet/Conv_2/biases/read*
T0*
validate_shape(*
use_locking( *6
_class,
*(loc:@target_q_func/convnet/Conv_2/biases*
_output_shapes
:@
ь
Assign_9Assign$target_q_func/convnet/Conv_2/weights/over_options/q_func/convnet/Conv_2/weights/read*
T0*
validate_shape(*
use_locking( *7
_class-
+)loc:@target_q_func/convnet/Conv_2/weights*&
_output_shapes
:@@
Х
!Update_target_fn/update_target_fnNoOp^Assign	^Assign_1	^Assign_2	^Assign_3	^Assign_4	^Assign_5	^Assign_6	^Assign_7	^Assign_8	^Assign_9
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
Ќ
save/SaveV2/tensor_namesConst*А
valueцBу
B7over_options/q_func/action_value/fully_connected/biasesB8over_options/q_func/action_value/fully_connected/weightsB9over_options/q_func/action_value/fully_connected_1/biasesB:over_options/q_func/action_value/fully_connected_1/weightsB'over_options/q_func/convnet/Conv/biasesB(over_options/q_func/convnet/Conv/weightsB)over_options/q_func/convnet/Conv_1/biasesB*over_options/q_func/convnet/Conv_1/weightsB)over_options/q_func/convnet/Conv_2/biasesB*over_options/q_func/convnet/Conv_2/weights*
dtype0*
_output_shapes
:

w
save/SaveV2/shape_and_slicesConst*'
valueB
B B B B B B B B B B *
dtype0*
_output_shapes
:

ё
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slices7over_options/q_func/action_value/fully_connected/biases8over_options/q_func/action_value/fully_connected/weights9over_options/q_func/action_value/fully_connected_1/biases:over_options/q_func/action_value/fully_connected_1/weights'over_options/q_func/convnet/Conv/biases(over_options/q_func/convnet/Conv/weights)over_options/q_func/convnet/Conv_1/biases*over_options/q_func/convnet/Conv_1/weights)over_options/q_func/convnet/Conv_2/biases*over_options/q_func/convnet/Conv_2/weights*
dtypes
2

}
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const*
_output_shapes
: 
Ы
save/RestoreV2/tensor_namesConst*L
valueCBAB7over_options/q_func/action_value/fully_connected/biases*
dtype0*
_output_shapes
:
h
save/RestoreV2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Р
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:
щ
save/AssignAssign7over_options/q_func/action_value/fully_connected/biasessave/RestoreV2*
T0*
validate_shape(*
use_locking(*J
_class@
><loc:@over_options/q_func/action_value/fully_connected/biases*
_output_shapes	
:А
Ю
save/RestoreV2_1/tensor_namesConst*M
valueDBBB8over_options/q_func/action_value/fully_connected/weights*
dtype0*
_output_shapes
:
j
!save/RestoreV2_1/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ц
save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes
:
Д
save/Assign_1Assign8over_options/q_func/action_value/fully_connected/weightssave/RestoreV2_1*
T0*
validate_shape(*
use_locking(*K
_classA
?=loc:@over_options/q_func/action_value/fully_connected/weights* 
_output_shapes
:
АА
Я
save/RestoreV2_2/tensor_namesConst*N
valueEBCB9over_options/q_func/action_value/fully_connected_1/biases*
dtype0*
_output_shapes
:
j
!save/RestoreV2_2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ц
save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
dtypes
2*
_output_shapes
:
А
save/Assign_2Assign9over_options/q_func/action_value/fully_connected_1/biasessave/RestoreV2_2*
T0*
validate_shape(*
use_locking(*L
_classB
@>loc:@over_options/q_func/action_value/fully_connected_1/biases*
_output_shapes
:
†
save/RestoreV2_3/tensor_namesConst*O
valueFBDB:over_options/q_func/action_value/fully_connected_1/weights*
dtype0*
_output_shapes
:
j
!save/RestoreV2_3/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ц
save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
dtypes
2*
_output_shapes
:
З
save/Assign_3Assign:over_options/q_func/action_value/fully_connected_1/weightssave/RestoreV2_3*
T0*
validate_shape(*
use_locking(*M
_classC
A?loc:@over_options/q_func/action_value/fully_connected_1/weights*
_output_shapes
:	А
Н
save/RestoreV2_4/tensor_namesConst*<
value3B1B'over_options/q_func/convnet/Conv/biases*
dtype0*
_output_shapes
:
j
!save/RestoreV2_4/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ц
save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
dtypes
2*
_output_shapes
:
№
save/Assign_4Assign'over_options/q_func/convnet/Conv/biasessave/RestoreV2_4*
T0*
validate_shape(*
use_locking(*:
_class0
.,loc:@over_options/q_func/convnet/Conv/biases*
_output_shapes
: 
О
save/RestoreV2_5/tensor_namesConst*=
value4B2B(over_options/q_func/convnet/Conv/weights*
dtype0*
_output_shapes
:
j
!save/RestoreV2_5/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ц
save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
dtypes
2*
_output_shapes
:
к
save/Assign_5Assign(over_options/q_func/convnet/Conv/weightssave/RestoreV2_5*
T0*
validate_shape(*
use_locking(*;
_class1
/-loc:@over_options/q_func/convnet/Conv/weights*&
_output_shapes
: 
П
save/RestoreV2_6/tensor_namesConst*>
value5B3B)over_options/q_func/convnet/Conv_1/biases*
dtype0*
_output_shapes
:
j
!save/RestoreV2_6/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ц
save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
dtypes
2*
_output_shapes
:
а
save/Assign_6Assign)over_options/q_func/convnet/Conv_1/biasessave/RestoreV2_6*
T0*
validate_shape(*
use_locking(*<
_class2
0.loc:@over_options/q_func/convnet/Conv_1/biases*
_output_shapes
:@
Р
save/RestoreV2_7/tensor_namesConst*?
value6B4B*over_options/q_func/convnet/Conv_1/weights*
dtype0*
_output_shapes
:
j
!save/RestoreV2_7/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ц
save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
dtypes
2*
_output_shapes
:
о
save/Assign_7Assign*over_options/q_func/convnet/Conv_1/weightssave/RestoreV2_7*
T0*
validate_shape(*
use_locking(*=
_class3
1/loc:@over_options/q_func/convnet/Conv_1/weights*&
_output_shapes
: @
П
save/RestoreV2_8/tensor_namesConst*>
value5B3B)over_options/q_func/convnet/Conv_2/biases*
dtype0*
_output_shapes
:
j
!save/RestoreV2_8/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ц
save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
dtypes
2*
_output_shapes
:
а
save/Assign_8Assign)over_options/q_func/convnet/Conv_2/biasessave/RestoreV2_8*
T0*
validate_shape(*
use_locking(*<
_class2
0.loc:@over_options/q_func/convnet/Conv_2/biases*
_output_shapes
:@
Р
save/RestoreV2_9/tensor_namesConst*?
value6B4B*over_options/q_func/convnet/Conv_2/weights*
dtype0*
_output_shapes
:
j
!save/RestoreV2_9/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
Ц
save/RestoreV2_9	RestoreV2
save/Constsave/RestoreV2_9/tensor_names!save/RestoreV2_9/shape_and_slices*
dtypes
2*
_output_shapes
:
о
save/Assign_9Assign*over_options/q_func/convnet/Conv_2/weightssave/RestoreV2_9*
T0*
validate_shape(*
use_locking(*=
_class3
1/loc:@over_options/q_func/convnet/Conv_2/weights*&
_output_shapes
:@@
ґ
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
>
initNoOp0^over_options/q_func/convnet/Conv/weights/Assign
?
init_1NoOp/^over_options/q_func/convnet/Conv/biases/Assign
B
init_2NoOp2^over_options/q_func/convnet/Conv_1/weights/Assign
A
init_3NoOp1^over_options/q_func/convnet/Conv_1/biases/Assign
B
init_4NoOp2^over_options/q_func/convnet/Conv_2/weights/Assign
A
init_5NoOp1^over_options/q_func/convnet/Conv_2/biases/Assign
P
init_6NoOp@^over_options/q_func/action_value/fully_connected/weights/Assign
O
init_7NoOp?^over_options/q_func/action_value/fully_connected/biases/Assign
R
init_8NoOpB^over_options/q_func/action_value/fully_connected_1/weights/Assign
Q
init_9NoOpA^over_options/q_func/action_value/fully_connected_1/biases/Assign
;
init_10NoOp*^target_q_func/convnet/Conv/weights/Assign
:
init_11NoOp)^target_q_func/convnet/Conv/biases/Assign
=
init_12NoOp,^target_q_func/convnet/Conv_1/weights/Assign
<
init_13NoOp+^target_q_func/convnet/Conv_1/biases/Assign
=
init_14NoOp,^target_q_func/convnet/Conv_2/weights/Assign
<
init_15NoOp+^target_q_func/convnet/Conv_2/biases/Assign
K
init_16NoOp:^target_q_func/action_value/fully_connected/weights/Assign
J
init_17NoOp9^target_q_func/action_value/fully_connected/biases/Assign
M
init_18NoOp<^target_q_func/action_value/fully_connected_1/weights/Assign
L
init_19NoOp;^target_q_func/action_value/fully_connected_1/biases/Assign
.
init_20NoOp^Optimizer/beta1_power/Assign
.
init_21NoOp^Optimizer/beta2_power/Assign
P
init_22NoOp?^Optimizer/over_options/q_func/convnet/Conv/weights/Adam/Assign
R
init_23NoOpA^Optimizer/over_options/q_func/convnet/Conv/weights/Adam_1/Assign
O
init_24NoOp>^Optimizer/over_options/q_func/convnet/Conv/biases/Adam/Assign
Q
init_25NoOp@^Optimizer/over_options/q_func/convnet/Conv/biases/Adam_1/Assign
R
init_26NoOpA^Optimizer/over_options/q_func/convnet/Conv_1/weights/Adam/Assign
T
init_27NoOpC^Optimizer/over_options/q_func/convnet/Conv_1/weights/Adam_1/Assign
Q
init_28NoOp@^Optimizer/over_options/q_func/convnet/Conv_1/biases/Adam/Assign
S
init_29NoOpB^Optimizer/over_options/q_func/convnet/Conv_1/biases/Adam_1/Assign
R
init_30NoOpA^Optimizer/over_options/q_func/convnet/Conv_2/weights/Adam/Assign
T
init_31NoOpC^Optimizer/over_options/q_func/convnet/Conv_2/weights/Adam_1/Assign
Q
init_32NoOp@^Optimizer/over_options/q_func/convnet/Conv_2/biases/Adam/Assign
S
init_33NoOpB^Optimizer/over_options/q_func/convnet/Conv_2/biases/Adam_1/Assign
`
init_34NoOpO^Optimizer/over_options/q_func/action_value/fully_connected/weights/Adam/Assign
b
init_35NoOpQ^Optimizer/over_options/q_func/action_value/fully_connected/weights/Adam_1/Assign
_
init_36NoOpN^Optimizer/over_options/q_func/action_value/fully_connected/biases/Adam/Assign
a
init_37NoOpP^Optimizer/over_options/q_func/action_value/fully_connected/biases/Adam_1/Assign
b
init_38NoOpQ^Optimizer/over_options/q_func/action_value/fully_connected_1/weights/Adam/Assign
d
init_39NoOpS^Optimizer/over_options/q_func/action_value/fully_connected_1/weights/Adam_1/Assign
a
init_40NoOpP^Optimizer/over_options/q_func/action_value/fully_connected_1/biases/Adam/Assign
c
init_41NoOpR^Optimizer/over_options/q_func/action_value/fully_connected_1/biases/Adam_1/Assign""ў
model_variables≈
¬
*over_options/q_func/convnet/Conv/weights:0
)over_options/q_func/convnet/Conv/biases:0
,over_options/q_func/convnet/Conv_1/weights:0
+over_options/q_func/convnet/Conv_1/biases:0
,over_options/q_func/convnet/Conv_2/weights:0
+over_options/q_func/convnet/Conv_2/biases:0
:over_options/q_func/action_value/fully_connected/weights:0
9over_options/q_func/action_value/fully_connected/biases:0
<over_options/q_func/action_value/fully_connected_1/weights:0
;over_options/q_func/action_value/fully_connected_1/biases:0
$target_q_func/convnet/Conv/weights:0
#target_q_func/convnet/Conv/biases:0
&target_q_func/convnet/Conv_1/weights:0
%target_q_func/convnet/Conv_1/biases:0
&target_q_func/convnet/Conv_2/weights:0
%target_q_func/convnet/Conv_2/biases:0
4target_q_func/action_value/fully_connected/weights:0
3target_q_func/action_value/fully_connected/biases:0
6target_q_func/action_value/fully_connected_1/weights:0
5target_q_func/action_value/fully_connected_1/biases:0"б
trainable_variables…∆
О
*over_options/q_func/convnet/Conv/weights:0/over_options/q_func/convnet/Conv/weights/Assign/over_options/q_func/convnet/Conv/weights/read:0
Л
)over_options/q_func/convnet/Conv/biases:0.over_options/q_func/convnet/Conv/biases/Assign.over_options/q_func/convnet/Conv/biases/read:0
Ф
,over_options/q_func/convnet/Conv_1/weights:01over_options/q_func/convnet/Conv_1/weights/Assign1over_options/q_func/convnet/Conv_1/weights/read:0
С
+over_options/q_func/convnet/Conv_1/biases:00over_options/q_func/convnet/Conv_1/biases/Assign0over_options/q_func/convnet/Conv_1/biases/read:0
Ф
,over_options/q_func/convnet/Conv_2/weights:01over_options/q_func/convnet/Conv_2/weights/Assign1over_options/q_func/convnet/Conv_2/weights/read:0
С
+over_options/q_func/convnet/Conv_2/biases:00over_options/q_func/convnet/Conv_2/biases/Assign0over_options/q_func/convnet/Conv_2/biases/read:0
Њ
:over_options/q_func/action_value/fully_connected/weights:0?over_options/q_func/action_value/fully_connected/weights/Assign?over_options/q_func/action_value/fully_connected/weights/read:0
ї
9over_options/q_func/action_value/fully_connected/biases:0>over_options/q_func/action_value/fully_connected/biases/Assign>over_options/q_func/action_value/fully_connected/biases/read:0
ƒ
<over_options/q_func/action_value/fully_connected_1/weights:0Aover_options/q_func/action_value/fully_connected_1/weights/AssignAover_options/q_func/action_value/fully_connected_1/weights/read:0
Ѕ
;over_options/q_func/action_value/fully_connected_1/biases:0@over_options/q_func/action_value/fully_connected_1/biases/Assign@over_options/q_func/action_value/fully_connected_1/biases/read:0
|
$target_q_func/convnet/Conv/weights:0)target_q_func/convnet/Conv/weights/Assign)target_q_func/convnet/Conv/weights/read:0
y
#target_q_func/convnet/Conv/biases:0(target_q_func/convnet/Conv/biases/Assign(target_q_func/convnet/Conv/biases/read:0
В
&target_q_func/convnet/Conv_1/weights:0+target_q_func/convnet/Conv_1/weights/Assign+target_q_func/convnet/Conv_1/weights/read:0

%target_q_func/convnet/Conv_1/biases:0*target_q_func/convnet/Conv_1/biases/Assign*target_q_func/convnet/Conv_1/biases/read:0
В
&target_q_func/convnet/Conv_2/weights:0+target_q_func/convnet/Conv_2/weights/Assign+target_q_func/convnet/Conv_2/weights/read:0

%target_q_func/convnet/Conv_2/biases:0*target_q_func/convnet/Conv_2/biases/Assign*target_q_func/convnet/Conv_2/biases/read:0
ђ
4target_q_func/action_value/fully_connected/weights:09target_q_func/action_value/fully_connected/weights/Assign9target_q_func/action_value/fully_connected/weights/read:0
©
3target_q_func/action_value/fully_connected/biases:08target_q_func/action_value/fully_connected/biases/Assign8target_q_func/action_value/fully_connected/biases/read:0
≤
6target_q_func/action_value/fully_connected_1/weights:0;target_q_func/action_value/fully_connected_1/weights/Assign;target_q_func/action_value/fully_connected_1/weights/read:0
ѓ
5target_q_func/action_value/fully_connected_1/biases:0:target_q_func/action_value/fully_connected_1/biases/Assign:target_q_func/action_value/fully_connected_1/biases/read:0"√;
	variablesµ;≤;
О
*over_options/q_func/convnet/Conv/weights:0/over_options/q_func/convnet/Conv/weights/Assign/over_options/q_func/convnet/Conv/weights/read:0
Л
)over_options/q_func/convnet/Conv/biases:0.over_options/q_func/convnet/Conv/biases/Assign.over_options/q_func/convnet/Conv/biases/read:0
Ф
,over_options/q_func/convnet/Conv_1/weights:01over_options/q_func/convnet/Conv_1/weights/Assign1over_options/q_func/convnet/Conv_1/weights/read:0
С
+over_options/q_func/convnet/Conv_1/biases:00over_options/q_func/convnet/Conv_1/biases/Assign0over_options/q_func/convnet/Conv_1/biases/read:0
Ф
,over_options/q_func/convnet/Conv_2/weights:01over_options/q_func/convnet/Conv_2/weights/Assign1over_options/q_func/convnet/Conv_2/weights/read:0
С
+over_options/q_func/convnet/Conv_2/biases:00over_options/q_func/convnet/Conv_2/biases/Assign0over_options/q_func/convnet/Conv_2/biases/read:0
Њ
:over_options/q_func/action_value/fully_connected/weights:0?over_options/q_func/action_value/fully_connected/weights/Assign?over_options/q_func/action_value/fully_connected/weights/read:0
ї
9over_options/q_func/action_value/fully_connected/biases:0>over_options/q_func/action_value/fully_connected/biases/Assign>over_options/q_func/action_value/fully_connected/biases/read:0
ƒ
<over_options/q_func/action_value/fully_connected_1/weights:0Aover_options/q_func/action_value/fully_connected_1/weights/AssignAover_options/q_func/action_value/fully_connected_1/weights/read:0
Ѕ
;over_options/q_func/action_value/fully_connected_1/biases:0@over_options/q_func/action_value/fully_connected_1/biases/Assign@over_options/q_func/action_value/fully_connected_1/biases/read:0
|
$target_q_func/convnet/Conv/weights:0)target_q_func/convnet/Conv/weights/Assign)target_q_func/convnet/Conv/weights/read:0
y
#target_q_func/convnet/Conv/biases:0(target_q_func/convnet/Conv/biases/Assign(target_q_func/convnet/Conv/biases/read:0
В
&target_q_func/convnet/Conv_1/weights:0+target_q_func/convnet/Conv_1/weights/Assign+target_q_func/convnet/Conv_1/weights/read:0

%target_q_func/convnet/Conv_1/biases:0*target_q_func/convnet/Conv_1/biases/Assign*target_q_func/convnet/Conv_1/biases/read:0
В
&target_q_func/convnet/Conv_2/weights:0+target_q_func/convnet/Conv_2/weights/Assign+target_q_func/convnet/Conv_2/weights/read:0

%target_q_func/convnet/Conv_2/biases:0*target_q_func/convnet/Conv_2/biases/Assign*target_q_func/convnet/Conv_2/biases/read:0
ђ
4target_q_func/action_value/fully_connected/weights:09target_q_func/action_value/fully_connected/weights/Assign9target_q_func/action_value/fully_connected/weights/read:0
©
3target_q_func/action_value/fully_connected/biases:08target_q_func/action_value/fully_connected/biases/Assign8target_q_func/action_value/fully_connected/biases/read:0
≤
6target_q_func/action_value/fully_connected_1/weights:0;target_q_func/action_value/fully_connected_1/weights/Assign;target_q_func/action_value/fully_connected_1/weights/read:0
ѓ
5target_q_func/action_value/fully_connected_1/biases:0:target_q_func/action_value/fully_connected_1/biases/Assign:target_q_func/action_value/fully_connected_1/biases/read:0
U
Optimizer/beta1_power:0Optimizer/beta1_power/AssignOptimizer/beta1_power/read:0
U
Optimizer/beta2_power:0Optimizer/beta2_power/AssignOptimizer/beta2_power/read:0
ї
9Optimizer/over_options/q_func/convnet/Conv/weights/Adam:0>Optimizer/over_options/q_func/convnet/Conv/weights/Adam/Assign>Optimizer/over_options/q_func/convnet/Conv/weights/Adam/read:0
Ѕ
;Optimizer/over_options/q_func/convnet/Conv/weights/Adam_1:0@Optimizer/over_options/q_func/convnet/Conv/weights/Adam_1/Assign@Optimizer/over_options/q_func/convnet/Conv/weights/Adam_1/read:0
Є
8Optimizer/over_options/q_func/convnet/Conv/biases/Adam:0=Optimizer/over_options/q_func/convnet/Conv/biases/Adam/Assign=Optimizer/over_options/q_func/convnet/Conv/biases/Adam/read:0
Њ
:Optimizer/over_options/q_func/convnet/Conv/biases/Adam_1:0?Optimizer/over_options/q_func/convnet/Conv/biases/Adam_1/Assign?Optimizer/over_options/q_func/convnet/Conv/biases/Adam_1/read:0
Ѕ
;Optimizer/over_options/q_func/convnet/Conv_1/weights/Adam:0@Optimizer/over_options/q_func/convnet/Conv_1/weights/Adam/Assign@Optimizer/over_options/q_func/convnet/Conv_1/weights/Adam/read:0
«
=Optimizer/over_options/q_func/convnet/Conv_1/weights/Adam_1:0BOptimizer/over_options/q_func/convnet/Conv_1/weights/Adam_1/AssignBOptimizer/over_options/q_func/convnet/Conv_1/weights/Adam_1/read:0
Њ
:Optimizer/over_options/q_func/convnet/Conv_1/biases/Adam:0?Optimizer/over_options/q_func/convnet/Conv_1/biases/Adam/Assign?Optimizer/over_options/q_func/convnet/Conv_1/biases/Adam/read:0
ƒ
<Optimizer/over_options/q_func/convnet/Conv_1/biases/Adam_1:0AOptimizer/over_options/q_func/convnet/Conv_1/biases/Adam_1/AssignAOptimizer/over_options/q_func/convnet/Conv_1/biases/Adam_1/read:0
Ѕ
;Optimizer/over_options/q_func/convnet/Conv_2/weights/Adam:0@Optimizer/over_options/q_func/convnet/Conv_2/weights/Adam/Assign@Optimizer/over_options/q_func/convnet/Conv_2/weights/Adam/read:0
«
=Optimizer/over_options/q_func/convnet/Conv_2/weights/Adam_1:0BOptimizer/over_options/q_func/convnet/Conv_2/weights/Adam_1/AssignBOptimizer/over_options/q_func/convnet/Conv_2/weights/Adam_1/read:0
Њ
:Optimizer/over_options/q_func/convnet/Conv_2/biases/Adam:0?Optimizer/over_options/q_func/convnet/Conv_2/biases/Adam/Assign?Optimizer/over_options/q_func/convnet/Conv_2/biases/Adam/read:0
ƒ
<Optimizer/over_options/q_func/convnet/Conv_2/biases/Adam_1:0AOptimizer/over_options/q_func/convnet/Conv_2/biases/Adam_1/AssignAOptimizer/over_options/q_func/convnet/Conv_2/biases/Adam_1/read:0
л
IOptimizer/over_options/q_func/action_value/fully_connected/weights/Adam:0NOptimizer/over_options/q_func/action_value/fully_connected/weights/Adam/AssignNOptimizer/over_options/q_func/action_value/fully_connected/weights/Adam/read:0
с
KOptimizer/over_options/q_func/action_value/fully_connected/weights/Adam_1:0POptimizer/over_options/q_func/action_value/fully_connected/weights/Adam_1/AssignPOptimizer/over_options/q_func/action_value/fully_connected/weights/Adam_1/read:0
и
HOptimizer/over_options/q_func/action_value/fully_connected/biases/Adam:0MOptimizer/over_options/q_func/action_value/fully_connected/biases/Adam/AssignMOptimizer/over_options/q_func/action_value/fully_connected/biases/Adam/read:0
о
JOptimizer/over_options/q_func/action_value/fully_connected/biases/Adam_1:0OOptimizer/over_options/q_func/action_value/fully_connected/biases/Adam_1/AssignOOptimizer/over_options/q_func/action_value/fully_connected/biases/Adam_1/read:0
с
KOptimizer/over_options/q_func/action_value/fully_connected_1/weights/Adam:0POptimizer/over_options/q_func/action_value/fully_connected_1/weights/Adam/AssignPOptimizer/over_options/q_func/action_value/fully_connected_1/weights/Adam/read:0
ч
MOptimizer/over_options/q_func/action_value/fully_connected_1/weights/Adam_1:0ROptimizer/over_options/q_func/action_value/fully_connected_1/weights/Adam_1/AssignROptimizer/over_options/q_func/action_value/fully_connected_1/weights/Adam_1/read:0
о
JOptimizer/over_options/q_func/action_value/fully_connected_1/biases/Adam:0OOptimizer/over_options/q_func/action_value/fully_connected_1/biases/Adam/AssignOOptimizer/over_options/q_func/action_value/fully_connected_1/biases/Adam/read:0
ф
LOptimizer/over_options/q_func/action_value/fully_connected_1/biases/Adam_1:0QOptimizer/over_options/q_func/action_value/fully_connected_1/biases/Adam_1/AssignQOptimizer/over_options/q_func/action_value/fully_connected_1/biases/Adam_1/read:0"
train_op

Optimizer/AdambOQњ