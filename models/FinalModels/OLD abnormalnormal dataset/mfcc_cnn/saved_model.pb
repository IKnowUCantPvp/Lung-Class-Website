яр
ђТ
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
2
L2Loss
t"T
output"T"
Ttype:
2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	
С
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring Ј
@
StaticRegexFullMatch	
input

output
"
patternstring
2
StopGradient

input"T
output"T"	
Ttype
ї
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
-
Tanh
x"T
y"T"
Ttype:

2

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8Њэ
~
Adam/softmax/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/softmax/bias/v
w
'Adam/softmax/bias/v/Read/ReadVariableOpReadVariableOpAdam/softmax/bias/v*
_output_shapes
:*
dtype0

Adam/softmax/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*&
shared_nameAdam/softmax/kernel/v

)Adam/softmax/kernel/v/Read/ReadVariableOpReadVariableOpAdam/softmax/kernel/v*
_output_shapes

:@*
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:@*
dtype0

Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:		@*$
shared_nameAdam/dense/kernel/v
|
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes
:		@*
dtype0

Adam/conv2d_relu_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameAdam/conv2d_relu_4/bias/v

-Adam/conv2d_relu_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_relu_4/bias/v*
_output_shapes
: *
dtype0

Adam/conv2d_relu_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *,
shared_nameAdam/conv2d_relu_4/kernel/v

/Adam/conv2d_relu_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_relu_4/kernel/v*&
_output_shapes
:  *
dtype0

Adam/conv2d_relu_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameAdam/conv2d_relu_3/bias/v

-Adam/conv2d_relu_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_relu_3/bias/v*
_output_shapes
: *
dtype0

Adam/conv2d_relu_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_nameAdam/conv2d_relu_3/kernel/v

/Adam/conv2d_relu_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_relu_3/kernel/v*&
_output_shapes
: *
dtype0

Adam/conv2d_relu_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameAdam/conv2d_relu_2/bias/v

-Adam/conv2d_relu_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_relu_2/bias/v*
_output_shapes
:*
dtype0

Adam/conv2d_relu_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameAdam/conv2d_relu_2/kernel/v

/Adam/conv2d_relu_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_relu_2/kernel/v*&
_output_shapes
:*
dtype0

Adam/conv2d_relu_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameAdam/conv2d_relu_1/bias/v

-Adam/conv2d_relu_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_relu_1/bias/v*
_output_shapes
:*
dtype0

Adam/conv2d_relu_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameAdam/conv2d_relu_1/kernel/v

/Adam/conv2d_relu_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_relu_1/kernel/v*&
_output_shapes
:*
dtype0

Adam/conv2d_tanh/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_tanh/bias/v

+Adam/conv2d_tanh/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_tanh/bias/v*
_output_shapes
:*
dtype0

Adam/conv2d_tanh/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameAdam/conv2d_tanh/kernel/v

-Adam/conv2d_tanh/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_tanh/kernel/v*&
_output_shapes
:*
dtype0

Adam/batch_norm/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:М*'
shared_nameAdam/batch_norm/beta/v
~
*Adam/batch_norm/beta/v/Read/ReadVariableOpReadVariableOpAdam/batch_norm/beta/v*
_output_shapes	
:М*
dtype0

Adam/batch_norm/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:М*(
shared_nameAdam/batch_norm/gamma/v

+Adam/batch_norm/gamma/v/Read/ReadVariableOpReadVariableOpAdam/batch_norm/gamma/v*
_output_shapes	
:М*
dtype0
~
Adam/softmax/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/softmax/bias/m
w
'Adam/softmax/bias/m/Read/ReadVariableOpReadVariableOpAdam/softmax/bias/m*
_output_shapes
:*
dtype0

Adam/softmax/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*&
shared_nameAdam/softmax/kernel/m

)Adam/softmax/kernel/m/Read/ReadVariableOpReadVariableOpAdam/softmax/kernel/m*
_output_shapes

:@*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:@*
dtype0

Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:		@*$
shared_nameAdam/dense/kernel/m
|
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes
:		@*
dtype0

Adam/conv2d_relu_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameAdam/conv2d_relu_4/bias/m

-Adam/conv2d_relu_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_relu_4/bias/m*
_output_shapes
: *
dtype0

Adam/conv2d_relu_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *,
shared_nameAdam/conv2d_relu_4/kernel/m

/Adam/conv2d_relu_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_relu_4/kernel/m*&
_output_shapes
:  *
dtype0

Adam/conv2d_relu_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameAdam/conv2d_relu_3/bias/m

-Adam/conv2d_relu_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_relu_3/bias/m*
_output_shapes
: *
dtype0

Adam/conv2d_relu_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_nameAdam/conv2d_relu_3/kernel/m

/Adam/conv2d_relu_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_relu_3/kernel/m*&
_output_shapes
: *
dtype0

Adam/conv2d_relu_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameAdam/conv2d_relu_2/bias/m

-Adam/conv2d_relu_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_relu_2/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d_relu_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameAdam/conv2d_relu_2/kernel/m

/Adam/conv2d_relu_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_relu_2/kernel/m*&
_output_shapes
:*
dtype0

Adam/conv2d_relu_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameAdam/conv2d_relu_1/bias/m

-Adam/conv2d_relu_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_relu_1/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d_relu_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameAdam/conv2d_relu_1/kernel/m

/Adam/conv2d_relu_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_relu_1/kernel/m*&
_output_shapes
:*
dtype0

Adam/conv2d_tanh/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_tanh/bias/m

+Adam/conv2d_tanh/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_tanh/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d_tanh/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameAdam/conv2d_tanh/kernel/m

-Adam/conv2d_tanh/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_tanh/kernel/m*&
_output_shapes
:*
dtype0

Adam/batch_norm/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:М*'
shared_nameAdam/batch_norm/beta/m
~
*Adam/batch_norm/beta/m/Read/ReadVariableOpReadVariableOpAdam/batch_norm/beta/m*
_output_shapes	
:М*
dtype0

Adam/batch_norm/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:М*(
shared_nameAdam/batch_norm/gamma/m

+Adam/batch_norm/gamma/m/Read/ReadVariableOpReadVariableOpAdam/batch_norm/gamma/m*
_output_shapes	
:М*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
p
softmax/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namesoftmax/bias
i
 softmax/bias/Read/ReadVariableOpReadVariableOpsoftmax/bias*
_output_shapes
:*
dtype0
x
softmax/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_namesoftmax/kernel
q
"softmax/kernel/Read/ReadVariableOpReadVariableOpsoftmax/kernel*
_output_shapes

:@*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:@*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:		@*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:		@*
dtype0
|
conv2d_relu_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameconv2d_relu_4/bias
u
&conv2d_relu_4/bias/Read/ReadVariableOpReadVariableOpconv2d_relu_4/bias*
_output_shapes
: *
dtype0

conv2d_relu_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *%
shared_nameconv2d_relu_4/kernel

(conv2d_relu_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_relu_4/kernel*&
_output_shapes
:  *
dtype0
|
conv2d_relu_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameconv2d_relu_3/bias
u
&conv2d_relu_3/bias/Read/ReadVariableOpReadVariableOpconv2d_relu_3/bias*
_output_shapes
: *
dtype0

conv2d_relu_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameconv2d_relu_3/kernel

(conv2d_relu_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_relu_3/kernel*&
_output_shapes
: *
dtype0
|
conv2d_relu_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameconv2d_relu_2/bias
u
&conv2d_relu_2/bias/Read/ReadVariableOpReadVariableOpconv2d_relu_2/bias*
_output_shapes
:*
dtype0

conv2d_relu_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameconv2d_relu_2/kernel

(conv2d_relu_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_relu_2/kernel*&
_output_shapes
:*
dtype0
|
conv2d_relu_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameconv2d_relu_1/bias
u
&conv2d_relu_1/bias/Read/ReadVariableOpReadVariableOpconv2d_relu_1/bias*
_output_shapes
:*
dtype0

conv2d_relu_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameconv2d_relu_1/kernel

(conv2d_relu_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_relu_1/kernel*&
_output_shapes
:*
dtype0
x
conv2d_tanh/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_tanh/bias
q
$conv2d_tanh/bias/Read/ReadVariableOpReadVariableOpconv2d_tanh/bias*
_output_shapes
:*
dtype0

conv2d_tanh/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameconv2d_tanh/kernel

&conv2d_tanh/kernel/Read/ReadVariableOpReadVariableOpconv2d_tanh/kernel*&
_output_shapes
:*
dtype0
w
batch_norm/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:М* 
shared_namebatch_norm/beta
p
#batch_norm/beta/Read/ReadVariableOpReadVariableOpbatch_norm/beta*
_output_shapes	
:М*
dtype0
y
batch_norm/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:М*!
shared_namebatch_norm/gamma
r
$batch_norm/gamma/Read/ReadVariableOpReadVariableOpbatch_norm/gamma*
_output_shapes	
:М*
dtype0

serving_default_input_1Placeholder*0
_output_shapes
:џџџџџџџџџ(М*
dtype0*%
shape:џџџџџџџџџ(М
ў
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1batch_norm/gammabatch_norm/betaconv2d_tanh/kernelconv2d_tanh/biasconv2d_relu_1/kernelconv2d_relu_1/biasconv2d_relu_2/kernelconv2d_relu_2/biasconv2d_relu_3/kernelconv2d_relu_3/biasconv2d_relu_4/kernelconv2d_relu_4/biasdense/kernel
dense/biassoftmax/kernelsoftmax/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference_signature_wrapper_101770

NoOpNoOp
Б
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ы
valueрBм Bд
ф
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer_with_weights-5
layer-10
layer-11
layer-12
layer_with_weights-6
layer-13
layer_with_weights-7
layer-14
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
* 
Џ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
axis
	 gamma
!beta*
Ш
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses

(kernel
)bias
 *_jit_compiled_convolution_op*

+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses* 
Ш
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses

7kernel
8bias
 9_jit_compiled_convolution_op*

:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses* 
Ш
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses

Fkernel
Gbias
 H_jit_compiled_convolution_op*

I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses* 
Ш
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses

Ukernel
Vbias
 W_jit_compiled_convolution_op*

X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses* 
Ш
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses

dkernel
ebias
 f_jit_compiled_convolution_op*

g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses* 
Ѕ
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses
s_random_generator* 
І
t	variables
utrainable_variables
vregularization_losses
w	keras_api
x__call__
*y&call_and_return_all_conditional_losses

zkernel
{bias*
Њ
|	variables
}trainable_variables
~regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias*
|
 0
!1
(2
)3
74
85
F6
G7
U8
V9
d10
e11
z12
{13
14
15*
|
 0
!1
(2
)3
74
85
F6
G7
U8
V9
d10
e11
z12
{13
14
15*
* 
Е
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
trace_0
trace_1
trace_2
trace_3* 
:
trace_0
trace_1
trace_2
trace_3* 
* 

	iter
beta_1
beta_2

decay
learning_rate m!m(m)m7m8mFmGmUmVmdmemzm{m	m	m v!v(v)v7v8vFv GvЁUvЂVvЃdvЄevЅzvІ{vЇ	vЈ	vЉ*

serving_default* 

 0
!1*

 0
!1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

trace_0* 

trace_0* 
* 
_Y
VARIABLE_VALUEbatch_norm/gamma5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEbatch_norm/beta4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUE*

(0
)1*

(0
)1*
* 

non_trainable_variables
layers
 metrics
 Ёlayer_regularization_losses
Ђlayer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses*

Ѓtrace_0* 

Єtrace_0* 
b\
VARIABLE_VALUEconv2d_tanh/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEconv2d_tanh/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

Ѕnon_trainable_variables
Іlayers
Їmetrics
 Јlayer_regularization_losses
Љlayer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses* 

Њtrace_0* 

Ћtrace_0* 

70
81*

70
81*
* 

Ќnon_trainable_variables
­layers
Ўmetrics
 Џlayer_regularization_losses
Аlayer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses*

Бtrace_0* 

Вtrace_0* 
d^
VARIABLE_VALUEconv2d_relu_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEconv2d_relu_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

Гnon_trainable_variables
Дlayers
Еmetrics
 Жlayer_regularization_losses
Зlayer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses* 

Иtrace_0* 

Йtrace_0* 

F0
G1*

F0
G1*
* 

Кnon_trainable_variables
Лlayers
Мmetrics
 Нlayer_regularization_losses
Оlayer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses*

Пtrace_0* 

Рtrace_0* 
d^
VARIABLE_VALUEconv2d_relu_2/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEconv2d_relu_2/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

Сnon_trainable_variables
Тlayers
Уmetrics
 Фlayer_regularization_losses
Хlayer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses* 

Цtrace_0* 

Чtrace_0* 

U0
V1*

U0
V1*
* 

Шnon_trainable_variables
Щlayers
Ъmetrics
 Ыlayer_regularization_losses
Ьlayer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses*

Эtrace_0* 

Юtrace_0* 
d^
VARIABLE_VALUEconv2d_relu_3/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEconv2d_relu_3/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

Яnon_trainable_variables
аlayers
бmetrics
 вlayer_regularization_losses
гlayer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses* 

дtrace_0* 

еtrace_0* 

d0
e1*

d0
e1*
* 

жnon_trainable_variables
зlayers
иmetrics
 йlayer_regularization_losses
кlayer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses*

лtrace_0* 

мtrace_0* 
d^
VARIABLE_VALUEconv2d_relu_4/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEconv2d_relu_4/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

нnon_trainable_variables
оlayers
пmetrics
 рlayer_regularization_losses
сlayer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses* 

тtrace_0* 

уtrace_0* 
* 
* 
* 

фnon_trainable_variables
хlayers
цmetrics
 чlayer_regularization_losses
шlayer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses* 

щtrace_0
ъtrace_1* 

ыtrace_0
ьtrace_1* 
* 

z0
{1*

z0
{1*
* 
З
эnon_trainable_variables
юlayers
яmetrics
 №layer_regularization_losses
ёlayer_metrics
t	variables
utrainable_variables
vregularization_losses
x__call__
ђactivity_regularizer_fn
*y&call_and_return_all_conditional_losses
'ѓ"call_and_return_conditional_losses*

єtrace_0* 

ѕtrace_0* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

іnon_trainable_variables
їlayers
јmetrics
 љlayer_regularization_losses
њlayer_metrics
|	variables
}trainable_variables
~regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

ћtrace_0* 

ќtrace_0* 
^X
VARIABLE_VALUEsoftmax/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEsoftmax/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
r
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14*

§0
ў1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

џtrace_0* 

trace_0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
	variables
	keras_api

total

count*
M
	variables
	keras_api

total

count

_fn_kwargs*
* 
* 

0
1*

	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
|
VARIABLE_VALUEAdam/batch_norm/gamma/mQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/batch_norm/beta/mPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/conv2d_tanh/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/conv2d_tanh/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/conv2d_relu_1/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/conv2d_relu_1/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/conv2d_relu_2/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/conv2d_relu_2/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/conv2d_relu_3/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/conv2d_relu_3/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/conv2d_relu_4/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/conv2d_relu_4/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/softmax/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/softmax/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/batch_norm/gamma/vQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/batch_norm/beta/vPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/conv2d_tanh/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/conv2d_tanh/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/conv2d_relu_1/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/conv2d_relu_1/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/conv2d_relu_2/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/conv2d_relu_2/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/conv2d_relu_3/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/conv2d_relu_3/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/conv2d_relu_4/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/conv2d_relu_4/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/softmax/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/softmax/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$batch_norm/gamma/Read/ReadVariableOp#batch_norm/beta/Read/ReadVariableOp&conv2d_tanh/kernel/Read/ReadVariableOp$conv2d_tanh/bias/Read/ReadVariableOp(conv2d_relu_1/kernel/Read/ReadVariableOp&conv2d_relu_1/bias/Read/ReadVariableOp(conv2d_relu_2/kernel/Read/ReadVariableOp&conv2d_relu_2/bias/Read/ReadVariableOp(conv2d_relu_3/kernel/Read/ReadVariableOp&conv2d_relu_3/bias/Read/ReadVariableOp(conv2d_relu_4/kernel/Read/ReadVariableOp&conv2d_relu_4/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"softmax/kernel/Read/ReadVariableOp softmax/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Adam/batch_norm/gamma/m/Read/ReadVariableOp*Adam/batch_norm/beta/m/Read/ReadVariableOp-Adam/conv2d_tanh/kernel/m/Read/ReadVariableOp+Adam/conv2d_tanh/bias/m/Read/ReadVariableOp/Adam/conv2d_relu_1/kernel/m/Read/ReadVariableOp-Adam/conv2d_relu_1/bias/m/Read/ReadVariableOp/Adam/conv2d_relu_2/kernel/m/Read/ReadVariableOp-Adam/conv2d_relu_2/bias/m/Read/ReadVariableOp/Adam/conv2d_relu_3/kernel/m/Read/ReadVariableOp-Adam/conv2d_relu_3/bias/m/Read/ReadVariableOp/Adam/conv2d_relu_4/kernel/m/Read/ReadVariableOp-Adam/conv2d_relu_4/bias/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/softmax/kernel/m/Read/ReadVariableOp'Adam/softmax/bias/m/Read/ReadVariableOp+Adam/batch_norm/gamma/v/Read/ReadVariableOp*Adam/batch_norm/beta/v/Read/ReadVariableOp-Adam/conv2d_tanh/kernel/v/Read/ReadVariableOp+Adam/conv2d_tanh/bias/v/Read/ReadVariableOp/Adam/conv2d_relu_1/kernel/v/Read/ReadVariableOp-Adam/conv2d_relu_1/bias/v/Read/ReadVariableOp/Adam/conv2d_relu_2/kernel/v/Read/ReadVariableOp-Adam/conv2d_relu_2/bias/v/Read/ReadVariableOp/Adam/conv2d_relu_3/kernel/v/Read/ReadVariableOp-Adam/conv2d_relu_3/bias/v/Read/ReadVariableOp/Adam/conv2d_relu_4/kernel/v/Read/ReadVariableOp-Adam/conv2d_relu_4/bias/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp)Adam/softmax/kernel/v/Read/ReadVariableOp'Adam/softmax/bias/v/Read/ReadVariableOpConst*F
Tin?
=2;	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *(
f#R!
__inference__traced_save_102497

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamebatch_norm/gammabatch_norm/betaconv2d_tanh/kernelconv2d_tanh/biasconv2d_relu_1/kernelconv2d_relu_1/biasconv2d_relu_2/kernelconv2d_relu_2/biasconv2d_relu_3/kernelconv2d_relu_3/biasconv2d_relu_4/kernelconv2d_relu_4/biasdense/kernel
dense/biassoftmax/kernelsoftmax/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/batch_norm/gamma/mAdam/batch_norm/beta/mAdam/conv2d_tanh/kernel/mAdam/conv2d_tanh/bias/mAdam/conv2d_relu_1/kernel/mAdam/conv2d_relu_1/bias/mAdam/conv2d_relu_2/kernel/mAdam/conv2d_relu_2/bias/mAdam/conv2d_relu_3/kernel/mAdam/conv2d_relu_3/bias/mAdam/conv2d_relu_4/kernel/mAdam/conv2d_relu_4/bias/mAdam/dense/kernel/mAdam/dense/bias/mAdam/softmax/kernel/mAdam/softmax/bias/mAdam/batch_norm/gamma/vAdam/batch_norm/beta/vAdam/conv2d_tanh/kernel/vAdam/conv2d_tanh/bias/vAdam/conv2d_relu_1/kernel/vAdam/conv2d_relu_1/bias/vAdam/conv2d_relu_2/kernel/vAdam/conv2d_relu_2/bias/vAdam/conv2d_relu_3/kernel/vAdam/conv2d_relu_3/bias/vAdam/conv2d_relu_4/kernel/vAdam/conv2d_relu_4/bias/vAdam/dense/kernel/vAdam/dense/bias/vAdam/softmax/kernel/vAdam/softmax/bias/v*E
Tin>
<2:*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__traced_restore_102678џЯ
Џ
И
$__inference_signature_wrapper_101770
input_1
unknown:	М
	unknown_0:	М#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7: 
	unknown_8: #
	unknown_9:  

unknown_10: 

unknown_11:		@

unknown_12:@

unknown_13:@

unknown_14:
identityЂStatefulPartitionedCallѕ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__wrapped_model_101034o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:џџџџџџџџџ(М: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:џџџџџџџџџ(М
!
_user_specified_name	input_1
Т

&__inference_dense_layer_call_fn_102261

inputs
unknown:		@
	unknown_0:@
identityЂStatefulPartitionedCallй
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_101244o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ	: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ	
 
_user_specified_nameinputs


I__inference_conv2d_relu_3_layer_call_and_return_conditional_losses_101194

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


є
C__inference_softmax_layer_call_and_return_conditional_losses_101269

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
к
a
C__inference_dropout_layer_call_and_return_conditional_losses_102240

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:џџџџџџџџџ	\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:џџџџџџџџџ	"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ	:P L
(
_output_shapes
:џџџџџџџџџ	
 
_user_specified_nameinputs
уr
Њ
J__inference_2d_convolution_layer_call_and_return_conditional_losses_101939

inputs9
*batch_norm_reshape_readvariableop_resource:	М;
,batch_norm_reshape_1_readvariableop_resource:	МD
*conv2d_tanh_conv2d_readvariableop_resource:9
+conv2d_tanh_biasadd_readvariableop_resource:F
,conv2d_relu_1_conv2d_readvariableop_resource:;
-conv2d_relu_1_biasadd_readvariableop_resource:F
,conv2d_relu_2_conv2d_readvariableop_resource:;
-conv2d_relu_2_biasadd_readvariableop_resource:F
,conv2d_relu_3_conv2d_readvariableop_resource: ;
-conv2d_relu_3_biasadd_readvariableop_resource: F
,conv2d_relu_4_conv2d_readvariableop_resource:  ;
-conv2d_relu_4_biasadd_readvariableop_resource: 7
$dense_matmul_readvariableop_resource:		@3
%dense_biasadd_readvariableop_resource:@8
&softmax_matmul_readvariableop_resource:@5
'softmax_biasadd_readvariableop_resource:
identity

identity_1Ђ!batch_norm/Reshape/ReadVariableOpЂ#batch_norm/Reshape_1/ReadVariableOpЂ$conv2d_relu_1/BiasAdd/ReadVariableOpЂ#conv2d_relu_1/Conv2D/ReadVariableOpЂ$conv2d_relu_2/BiasAdd/ReadVariableOpЂ#conv2d_relu_2/Conv2D/ReadVariableOpЂ$conv2d_relu_3/BiasAdd/ReadVariableOpЂ#conv2d_relu_3/Conv2D/ReadVariableOpЂ$conv2d_relu_4/BiasAdd/ReadVariableOpЂ#conv2d_relu_4/Conv2D/ReadVariableOpЂ"conv2d_tanh/BiasAdd/ReadVariableOpЂ!conv2d_tanh/Conv2D/ReadVariableOpЂdense/BiasAdd/ReadVariableOpЂdense/MatMul/ReadVariableOpЂsoftmax/BiasAdd/ReadVariableOpЂsoftmax/MatMul/ReadVariableOps
)batch_norm/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:І
batch_norm/moments/meanMeaninputs2batch_norm/moments/mean/reduction_indices:output:0*
T0*/
_output_shapes
:џџџџџџџџџ(*
	keep_dims(
batch_norm/moments/StopGradientStopGradient batch_norm/moments/mean:output:0*
T0*/
_output_shapes
:џџџџџџџџџ(І
$batch_norm/moments/SquaredDifferenceSquaredDifferenceinputs(batch_norm/moments/StopGradient:output:0*
T0*0
_output_shapes
:џџџџџџџџџ(Мw
-batch_norm/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:а
batch_norm/moments/varianceMean(batch_norm/moments/SquaredDifference:z:06batch_norm/moments/variance/reduction_indices:output:0*
T0*/
_output_shapes
:џџџџџџџџџ(*
	keep_dims(
!batch_norm/Reshape/ReadVariableOpReadVariableOp*batch_norm_reshape_readvariableop_resource*
_output_shapes	
:М*
dtype0q
batch_norm/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      М      
batch_norm/ReshapeReshape)batch_norm/Reshape/ReadVariableOp:value:0!batch_norm/Reshape/shape:output:0*
T0*'
_output_shapes
:М
#batch_norm/Reshape_1/ReadVariableOpReadVariableOp,batch_norm_reshape_1_readvariableop_resource*
_output_shapes	
:М*
dtype0s
batch_norm/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      М      Ѓ
batch_norm/Reshape_1Reshape+batch_norm/Reshape_1/ReadVariableOp:value:0#batch_norm/Reshape_1/shape:output:0*
T0*'
_output_shapes
:М_
batch_norm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:І
batch_norm/batchnorm/addAddV2$batch_norm/moments/variance:output:0#batch_norm/batchnorm/add/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ({
batch_norm/batchnorm/RsqrtRsqrtbatch_norm/batchnorm/add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ(
batch_norm/batchnorm/mulMulbatch_norm/batchnorm/Rsqrt:y:0batch_norm/Reshape:output:0*
T0*0
_output_shapes
:џџџџџџџџџ(М
batch_norm/batchnorm/mul_1Mulinputsbatch_norm/batchnorm/mul:z:0*
T0*0
_output_shapes
:џџџџџџџџџ(М
batch_norm/batchnorm/mul_2Mul batch_norm/moments/mean:output:0batch_norm/batchnorm/mul:z:0*
T0*0
_output_shapes
:џџџџџџџџџ(М
batch_norm/batchnorm/subSubbatch_norm/Reshape_1:output:0batch_norm/batchnorm/mul_2:z:0*
T0*0
_output_shapes
:џџџџџџџџџ(М
batch_norm/batchnorm/add_1AddV2batch_norm/batchnorm/mul_1:z:0batch_norm/batchnorm/sub:z:0*
T0*0
_output_shapes
:џџџџџџџџџ(М
!conv2d_tanh/Conv2D/ReadVariableOpReadVariableOp*conv2d_tanh_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ъ
conv2d_tanh/Conv2DConv2Dbatch_norm/batchnorm/add_1:z:0)conv2d_tanh/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ(М*
paddingSAME*
strides

"conv2d_tanh/BiasAdd/ReadVariableOpReadVariableOp+conv2d_tanh_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ђ
conv2d_tanh/BiasAddBiasAddconv2d_tanh/Conv2D:output:0*conv2d_tanh/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ(Мq
conv2d_tanh/TanhTanhconv2d_tanh/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ(МЂ
max_pool_2d_1/MaxPoolMaxPoolconv2d_tanh/Tanh:y:0*/
_output_shapes
:џџџџџџџџџ^*
ksize
*
paddingSAME*
strides

#conv2d_relu_1/Conv2D/ReadVariableOpReadVariableOp,conv2d_relu_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Э
conv2d_relu_1/Conv2DConv2Dmax_pool_2d_1/MaxPool:output:0+conv2d_relu_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ^*
paddingSAME*
strides

$conv2d_relu_1/BiasAdd/ReadVariableOpReadVariableOp-conv2d_relu_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ї
conv2d_relu_1/BiasAddBiasAddconv2d_relu_1/Conv2D:output:0,conv2d_relu_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ^t
conv2d_relu_1/ReluReluconv2d_relu_1/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ^Ў
max_pool_2d_2/MaxPoolMaxPool conv2d_relu_1/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ
/*
ksize
*
paddingSAME*
strides

#conv2d_relu_2/Conv2D/ReadVariableOpReadVariableOp,conv2d_relu_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Э
conv2d_relu_2/Conv2DConv2Dmax_pool_2d_2/MaxPool:output:0+conv2d_relu_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ
/*
paddingSAME*
strides

$conv2d_relu_2/BiasAdd/ReadVariableOpReadVariableOp-conv2d_relu_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ї
conv2d_relu_2/BiasAddBiasAddconv2d_relu_2/Conv2D:output:0,conv2d_relu_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ
/t
conv2d_relu_2/ReluReluconv2d_relu_2/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
/Ў
max_pool_2d_3/MaxPoolMaxPool conv2d_relu_2/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingSAME*
strides

#conv2d_relu_3/Conv2D/ReadVariableOpReadVariableOp,conv2d_relu_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Э
conv2d_relu_3/Conv2DConv2Dmax_pool_2d_3/MaxPool:output:0+conv2d_relu_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingSAME*
strides

$conv2d_relu_3/BiasAdd/ReadVariableOpReadVariableOp-conv2d_relu_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ї
conv2d_relu_3/BiasAddBiasAddconv2d_relu_3/Conv2D:output:0,conv2d_relu_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ t
conv2d_relu_3/ReluReluconv2d_relu_3/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ Ў
max_pool_2d_4/MaxPoolMaxPool conv2d_relu_3/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ *
ksize
*
paddingSAME*
strides

#conv2d_relu_4/Conv2D/ReadVariableOpReadVariableOp,conv2d_relu_4_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0Э
conv2d_relu_4/Conv2DConv2Dmax_pool_2d_4/MaxPool:output:0+conv2d_relu_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingSAME*
strides

$conv2d_relu_4/BiasAdd/ReadVariableOpReadVariableOp-conv2d_relu_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ї
conv2d_relu_4/BiasAddBiasAddconv2d_relu_4/Conv2D:output:0,conv2d_relu_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ t
conv2d_relu_4/ReluReluconv2d_relu_4/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ ^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ  
flatten/ReshapeReshape conv2d_relu_4/Relu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ	i
dropout/IdentityIdentityflatten/Reshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ	
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:		@*
dtype0
dense/MatMulMatMuldropout/Identity:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@e
 dense/ActivityRegularizer/L2LossL2Lossdense/Relu:activations:0*
T0*
_output_shapes
: d
dense/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;
dense/ActivityRegularizer/mulMul(dense/ActivityRegularizer/mul/x:output:0)dense/ActivityRegularizer/L2Loss:output:0*
T0*
_output_shapes
: g
dense/ActivityRegularizer/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:w
-dense/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/dense/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/dense/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:г
'dense/ActivityRegularizer/strided_sliceStridedSlice(dense/ActivityRegularizer/Shape:output:06dense/ActivityRegularizer/strided_slice/stack:output:08dense/ActivityRegularizer/strided_slice/stack_1:output:08dense/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
dense/ActivityRegularizer/CastCast0dense/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 
!dense/ActivityRegularizer/truedivRealDiv!dense/ActivityRegularizer/mul:z:0"dense/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 
softmax/MatMul/ReadVariableOpReadVariableOp&softmax_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
softmax/MatMulMatMuldense/Relu:activations:0%softmax/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
softmax/BiasAdd/ReadVariableOpReadVariableOp'softmax_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
softmax/BiasAddBiasAddsoftmax/MatMul:product:0&softmax/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџf
softmax/SoftmaxSoftmaxsoftmax/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџh
IdentityIdentitysoftmax/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџe

Identity_1Identity%dense/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 
NoOpNoOp"^batch_norm/Reshape/ReadVariableOp$^batch_norm/Reshape_1/ReadVariableOp%^conv2d_relu_1/BiasAdd/ReadVariableOp$^conv2d_relu_1/Conv2D/ReadVariableOp%^conv2d_relu_2/BiasAdd/ReadVariableOp$^conv2d_relu_2/Conv2D/ReadVariableOp%^conv2d_relu_3/BiasAdd/ReadVariableOp$^conv2d_relu_3/Conv2D/ReadVariableOp%^conv2d_relu_4/BiasAdd/ReadVariableOp$^conv2d_relu_4/Conv2D/ReadVariableOp#^conv2d_tanh/BiasAdd/ReadVariableOp"^conv2d_tanh/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^softmax/BiasAdd/ReadVariableOp^softmax/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:џџџџџџџџџ(М: : : : : : : : : : : : : : : : 2F
!batch_norm/Reshape/ReadVariableOp!batch_norm/Reshape/ReadVariableOp2J
#batch_norm/Reshape_1/ReadVariableOp#batch_norm/Reshape_1/ReadVariableOp2L
$conv2d_relu_1/BiasAdd/ReadVariableOp$conv2d_relu_1/BiasAdd/ReadVariableOp2J
#conv2d_relu_1/Conv2D/ReadVariableOp#conv2d_relu_1/Conv2D/ReadVariableOp2L
$conv2d_relu_2/BiasAdd/ReadVariableOp$conv2d_relu_2/BiasAdd/ReadVariableOp2J
#conv2d_relu_2/Conv2D/ReadVariableOp#conv2d_relu_2/Conv2D/ReadVariableOp2L
$conv2d_relu_3/BiasAdd/ReadVariableOp$conv2d_relu_3/BiasAdd/ReadVariableOp2J
#conv2d_relu_3/Conv2D/ReadVariableOp#conv2d_relu_3/Conv2D/ReadVariableOp2L
$conv2d_relu_4/BiasAdd/ReadVariableOp$conv2d_relu_4/BiasAdd/ReadVariableOp2J
#conv2d_relu_4/Conv2D/ReadVariableOp#conv2d_relu_4/Conv2D/ReadVariableOp2H
"conv2d_tanh/BiasAdd/ReadVariableOp"conv2d_tanh/BiasAdd/ReadVariableOp2F
!conv2d_tanh/Conv2D/ReadVariableOp!conv2d_tanh/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
softmax/BiasAdd/ReadVariableOpsoftmax/BiasAdd/ReadVariableOp2>
softmax/MatMul/ReadVariableOpsoftmax/MatMul/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ(М
 
_user_specified_nameinputs
А
ћ
F__inference_batch_norm_layer_call_and_return_conditional_losses_102074

inputs.
reshape_readvariableop_resource:	М0
!reshape_1_readvariableop_resource:	М
identityЂReshape/ReadVariableOpЂReshape_1/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*/
_output_shapes
:џџџџџџџџџ(*
	keep_dims(u
moments/StopGradientStopGradientmoments/mean:output:0*
T0*/
_output_shapes
:џџџџџџџџџ(
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*0
_output_shapes
:џџџџџџџџџ(Мl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:Џ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*/
_output_shapes
:џџџџџџџџџ(*
	keep_dims(s
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes	
:М*
dtype0f
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      М      |
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*'
_output_shapes
:Мw
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes	
:М*
dtype0h
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      М      
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*'
_output_shapes
:МT
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ(e
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ(v
batchnorm/mulMulbatchnorm/Rsqrt:y:0Reshape:output:0*
T0*0
_output_shapes
:џџџџџџџџџ(Мl
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*0
_output_shapes
:џџџџџџџџџ(М{
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*0
_output_shapes
:џџџџџџџџџ(Мx
batchnorm/subSubReshape_1:output:0batchnorm/mul_2:z:0*
T0*0
_output_shapes
:џџџџџџџџџ(М{
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*0
_output_shapes
:џџџџџџџџџ(Мk
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ(Мz
NoOpNoOp^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ(М: : 20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ(М
 
_user_specified_nameinputs
ц
У
/__inference_2d_convolution_layer_call_fn_101313
input_1
unknown:	М
	unknown_0:	М#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7: 
	unknown_8: #
	unknown_9:  

unknown_10: 

unknown_11:		@

unknown_12:@

unknown_13:@

unknown_14:
identityЂStatefulPartitionedCallЁ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:џџџџџџџџџ: *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_2d_convolution_layer_call_and_return_conditional_losses_101277o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:џџџџџџџџџ(М: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:џџџџџџџџџ(М
!
_user_specified_name	input_1
Х
_
C__inference_flatten_layer_call_and_return_conditional_losses_102225

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџ	Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ :W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
ы

+__inference_batch_norm_layer_call_fn_102048

inputs
unknown:	М
	unknown_0:	М
identityЂStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ(М*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_batch_norm_layer_call_and_return_conditional_losses_101123x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ(М`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ(М: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџ(М
 
_user_specified_nameinputs

e
I__inference_max_pool_2d_3_layer_call_and_return_conditional_losses_101067

inputs
identityЁ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ыф
$
"__inference__traced_restore_102678
file_prefix0
!assignvariableop_batch_norm_gamma:	М1
"assignvariableop_1_batch_norm_beta:	М?
%assignvariableop_2_conv2d_tanh_kernel:1
#assignvariableop_3_conv2d_tanh_bias:A
'assignvariableop_4_conv2d_relu_1_kernel:3
%assignvariableop_5_conv2d_relu_1_bias:A
'assignvariableop_6_conv2d_relu_2_kernel:3
%assignvariableop_7_conv2d_relu_2_bias:A
'assignvariableop_8_conv2d_relu_3_kernel: 3
%assignvariableop_9_conv2d_relu_3_bias: B
(assignvariableop_10_conv2d_relu_4_kernel:  4
&assignvariableop_11_conv2d_relu_4_bias: 3
 assignvariableop_12_dense_kernel:		@,
assignvariableop_13_dense_bias:@4
"assignvariableop_14_softmax_kernel:@.
 assignvariableop_15_softmax_bias:'
assignvariableop_16_adam_iter:	 )
assignvariableop_17_adam_beta_1: )
assignvariableop_18_adam_beta_2: (
assignvariableop_19_adam_decay: 0
&assignvariableop_20_adam_learning_rate: %
assignvariableop_21_total_1: %
assignvariableop_22_count_1: #
assignvariableop_23_total: #
assignvariableop_24_count: :
+assignvariableop_25_adam_batch_norm_gamma_m:	М9
*assignvariableop_26_adam_batch_norm_beta_m:	МG
-assignvariableop_27_adam_conv2d_tanh_kernel_m:9
+assignvariableop_28_adam_conv2d_tanh_bias_m:I
/assignvariableop_29_adam_conv2d_relu_1_kernel_m:;
-assignvariableop_30_adam_conv2d_relu_1_bias_m:I
/assignvariableop_31_adam_conv2d_relu_2_kernel_m:;
-assignvariableop_32_adam_conv2d_relu_2_bias_m:I
/assignvariableop_33_adam_conv2d_relu_3_kernel_m: ;
-assignvariableop_34_adam_conv2d_relu_3_bias_m: I
/assignvariableop_35_adam_conv2d_relu_4_kernel_m:  ;
-assignvariableop_36_adam_conv2d_relu_4_bias_m: :
'assignvariableop_37_adam_dense_kernel_m:		@3
%assignvariableop_38_adam_dense_bias_m:@;
)assignvariableop_39_adam_softmax_kernel_m:@5
'assignvariableop_40_adam_softmax_bias_m::
+assignvariableop_41_adam_batch_norm_gamma_v:	М9
*assignvariableop_42_adam_batch_norm_beta_v:	МG
-assignvariableop_43_adam_conv2d_tanh_kernel_v:9
+assignvariableop_44_adam_conv2d_tanh_bias_v:I
/assignvariableop_45_adam_conv2d_relu_1_kernel_v:;
-assignvariableop_46_adam_conv2d_relu_1_bias_v:I
/assignvariableop_47_adam_conv2d_relu_2_kernel_v:;
-assignvariableop_48_adam_conv2d_relu_2_bias_v:I
/assignvariableop_49_adam_conv2d_relu_3_kernel_v: ;
-assignvariableop_50_adam_conv2d_relu_3_bias_v: I
/assignvariableop_51_adam_conv2d_relu_4_kernel_v:  ;
-assignvariableop_52_adam_conv2d_relu_4_bias_v: :
'assignvariableop_53_adam_dense_kernel_v:		@3
%assignvariableop_54_adam_dense_bias_v:@;
)assignvariableop_55_adam_softmax_kernel_v:@5
'assignvariableop_56_adam_softmax_bias_v:
identity_58ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_30ЂAssignVariableOp_31ЂAssignVariableOp_32ЂAssignVariableOp_33ЂAssignVariableOp_34ЂAssignVariableOp_35ЂAssignVariableOp_36ЂAssignVariableOp_37ЂAssignVariableOp_38ЂAssignVariableOp_39ЂAssignVariableOp_4ЂAssignVariableOp_40ЂAssignVariableOp_41ЂAssignVariableOp_42ЂAssignVariableOp_43ЂAssignVariableOp_44ЂAssignVariableOp_45ЂAssignVariableOp_46ЂAssignVariableOp_47ЂAssignVariableOp_48ЂAssignVariableOp_49ЂAssignVariableOp_5ЂAssignVariableOp_50ЂAssignVariableOp_51ЂAssignVariableOp_52ЂAssignVariableOp_53ЂAssignVariableOp_54ЂAssignVariableOp_55ЂAssignVariableOp_56ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9 
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
::*
dtype0*Н
valueГBА:B5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHх
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
::*
dtype0*
value~B|:B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B У
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ў
_output_shapesы
ш::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*H
dtypes>
<2:	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp!assignvariableop_batch_norm_gammaIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp"assignvariableop_1_batch_norm_betaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp%assignvariableop_2_conv2d_tanh_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp#assignvariableop_3_conv2d_tanh_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp'assignvariableop_4_conv2d_relu_1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp%assignvariableop_5_conv2d_relu_1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp'assignvariableop_6_conv2d_relu_2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp%assignvariableop_7_conv2d_relu_2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp'assignvariableop_8_conv2d_relu_3_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp%assignvariableop_9_conv2d_relu_3_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp(assignvariableop_10_conv2d_relu_4_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp&assignvariableop_11_conv2d_relu_4_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp assignvariableop_12_dense_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_dense_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp"assignvariableop_14_softmax_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp assignvariableop_15_softmax_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_16AssignVariableOpassignvariableop_16_adam_iterIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOpassignvariableop_17_adam_beta_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOpassignvariableop_18_adam_beta_2Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOpassignvariableop_19_adam_decayIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp&assignvariableop_20_adam_learning_rateIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOpassignvariableop_21_total_1Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOpassignvariableop_22_count_1Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOpassignvariableop_23_totalIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOpassignvariableop_24_countIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp+assignvariableop_25_adam_batch_norm_gamma_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp*assignvariableop_26_adam_batch_norm_beta_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp-assignvariableop_27_adam_conv2d_tanh_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp+assignvariableop_28_adam_conv2d_tanh_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_29AssignVariableOp/assignvariableop_29_adam_conv2d_relu_1_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp-assignvariableop_30_adam_conv2d_relu_1_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_31AssignVariableOp/assignvariableop_31_adam_conv2d_relu_2_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp-assignvariableop_32_adam_conv2d_relu_2_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_33AssignVariableOp/assignvariableop_33_adam_conv2d_relu_3_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp-assignvariableop_34_adam_conv2d_relu_3_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_35AssignVariableOp/assignvariableop_35_adam_conv2d_relu_4_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_36AssignVariableOp-assignvariableop_36_adam_conv2d_relu_4_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_37AssignVariableOp'assignvariableop_37_adam_dense_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_38AssignVariableOp%assignvariableop_38_adam_dense_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_39AssignVariableOp)assignvariableop_39_adam_softmax_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_40AssignVariableOp'assignvariableop_40_adam_softmax_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_41AssignVariableOp+assignvariableop_41_adam_batch_norm_gamma_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_42AssignVariableOp*assignvariableop_42_adam_batch_norm_beta_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_43AssignVariableOp-assignvariableop_43_adam_conv2d_tanh_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_44AssignVariableOp+assignvariableop_44_adam_conv2d_tanh_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_45AssignVariableOp/assignvariableop_45_adam_conv2d_relu_1_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_46AssignVariableOp-assignvariableop_46_adam_conv2d_relu_1_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_47AssignVariableOp/assignvariableop_47_adam_conv2d_relu_2_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_48AssignVariableOp-assignvariableop_48_adam_conv2d_relu_2_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_49AssignVariableOp/assignvariableop_49_adam_conv2d_relu_3_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_50AssignVariableOp-assignvariableop_50_adam_conv2d_relu_3_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_51AssignVariableOp/assignvariableop_51_adam_conv2d_relu_4_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_52AssignVariableOp-assignvariableop_52_adam_conv2d_relu_4_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_53AssignVariableOp'assignvariableop_53_adam_dense_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_54AssignVariableOp%assignvariableop_54_adam_dense_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_55AssignVariableOp)assignvariableop_55_adam_softmax_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_56AssignVariableOp'assignvariableop_56_adam_softmax_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 Е

Identity_57Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_58IdentityIdentity_57:output:0^NoOp_1*
T0*
_output_shapes
: Ђ

NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_58Identity_58:output:0*
_input_shapesv
t: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix


ѓ
A__inference_dense_layer_call_and_return_conditional_losses_101244

inputs1
matmul_readvariableop_resource:		@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:		@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ	
 
_user_specified_nameinputs
z
Њ
J__inference_2d_convolution_layer_call_and_return_conditional_losses_102039

inputs9
*batch_norm_reshape_readvariableop_resource:	М;
,batch_norm_reshape_1_readvariableop_resource:	МD
*conv2d_tanh_conv2d_readvariableop_resource:9
+conv2d_tanh_biasadd_readvariableop_resource:F
,conv2d_relu_1_conv2d_readvariableop_resource:;
-conv2d_relu_1_biasadd_readvariableop_resource:F
,conv2d_relu_2_conv2d_readvariableop_resource:;
-conv2d_relu_2_biasadd_readvariableop_resource:F
,conv2d_relu_3_conv2d_readvariableop_resource: ;
-conv2d_relu_3_biasadd_readvariableop_resource: F
,conv2d_relu_4_conv2d_readvariableop_resource:  ;
-conv2d_relu_4_biasadd_readvariableop_resource: 7
$dense_matmul_readvariableop_resource:		@3
%dense_biasadd_readvariableop_resource:@8
&softmax_matmul_readvariableop_resource:@5
'softmax_biasadd_readvariableop_resource:
identity

identity_1Ђ!batch_norm/Reshape/ReadVariableOpЂ#batch_norm/Reshape_1/ReadVariableOpЂ$conv2d_relu_1/BiasAdd/ReadVariableOpЂ#conv2d_relu_1/Conv2D/ReadVariableOpЂ$conv2d_relu_2/BiasAdd/ReadVariableOpЂ#conv2d_relu_2/Conv2D/ReadVariableOpЂ$conv2d_relu_3/BiasAdd/ReadVariableOpЂ#conv2d_relu_3/Conv2D/ReadVariableOpЂ$conv2d_relu_4/BiasAdd/ReadVariableOpЂ#conv2d_relu_4/Conv2D/ReadVariableOpЂ"conv2d_tanh/BiasAdd/ReadVariableOpЂ!conv2d_tanh/Conv2D/ReadVariableOpЂdense/BiasAdd/ReadVariableOpЂdense/MatMul/ReadVariableOpЂsoftmax/BiasAdd/ReadVariableOpЂsoftmax/MatMul/ReadVariableOps
)batch_norm/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:І
batch_norm/moments/meanMeaninputs2batch_norm/moments/mean/reduction_indices:output:0*
T0*/
_output_shapes
:џџџџџџџџџ(*
	keep_dims(
batch_norm/moments/StopGradientStopGradient batch_norm/moments/mean:output:0*
T0*/
_output_shapes
:џџџџџџџџџ(І
$batch_norm/moments/SquaredDifferenceSquaredDifferenceinputs(batch_norm/moments/StopGradient:output:0*
T0*0
_output_shapes
:џџџџџџџџџ(Мw
-batch_norm/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:а
batch_norm/moments/varianceMean(batch_norm/moments/SquaredDifference:z:06batch_norm/moments/variance/reduction_indices:output:0*
T0*/
_output_shapes
:џџџџџџџџџ(*
	keep_dims(
!batch_norm/Reshape/ReadVariableOpReadVariableOp*batch_norm_reshape_readvariableop_resource*
_output_shapes	
:М*
dtype0q
batch_norm/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      М      
batch_norm/ReshapeReshape)batch_norm/Reshape/ReadVariableOp:value:0!batch_norm/Reshape/shape:output:0*
T0*'
_output_shapes
:М
#batch_norm/Reshape_1/ReadVariableOpReadVariableOp,batch_norm_reshape_1_readvariableop_resource*
_output_shapes	
:М*
dtype0s
batch_norm/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      М      Ѓ
batch_norm/Reshape_1Reshape+batch_norm/Reshape_1/ReadVariableOp:value:0#batch_norm/Reshape_1/shape:output:0*
T0*'
_output_shapes
:М_
batch_norm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:І
batch_norm/batchnorm/addAddV2$batch_norm/moments/variance:output:0#batch_norm/batchnorm/add/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ({
batch_norm/batchnorm/RsqrtRsqrtbatch_norm/batchnorm/add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ(
batch_norm/batchnorm/mulMulbatch_norm/batchnorm/Rsqrt:y:0batch_norm/Reshape:output:0*
T0*0
_output_shapes
:џџџџџџџџџ(М
batch_norm/batchnorm/mul_1Mulinputsbatch_norm/batchnorm/mul:z:0*
T0*0
_output_shapes
:џџџџџџџџџ(М
batch_norm/batchnorm/mul_2Mul batch_norm/moments/mean:output:0batch_norm/batchnorm/mul:z:0*
T0*0
_output_shapes
:џџџџџџџџџ(М
batch_norm/batchnorm/subSubbatch_norm/Reshape_1:output:0batch_norm/batchnorm/mul_2:z:0*
T0*0
_output_shapes
:џџџџџџџџџ(М
batch_norm/batchnorm/add_1AddV2batch_norm/batchnorm/mul_1:z:0batch_norm/batchnorm/sub:z:0*
T0*0
_output_shapes
:џџџџџџџџџ(М
!conv2d_tanh/Conv2D/ReadVariableOpReadVariableOp*conv2d_tanh_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ъ
conv2d_tanh/Conv2DConv2Dbatch_norm/batchnorm/add_1:z:0)conv2d_tanh/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ(М*
paddingSAME*
strides

"conv2d_tanh/BiasAdd/ReadVariableOpReadVariableOp+conv2d_tanh_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ђ
conv2d_tanh/BiasAddBiasAddconv2d_tanh/Conv2D:output:0*conv2d_tanh/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ(Мq
conv2d_tanh/TanhTanhconv2d_tanh/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ(МЂ
max_pool_2d_1/MaxPoolMaxPoolconv2d_tanh/Tanh:y:0*/
_output_shapes
:џџџџџџџџџ^*
ksize
*
paddingSAME*
strides

#conv2d_relu_1/Conv2D/ReadVariableOpReadVariableOp,conv2d_relu_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Э
conv2d_relu_1/Conv2DConv2Dmax_pool_2d_1/MaxPool:output:0+conv2d_relu_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ^*
paddingSAME*
strides

$conv2d_relu_1/BiasAdd/ReadVariableOpReadVariableOp-conv2d_relu_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ї
conv2d_relu_1/BiasAddBiasAddconv2d_relu_1/Conv2D:output:0,conv2d_relu_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ^t
conv2d_relu_1/ReluReluconv2d_relu_1/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ^Ў
max_pool_2d_2/MaxPoolMaxPool conv2d_relu_1/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ
/*
ksize
*
paddingSAME*
strides

#conv2d_relu_2/Conv2D/ReadVariableOpReadVariableOp,conv2d_relu_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Э
conv2d_relu_2/Conv2DConv2Dmax_pool_2d_2/MaxPool:output:0+conv2d_relu_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ
/*
paddingSAME*
strides

$conv2d_relu_2/BiasAdd/ReadVariableOpReadVariableOp-conv2d_relu_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ї
conv2d_relu_2/BiasAddBiasAddconv2d_relu_2/Conv2D:output:0,conv2d_relu_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ
/t
conv2d_relu_2/ReluReluconv2d_relu_2/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
/Ў
max_pool_2d_3/MaxPoolMaxPool conv2d_relu_2/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingSAME*
strides

#conv2d_relu_3/Conv2D/ReadVariableOpReadVariableOp,conv2d_relu_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Э
conv2d_relu_3/Conv2DConv2Dmax_pool_2d_3/MaxPool:output:0+conv2d_relu_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingSAME*
strides

$conv2d_relu_3/BiasAdd/ReadVariableOpReadVariableOp-conv2d_relu_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ї
conv2d_relu_3/BiasAddBiasAddconv2d_relu_3/Conv2D:output:0,conv2d_relu_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ t
conv2d_relu_3/ReluReluconv2d_relu_3/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ Ў
max_pool_2d_4/MaxPoolMaxPool conv2d_relu_3/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ *
ksize
*
paddingSAME*
strides

#conv2d_relu_4/Conv2D/ReadVariableOpReadVariableOp,conv2d_relu_4_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0Э
conv2d_relu_4/Conv2DConv2Dmax_pool_2d_4/MaxPool:output:0+conv2d_relu_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingSAME*
strides

$conv2d_relu_4/BiasAdd/ReadVariableOpReadVariableOp-conv2d_relu_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ї
conv2d_relu_4/BiasAddBiasAddconv2d_relu_4/Conv2D:output:0,conv2d_relu_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ t
conv2d_relu_4/ReluReluconv2d_relu_4/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ ^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ  
flatten/ReshapeReshape conv2d_relu_4/Relu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ	Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
dropout/dropout/MulMulflatten/Reshape:output:0dropout/dropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ	]
dropout/dropout/ShapeShapeflatten/Reshape:output:0*
T0*
_output_shapes
:
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ	*
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>П
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџ	
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџ	
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџ	
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:		@*
dtype0
dense/MatMulMatMuldropout/dropout/Mul_1:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@e
 dense/ActivityRegularizer/L2LossL2Lossdense/Relu:activations:0*
T0*
_output_shapes
: d
dense/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;
dense/ActivityRegularizer/mulMul(dense/ActivityRegularizer/mul/x:output:0)dense/ActivityRegularizer/L2Loss:output:0*
T0*
_output_shapes
: g
dense/ActivityRegularizer/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:w
-dense/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/dense/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/dense/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:г
'dense/ActivityRegularizer/strided_sliceStridedSlice(dense/ActivityRegularizer/Shape:output:06dense/ActivityRegularizer/strided_slice/stack:output:08dense/ActivityRegularizer/strided_slice/stack_1:output:08dense/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
dense/ActivityRegularizer/CastCast0dense/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 
!dense/ActivityRegularizer/truedivRealDiv!dense/ActivityRegularizer/mul:z:0"dense/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 
softmax/MatMul/ReadVariableOpReadVariableOp&softmax_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0
softmax/MatMulMatMuldense/Relu:activations:0%softmax/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
softmax/BiasAdd/ReadVariableOpReadVariableOp'softmax_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
softmax/BiasAddBiasAddsoftmax/MatMul:product:0&softmax/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџf
softmax/SoftmaxSoftmaxsoftmax/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџh
IdentityIdentitysoftmax/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџe

Identity_1Identity%dense/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 
NoOpNoOp"^batch_norm/Reshape/ReadVariableOp$^batch_norm/Reshape_1/ReadVariableOp%^conv2d_relu_1/BiasAdd/ReadVariableOp$^conv2d_relu_1/Conv2D/ReadVariableOp%^conv2d_relu_2/BiasAdd/ReadVariableOp$^conv2d_relu_2/Conv2D/ReadVariableOp%^conv2d_relu_3/BiasAdd/ReadVariableOp$^conv2d_relu_3/Conv2D/ReadVariableOp%^conv2d_relu_4/BiasAdd/ReadVariableOp$^conv2d_relu_4/Conv2D/ReadVariableOp#^conv2d_tanh/BiasAdd/ReadVariableOp"^conv2d_tanh/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^softmax/BiasAdd/ReadVariableOp^softmax/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:џџџџџџџџџ(М: : : : : : : : : : : : : : : : 2F
!batch_norm/Reshape/ReadVariableOp!batch_norm/Reshape/ReadVariableOp2J
#batch_norm/Reshape_1/ReadVariableOp#batch_norm/Reshape_1/ReadVariableOp2L
$conv2d_relu_1/BiasAdd/ReadVariableOp$conv2d_relu_1/BiasAdd/ReadVariableOp2J
#conv2d_relu_1/Conv2D/ReadVariableOp#conv2d_relu_1/Conv2D/ReadVariableOp2L
$conv2d_relu_2/BiasAdd/ReadVariableOp$conv2d_relu_2/BiasAdd/ReadVariableOp2J
#conv2d_relu_2/Conv2D/ReadVariableOp#conv2d_relu_2/Conv2D/ReadVariableOp2L
$conv2d_relu_3/BiasAdd/ReadVariableOp$conv2d_relu_3/BiasAdd/ReadVariableOp2J
#conv2d_relu_3/Conv2D/ReadVariableOp#conv2d_relu_3/Conv2D/ReadVariableOp2L
$conv2d_relu_4/BiasAdd/ReadVariableOp$conv2d_relu_4/BiasAdd/ReadVariableOp2J
#conv2d_relu_4/Conv2D/ReadVariableOp#conv2d_relu_4/Conv2D/ReadVariableOp2H
"conv2d_tanh/BiasAdd/ReadVariableOp"conv2d_tanh/BiasAdd/ReadVariableOp2F
!conv2d_tanh/Conv2D/ReadVariableOp!conv2d_tanh/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
softmax/BiasAdd/ReadVariableOpsoftmax/BiasAdd/ReadVariableOp2>
softmax/MatMul/ReadVariableOpsoftmax/MatMul/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ(М
 
_user_specified_nameinputs

У
E__inference_dense_layer_call_and_return_all_conditional_losses_102272

inputs
unknown:		@
	unknown_0:@
identity

identity_1ЂStatefulPartitionedCallй
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_101244Є
PartitionedCallPartitionedCall StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *6
f1R/
-__inference_dense_activity_regularizer_101090o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@X

Identity_1IdentityPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ	: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ	
 
_user_specified_nameinputs
љ	
b
C__inference_dropout_layer_call_and_return_conditional_losses_102252

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ	C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ	*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>Ї
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџ	p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџ	j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџ	Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџ	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ	:P L
(
_output_shapes
:џџџџџџџџџ	
 
_user_specified_nameinputs
З
J
.__inference_max_pool_2d_4_layer_call_fn_102189

inputs
identityк
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_max_pool_2d_4_layer_call_and_return_conditional_losses_101079
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
А
D
(__inference_flatten_layer_call_fn_102219

inputs
identityВ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_101224a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџ	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ :W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
љ	
b
C__inference_dropout_layer_call_and_return_conditional_losses_101365

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ	C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ	*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬL>Ї
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџ	p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџ	j
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџ	Z
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџ	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ	:P L
(
_output_shapes
:џџџџџџџџџ	
 
_user_specified_nameinputs


I__inference_conv2d_relu_1_layer_call_and_return_conditional_losses_101158

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ^*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ^X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ^i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ^w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ^: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ^
 
_user_specified_nameinputs
ЖJ

J__inference_2d_convolution_layer_call_and_return_conditional_losses_101277

inputs 
batch_norm_101124:	М 
batch_norm_101126:	М,
conv2d_tanh_101141: 
conv2d_tanh_101143:.
conv2d_relu_1_101159:"
conv2d_relu_1_101161:.
conv2d_relu_2_101177:"
conv2d_relu_2_101179:.
conv2d_relu_3_101195: "
conv2d_relu_3_101197: .
conv2d_relu_4_101213:  "
conv2d_relu_4_101215: 
dense_101245:		@
dense_101247:@ 
softmax_101270:@
softmax_101272:
identity

identity_1Ђ"batch_norm/StatefulPartitionedCallЂ%conv2d_relu_1/StatefulPartitionedCallЂ%conv2d_relu_2/StatefulPartitionedCallЂ%conv2d_relu_3/StatefulPartitionedCallЂ%conv2d_relu_4/StatefulPartitionedCallЂ#conv2d_tanh/StatefulPartitionedCallЂdense/StatefulPartitionedCallЂsoftmax/StatefulPartitionedCall
"batch_norm/StatefulPartitionedCallStatefulPartitionedCallinputsbatch_norm_101124batch_norm_101126*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ(М*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_batch_norm_layer_call_and_return_conditional_losses_101123­
#conv2d_tanh/StatefulPartitionedCallStatefulPartitionedCall+batch_norm/StatefulPartitionedCall:output:0conv2d_tanh_101141conv2d_tanh_101143*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ(М*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv2d_tanh_layer_call_and_return_conditional_losses_101140ѓ
max_pool_2d_1/PartitionedCallPartitionedCall,conv2d_tanh/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ^* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_max_pool_2d_1_layer_call_and_return_conditional_losses_101043Џ
%conv2d_relu_1/StatefulPartitionedCallStatefulPartitionedCall&max_pool_2d_1/PartitionedCall:output:0conv2d_relu_1_101159conv2d_relu_1_101161*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ^*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv2d_relu_1_layer_call_and_return_conditional_losses_101158ѕ
max_pool_2d_2/PartitionedCallPartitionedCall.conv2d_relu_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ
/* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_max_pool_2d_2_layer_call_and_return_conditional_losses_101055Џ
%conv2d_relu_2/StatefulPartitionedCallStatefulPartitionedCall&max_pool_2d_2/PartitionedCall:output:0conv2d_relu_2_101177conv2d_relu_2_101179*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ
/*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv2d_relu_2_layer_call_and_return_conditional_losses_101176ѕ
max_pool_2d_3/PartitionedCallPartitionedCall.conv2d_relu_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_max_pool_2d_3_layer_call_and_return_conditional_losses_101067Џ
%conv2d_relu_3/StatefulPartitionedCallStatefulPartitionedCall&max_pool_2d_3/PartitionedCall:output:0conv2d_relu_3_101195conv2d_relu_3_101197*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv2d_relu_3_layer_call_and_return_conditional_losses_101194ѕ
max_pool_2d_4/PartitionedCallPartitionedCall.conv2d_relu_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_max_pool_2d_4_layer_call_and_return_conditional_losses_101079Џ
%conv2d_relu_4/StatefulPartitionedCallStatefulPartitionedCall&max_pool_2d_4/PartitionedCall:output:0conv2d_relu_4_101213conv2d_relu_4_101215*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv2d_relu_4_layer_call_and_return_conditional_losses_101212т
flatten/PartitionedCallPartitionedCall.conv2d_relu_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_101224д
dropout/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_101231
dense/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_101245dense_101247*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_101244Ф
)dense/ActivityRegularizer/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *6
f1R/
-__inference_dense_activity_regularizer_101090u
dense/ActivityRegularizer/ShapeShape&dense/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:w
-dense/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/dense/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/dense/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:г
'dense/ActivityRegularizer/strided_sliceStridedSlice(dense/ActivityRegularizer/Shape:output:06dense/ActivityRegularizer/strided_slice/stack:output:08dense/ActivityRegularizer/strided_slice/stack_1:output:08dense/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
dense/ActivityRegularizer/CastCast0dense/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: Ѕ
!dense/ActivityRegularizer/truedivRealDiv2dense/ActivityRegularizer/PartitionedCall:output:0"dense/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 
softmax/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0softmax_101270softmax_101272*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_softmax_layer_call_and_return_conditional_losses_101269w
IdentityIdentity(softmax/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџe

Identity_1Identity%dense/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: ѓ
NoOpNoOp#^batch_norm/StatefulPartitionedCall&^conv2d_relu_1/StatefulPartitionedCall&^conv2d_relu_2/StatefulPartitionedCall&^conv2d_relu_3/StatefulPartitionedCall&^conv2d_relu_4/StatefulPartitionedCall$^conv2d_tanh/StatefulPartitionedCall^dense/StatefulPartitionedCall ^softmax/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:џџџџџџџџџ(М: : : : : : : : : : : : : : : : 2H
"batch_norm/StatefulPartitionedCall"batch_norm/StatefulPartitionedCall2N
%conv2d_relu_1/StatefulPartitionedCall%conv2d_relu_1/StatefulPartitionedCall2N
%conv2d_relu_2/StatefulPartitionedCall%conv2d_relu_2/StatefulPartitionedCall2N
%conv2d_relu_3/StatefulPartitionedCall%conv2d_relu_3/StatefulPartitionedCall2N
%conv2d_relu_4/StatefulPartitionedCall%conv2d_relu_4/StatefulPartitionedCall2J
#conv2d_tanh/StatefulPartitionedCall#conv2d_tanh/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
softmax/StatefulPartitionedCallsoftmax/StatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџ(М
 
_user_specified_nameinputs

e
I__inference_max_pool_2d_1_layer_call_and_return_conditional_losses_101043

inputs
identityЁ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
у
Т
/__inference_2d_convolution_layer_call_fn_101808

inputs
unknown:	М
	unknown_0:	М#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7: 
	unknown_8: #
	unknown_9:  

unknown_10: 

unknown_11:		@

unknown_12:@

unknown_13:@

unknown_14:
identityЂStatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:џџџџџџџџџ: *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_2d_convolution_layer_call_and_return_conditional_losses_101277o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:џџџџџџџџџ(М: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџ(М
 
_user_specified_nameinputs


I__inference_conv2d_relu_2_layer_call_and_return_conditional_losses_101176

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ
/*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ
/X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
/i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ
/w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ
/: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ
/
 
_user_specified_nameinputs
у
Т
/__inference_2d_convolution_layer_call_fn_101846

inputs
unknown:	М
	unknown_0:	М#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7: 
	unknown_8: #
	unknown_9:  

unknown_10: 

unknown_11:		@

unknown_12:@

unknown_13:@

unknown_14:
identityЂStatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:џџџџџџџџџ: *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_2d_convolution_layer_call_and_return_conditional_losses_101533o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:џџџџџџџџџ(М: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџ(М
 
_user_specified_nameinputs
ї
Ѓ
.__inference_conv2d_relu_3_layer_call_fn_102173

inputs!
unknown: 
	unknown_0: 
identityЂStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv2d_relu_3_layer_call_and_return_conditional_losses_101194w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


G__inference_conv2d_tanh_layer_call_and_return_conditional_losses_101140

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ(М*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ(МY
TanhTanhBiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ(М`
IdentityIdentityTanh:y:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ(Мw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ(М: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ(М
 
_user_specified_nameinputs


I__inference_conv2d_relu_4_layer_call_and_return_conditional_losses_102214

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Ђ
D
(__inference_dropout_layer_call_fn_102230

inputs
identityВ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_101231a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџ	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ	:P L
(
_output_shapes
:џџџџџџџџџ	
 
_user_specified_nameinputs
ї
Ѓ
.__inference_conv2d_relu_1_layer_call_fn_102113

inputs!
unknown:
	unknown_0:
identityЂStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ^*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv2d_relu_1_layer_call_and_return_conditional_losses_101158w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ^`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ^: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ^
 
_user_specified_nameinputs


G__inference_conv2d_tanh_layer_call_and_return_conditional_losses_102094

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ(М*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ(МY
TanhTanhBiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ(М`
IdentityIdentityTanh:y:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ(Мw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ(М: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ(М
 
_user_specified_nameinputs


є
C__inference_softmax_layer_call_and_return_conditional_losses_102303

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
У

(__inference_softmax_layer_call_fn_102292

inputs
unknown:@
	unknown_0:
identityЂStatefulPartitionedCallл
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_softmax_layer_call_and_return_conditional_losses_101269o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
к
a
C__inference_dropout_layer_call_and_return_conditional_losses_101231

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:џџџџџџџџџ	\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:џџџџџџџџџ	"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ	:P L
(
_output_shapes
:џџџџџџџџџ	
 
_user_specified_nameinputs
ї
Ѓ
.__inference_conv2d_relu_4_layer_call_fn_102203

inputs!
unknown:  
	unknown_0: 
identityЂStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv2d_relu_4_layer_call_and_return_conditional_losses_101212w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
ѕ
D
-__inference_dense_activity_regularizer_101090
x
identity4
L2LossL2Lossx*
T0*
_output_shapes
: J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;L
mulMulmul/x:output:0L2Loss:output:0*
T0*
_output_shapes
: >
IdentityIdentitymul:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
::; 7

_output_shapes
:

_user_specified_namex


ѓ
A__inference_dense_layer_call_and_return_conditional_losses_102283

inputs1
matmul_readvariableop_resource:		@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:		@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ	
 
_user_specified_nameinputs


I__inference_conv2d_relu_4_layer_call_and_return_conditional_losses_101212

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
З
J
.__inference_max_pool_2d_3_layer_call_fn_102159

inputs
identityк
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_max_pool_2d_3_layer_call_and_return_conditional_losses_101067
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

e
I__inference_max_pool_2d_4_layer_call_and_return_conditional_losses_101079

inputs
identityЁ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs


I__inference_conv2d_relu_3_layer_call_and_return_conditional_losses_102184

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
З
J
.__inference_max_pool_2d_1_layer_call_fn_102099

inputs
identityк
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_max_pool_2d_1_layer_call_and_return_conditional_losses_101043
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

e
I__inference_max_pool_2d_2_layer_call_and_return_conditional_losses_102134

inputs
identityЁ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

e
I__inference_max_pool_2d_1_layer_call_and_return_conditional_losses_102104

inputs
identityЁ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
йK
Њ
J__inference_2d_convolution_layer_call_and_return_conditional_losses_101725
input_1 
batch_norm_101669:	М 
batch_norm_101671:	М,
conv2d_tanh_101674: 
conv2d_tanh_101676:.
conv2d_relu_1_101680:"
conv2d_relu_1_101682:.
conv2d_relu_2_101686:"
conv2d_relu_2_101688:.
conv2d_relu_3_101692: "
conv2d_relu_3_101694: .
conv2d_relu_4_101698:  "
conv2d_relu_4_101700: 
dense_101705:		@
dense_101707:@ 
softmax_101718:@
softmax_101720:
identity

identity_1Ђ"batch_norm/StatefulPartitionedCallЂ%conv2d_relu_1/StatefulPartitionedCallЂ%conv2d_relu_2/StatefulPartitionedCallЂ%conv2d_relu_3/StatefulPartitionedCallЂ%conv2d_relu_4/StatefulPartitionedCallЂ#conv2d_tanh/StatefulPartitionedCallЂdense/StatefulPartitionedCallЂdropout/StatefulPartitionedCallЂsoftmax/StatefulPartitionedCall
"batch_norm/StatefulPartitionedCallStatefulPartitionedCallinput_1batch_norm_101669batch_norm_101671*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ(М*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_batch_norm_layer_call_and_return_conditional_losses_101123­
#conv2d_tanh/StatefulPartitionedCallStatefulPartitionedCall+batch_norm/StatefulPartitionedCall:output:0conv2d_tanh_101674conv2d_tanh_101676*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ(М*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv2d_tanh_layer_call_and_return_conditional_losses_101140ѓ
max_pool_2d_1/PartitionedCallPartitionedCall,conv2d_tanh/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ^* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_max_pool_2d_1_layer_call_and_return_conditional_losses_101043Џ
%conv2d_relu_1/StatefulPartitionedCallStatefulPartitionedCall&max_pool_2d_1/PartitionedCall:output:0conv2d_relu_1_101680conv2d_relu_1_101682*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ^*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv2d_relu_1_layer_call_and_return_conditional_losses_101158ѕ
max_pool_2d_2/PartitionedCallPartitionedCall.conv2d_relu_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ
/* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_max_pool_2d_2_layer_call_and_return_conditional_losses_101055Џ
%conv2d_relu_2/StatefulPartitionedCallStatefulPartitionedCall&max_pool_2d_2/PartitionedCall:output:0conv2d_relu_2_101686conv2d_relu_2_101688*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ
/*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv2d_relu_2_layer_call_and_return_conditional_losses_101176ѕ
max_pool_2d_3/PartitionedCallPartitionedCall.conv2d_relu_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_max_pool_2d_3_layer_call_and_return_conditional_losses_101067Џ
%conv2d_relu_3/StatefulPartitionedCallStatefulPartitionedCall&max_pool_2d_3/PartitionedCall:output:0conv2d_relu_3_101692conv2d_relu_3_101694*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv2d_relu_3_layer_call_and_return_conditional_losses_101194ѕ
max_pool_2d_4/PartitionedCallPartitionedCall.conv2d_relu_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_max_pool_2d_4_layer_call_and_return_conditional_losses_101079Џ
%conv2d_relu_4/StatefulPartitionedCallStatefulPartitionedCall&max_pool_2d_4/PartitionedCall:output:0conv2d_relu_4_101698conv2d_relu_4_101700*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv2d_relu_4_layer_call_and_return_conditional_losses_101212т
flatten/PartitionedCallPartitionedCall.conv2d_relu_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_101224ф
dropout/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_101365
dense/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_101705dense_101707*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_101244Ф
)dense/ActivityRegularizer/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *6
f1R/
-__inference_dense_activity_regularizer_101090u
dense/ActivityRegularizer/ShapeShape&dense/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:w
-dense/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/dense/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/dense/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:г
'dense/ActivityRegularizer/strided_sliceStridedSlice(dense/ActivityRegularizer/Shape:output:06dense/ActivityRegularizer/strided_slice/stack:output:08dense/ActivityRegularizer/strided_slice/stack_1:output:08dense/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
dense/ActivityRegularizer/CastCast0dense/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: Ѕ
!dense/ActivityRegularizer/truedivRealDiv2dense/ActivityRegularizer/PartitionedCall:output:0"dense/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 
softmax/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0softmax_101718softmax_101720*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_softmax_layer_call_and_return_conditional_losses_101269w
IdentityIdentity(softmax/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџe

Identity_1Identity%dense/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 
NoOpNoOp#^batch_norm/StatefulPartitionedCall&^conv2d_relu_1/StatefulPartitionedCall&^conv2d_relu_2/StatefulPartitionedCall&^conv2d_relu_3/StatefulPartitionedCall&^conv2d_relu_4/StatefulPartitionedCall$^conv2d_tanh/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dropout/StatefulPartitionedCall ^softmax/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:џџџџџџџџџ(М: : : : : : : : : : : : : : : : 2H
"batch_norm/StatefulPartitionedCall"batch_norm/StatefulPartitionedCall2N
%conv2d_relu_1/StatefulPartitionedCall%conv2d_relu_1/StatefulPartitionedCall2N
%conv2d_relu_2/StatefulPartitionedCall%conv2d_relu_2/StatefulPartitionedCall2N
%conv2d_relu_3/StatefulPartitionedCall%conv2d_relu_3/StatefulPartitionedCall2N
%conv2d_relu_4/StatefulPartitionedCall%conv2d_relu_4/StatefulPartitionedCall2J
#conv2d_tanh/StatefulPartitionedCall#conv2d_tanh/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2B
softmax/StatefulPartitionedCallsoftmax/StatefulPartitionedCall:Y U
0
_output_shapes
:џџџџџџџџџ(М
!
_user_specified_name	input_1


I__inference_conv2d_relu_2_layer_call_and_return_conditional_losses_102154

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ
/*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ
/X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
/i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ
/w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ
/: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ
/
 
_user_specified_nameinputs
s

__inference__traced_save_102497
file_prefix/
+savev2_batch_norm_gamma_read_readvariableop.
*savev2_batch_norm_beta_read_readvariableop1
-savev2_conv2d_tanh_kernel_read_readvariableop/
+savev2_conv2d_tanh_bias_read_readvariableop3
/savev2_conv2d_relu_1_kernel_read_readvariableop1
-savev2_conv2d_relu_1_bias_read_readvariableop3
/savev2_conv2d_relu_2_kernel_read_readvariableop1
-savev2_conv2d_relu_2_bias_read_readvariableop3
/savev2_conv2d_relu_3_kernel_read_readvariableop1
-savev2_conv2d_relu_3_bias_read_readvariableop3
/savev2_conv2d_relu_4_kernel_read_readvariableop1
-savev2_conv2d_relu_4_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_softmax_kernel_read_readvariableop+
'savev2_softmax_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_adam_batch_norm_gamma_m_read_readvariableop5
1savev2_adam_batch_norm_beta_m_read_readvariableop8
4savev2_adam_conv2d_tanh_kernel_m_read_readvariableop6
2savev2_adam_conv2d_tanh_bias_m_read_readvariableop:
6savev2_adam_conv2d_relu_1_kernel_m_read_readvariableop8
4savev2_adam_conv2d_relu_1_bias_m_read_readvariableop:
6savev2_adam_conv2d_relu_2_kernel_m_read_readvariableop8
4savev2_adam_conv2d_relu_2_bias_m_read_readvariableop:
6savev2_adam_conv2d_relu_3_kernel_m_read_readvariableop8
4savev2_adam_conv2d_relu_3_bias_m_read_readvariableop:
6savev2_adam_conv2d_relu_4_kernel_m_read_readvariableop8
4savev2_adam_conv2d_relu_4_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop4
0savev2_adam_softmax_kernel_m_read_readvariableop2
.savev2_adam_softmax_bias_m_read_readvariableop6
2savev2_adam_batch_norm_gamma_v_read_readvariableop5
1savev2_adam_batch_norm_beta_v_read_readvariableop8
4savev2_adam_conv2d_tanh_kernel_v_read_readvariableop6
2savev2_adam_conv2d_tanh_bias_v_read_readvariableop:
6savev2_adam_conv2d_relu_1_kernel_v_read_readvariableop8
4savev2_adam_conv2d_relu_1_bias_v_read_readvariableop:
6savev2_adam_conv2d_relu_2_kernel_v_read_readvariableop8
4savev2_adam_conv2d_relu_2_bias_v_read_readvariableop:
6savev2_adam_conv2d_relu_3_kernel_v_read_readvariableop8
4savev2_adam_conv2d_relu_3_bias_v_read_readvariableop:
6savev2_adam_conv2d_relu_4_kernel_v_read_readvariableop8
4savev2_adam_conv2d_relu_4_bias_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop4
0savev2_adam_softmax_kernel_v_read_readvariableop2
.savev2_adam_softmax_bias_v_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
:  
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
::*
dtype0*Н
valueГBА:B5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHт
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
::*
dtype0*
value~B|:B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ќ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_batch_norm_gamma_read_readvariableop*savev2_batch_norm_beta_read_readvariableop-savev2_conv2d_tanh_kernel_read_readvariableop+savev2_conv2d_tanh_bias_read_readvariableop/savev2_conv2d_relu_1_kernel_read_readvariableop-savev2_conv2d_relu_1_bias_read_readvariableop/savev2_conv2d_relu_2_kernel_read_readvariableop-savev2_conv2d_relu_2_bias_read_readvariableop/savev2_conv2d_relu_3_kernel_read_readvariableop-savev2_conv2d_relu_3_bias_read_readvariableop/savev2_conv2d_relu_4_kernel_read_readvariableop-savev2_conv2d_relu_4_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_softmax_kernel_read_readvariableop'savev2_softmax_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_adam_batch_norm_gamma_m_read_readvariableop1savev2_adam_batch_norm_beta_m_read_readvariableop4savev2_adam_conv2d_tanh_kernel_m_read_readvariableop2savev2_adam_conv2d_tanh_bias_m_read_readvariableop6savev2_adam_conv2d_relu_1_kernel_m_read_readvariableop4savev2_adam_conv2d_relu_1_bias_m_read_readvariableop6savev2_adam_conv2d_relu_2_kernel_m_read_readvariableop4savev2_adam_conv2d_relu_2_bias_m_read_readvariableop6savev2_adam_conv2d_relu_3_kernel_m_read_readvariableop4savev2_adam_conv2d_relu_3_bias_m_read_readvariableop6savev2_adam_conv2d_relu_4_kernel_m_read_readvariableop4savev2_adam_conv2d_relu_4_bias_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_softmax_kernel_m_read_readvariableop.savev2_adam_softmax_bias_m_read_readvariableop2savev2_adam_batch_norm_gamma_v_read_readvariableop1savev2_adam_batch_norm_beta_v_read_readvariableop4savev2_adam_conv2d_tanh_kernel_v_read_readvariableop2savev2_adam_conv2d_tanh_bias_v_read_readvariableop6savev2_adam_conv2d_relu_1_kernel_v_read_readvariableop4savev2_adam_conv2d_relu_1_bias_v_read_readvariableop6savev2_adam_conv2d_relu_2_kernel_v_read_readvariableop4savev2_adam_conv2d_relu_2_bias_v_read_readvariableop6savev2_adam_conv2d_relu_3_kernel_v_read_readvariableop4savev2_adam_conv2d_relu_3_bias_v_read_readvariableop6savev2_adam_conv2d_relu_4_kernel_v_read_readvariableop4savev2_adam_conv2d_relu_4_bias_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop0savev2_adam_softmax_kernel_v_read_readvariableop.savev2_adam_softmax_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *H
dtypes>
<2:	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0* 
_input_shapes
: :М:М::::::: : :  : :		@:@:@:: : : : : : : : : :М:М::::::: : :  : :		@:@:@::М:М::::::: : :  : :		@:@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:!

_output_shapes	
:М:!

_output_shapes	
:М:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,	(
&
_output_shapes
: : 


_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :%!

_output_shapes
:		@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :!

_output_shapes	
:М:!

_output_shapes	
:М:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::, (
&
_output_shapes
:: !

_output_shapes
::,"(
&
_output_shapes
: : #

_output_shapes
: :,$(
&
_output_shapes
:  : %

_output_shapes
: :%&!

_output_shapes
:		@: '

_output_shapes
:@:$( 

_output_shapes

:@: )

_output_shapes
::!*

_output_shapes	
:М:!+

_output_shapes	
:М:,,(
&
_output_shapes
:: -

_output_shapes
::,.(
&
_output_shapes
:: /

_output_shapes
::,0(
&
_output_shapes
:: 1

_output_shapes
::,2(
&
_output_shapes
: : 3

_output_shapes
: :,4(
&
_output_shapes
:  : 5

_output_shapes
: :%6!

_output_shapes
:		@: 7

_output_shapes
:@:$8 

_output_shapes

:@: 9

_output_shapes
:::

_output_shapes
: 

e
I__inference_max_pool_2d_2_layer_call_and_return_conditional_losses_101055

inputs
identityЁ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ї
Ѓ
.__inference_conv2d_relu_2_layer_call_fn_102143

inputs!
unknown:
	unknown_0:
identityЂStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ
/*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv2d_relu_2_layer_call_and_return_conditional_losses_101176w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ
/`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ
/: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ
/
 
_user_specified_nameinputs


I__inference_conv2d_relu_1_layer_call_and_return_conditional_losses_102124

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ^*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ^X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ^i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ^w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ^: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ^
 
_user_specified_nameinputs
ц
У
/__inference_2d_convolution_layer_call_fn_101607
input_1
unknown:	М
	unknown_0:	М#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7: 
	unknown_8: #
	unknown_9:  

unknown_10: 

unknown_11:		@

unknown_12:@

unknown_13:@

unknown_14:
identityЂStatefulPartitionedCallЁ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:џџџџџџџџџ: *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_2d_convolution_layer_call_and_return_conditional_losses_101533o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:џџџџџџџџџ(М: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:џџџџџџџџџ(М
!
_user_specified_name	input_1
З
J
.__inference_max_pool_2d_2_layer_call_fn_102129

inputs
identityк
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_max_pool_2d_2_layer_call_and_return_conditional_losses_101055
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
А
ћ
F__inference_batch_norm_layer_call_and_return_conditional_losses_101123

inputs.
reshape_readvariableop_resource:	М0
!reshape_1_readvariableop_resource:	М
identityЂReshape/ReadVariableOpЂReshape_1/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*/
_output_shapes
:џџџџџџџџџ(*
	keep_dims(u
moments/StopGradientStopGradientmoments/mean:output:0*
T0*/
_output_shapes
:џџџџџџџџџ(
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*0
_output_shapes
:џџџџџџџџџ(Мl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:Џ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*/
_output_shapes
:џџџџџџџџџ(*
	keep_dims(s
Reshape/ReadVariableOpReadVariableOpreshape_readvariableop_resource*
_output_shapes	
:М*
dtype0f
Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      М      |
ReshapeReshapeReshape/ReadVariableOp:value:0Reshape/shape:output:0*
T0*'
_output_shapes
:Мw
Reshape_1/ReadVariableOpReadVariableOp!reshape_1_readvariableop_resource*
_output_shapes	
:М*
dtype0h
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      М      
	Reshape_1Reshape Reshape_1/ReadVariableOp:value:0Reshape_1/shape:output:0*
T0*'
_output_shapes
:МT
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:
batchnorm/addAddV2moments/variance:output:0batchnorm/add/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ(e
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ(v
batchnorm/mulMulbatchnorm/Rsqrt:y:0Reshape:output:0*
T0*0
_output_shapes
:џџџџџџџџџ(Мl
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*0
_output_shapes
:џџџџџџџџџ(М{
batchnorm/mul_2Mulmoments/mean:output:0batchnorm/mul:z:0*
T0*0
_output_shapes
:џџџџџџџџџ(Мx
batchnorm/subSubReshape_1:output:0batchnorm/mul_2:z:0*
T0*0
_output_shapes
:џџџџџџџџџ(М{
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*0
_output_shapes
:џџџџџџџџџ(Мk
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ(Мz
NoOpNoOp^Reshape/ReadVariableOp^Reshape_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ(М: : 20
Reshape/ReadVariableOpReshape/ReadVariableOp24
Reshape_1/ReadVariableOpReshape_1/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ(М
 
_user_specified_nameinputs
д
Т
!__inference__wrapped_model_101034
input_1G
8d_convolution_batch_norm_reshape_readvariableop_resource:	МI
:d_convolution_batch_norm_reshape_1_readvariableop_resource:	МR
8d_convolution_conv2d_tanh_conv2d_readvariableop_resource:G
9d_convolution_conv2d_tanh_biasadd_readvariableop_resource:T
:d_convolution_conv2d_relu_1_conv2d_readvariableop_resource:I
;d_convolution_conv2d_relu_1_biasadd_readvariableop_resource:T
:d_convolution_conv2d_relu_2_conv2d_readvariableop_resource:I
;d_convolution_conv2d_relu_2_biasadd_readvariableop_resource:T
:d_convolution_conv2d_relu_3_conv2d_readvariableop_resource: I
;d_convolution_conv2d_relu_3_biasadd_readvariableop_resource: T
:d_convolution_conv2d_relu_4_conv2d_readvariableop_resource:  I
;d_convolution_conv2d_relu_4_biasadd_readvariableop_resource: E
2d_convolution_dense_matmul_readvariableop_resource:		@A
3d_convolution_dense_biasadd_readvariableop_resource:@F
4d_convolution_softmax_matmul_readvariableop_resource:@C
5d_convolution_softmax_biasadd_readvariableop_resource:
identityЂ02d_convolution/batch_norm/Reshape/ReadVariableOpЂ22d_convolution/batch_norm/Reshape_1/ReadVariableOpЂ32d_convolution/conv2d_relu_1/BiasAdd/ReadVariableOpЂ22d_convolution/conv2d_relu_1/Conv2D/ReadVariableOpЂ32d_convolution/conv2d_relu_2/BiasAdd/ReadVariableOpЂ22d_convolution/conv2d_relu_2/Conv2D/ReadVariableOpЂ32d_convolution/conv2d_relu_3/BiasAdd/ReadVariableOpЂ22d_convolution/conv2d_relu_3/Conv2D/ReadVariableOpЂ32d_convolution/conv2d_relu_4/BiasAdd/ReadVariableOpЂ22d_convolution/conv2d_relu_4/Conv2D/ReadVariableOpЂ12d_convolution/conv2d_tanh/BiasAdd/ReadVariableOpЂ02d_convolution/conv2d_tanh/Conv2D/ReadVariableOpЂ+2d_convolution/dense/BiasAdd/ReadVariableOpЂ*2d_convolution/dense/MatMul/ReadVariableOpЂ-2d_convolution/softmax/BiasAdd/ReadVariableOpЂ,2d_convolution/softmax/MatMul/ReadVariableOp
82d_convolution/batch_norm/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:Х
&2d_convolution/batch_norm/moments/meanMeaninput_1A2d_convolution/batch_norm/moments/mean/reduction_indices:output:0*
T0*/
_output_shapes
:џџџџџџџџџ(*
	keep_dims(Љ
.2d_convolution/batch_norm/moments/StopGradientStopGradient/2d_convolution/batch_norm/moments/mean:output:0*
T0*/
_output_shapes
:џџџџџџџџџ(Х
32d_convolution/batch_norm/moments/SquaredDifferenceSquaredDifferenceinput_172d_convolution/batch_norm/moments/StopGradient:output:0*
T0*0
_output_shapes
:џџџџџџџџџ(М
<2d_convolution/batch_norm/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:§
*2d_convolution/batch_norm/moments/varianceMean72d_convolution/batch_norm/moments/SquaredDifference:z:0E2d_convolution/batch_norm/moments/variance/reduction_indices:output:0*
T0*/
_output_shapes
:џџџџџџџџџ(*
	keep_dims(І
02d_convolution/batch_norm/Reshape/ReadVariableOpReadVariableOp8d_convolution_batch_norm_reshape_readvariableop_resource*
_output_shapes	
:М*
dtype0
'2d_convolution/batch_norm/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      М      Ъ
!2d_convolution/batch_norm/ReshapeReshape82d_convolution/batch_norm/Reshape/ReadVariableOp:value:002d_convolution/batch_norm/Reshape/shape:output:0*
T0*'
_output_shapes
:МЊ
22d_convolution/batch_norm/Reshape_1/ReadVariableOpReadVariableOp:d_convolution_batch_norm_reshape_1_readvariableop_resource*
_output_shapes	
:М*
dtype0
)2d_convolution/batch_norm/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      М      а
#2d_convolution/batch_norm/Reshape_1Reshape:2d_convolution/batch_norm/Reshape_1/ReadVariableOp:value:022d_convolution/batch_norm/Reshape_1/shape:output:0*
T0*'
_output_shapes
:Мn
)2d_convolution/batch_norm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:г
'2d_convolution/batch_norm/batchnorm/addAddV232d_convolution/batch_norm/moments/variance:output:022d_convolution/batch_norm/batchnorm/add/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ(
)2d_convolution/batch_norm/batchnorm/RsqrtRsqrt+2d_convolution/batch_norm/batchnorm/add:z:0*
T0*/
_output_shapes
:џџџџџџџџџ(Ф
'2d_convolution/batch_norm/batchnorm/mulMul-2d_convolution/batch_norm/batchnorm/Rsqrt:y:0*2d_convolution/batch_norm/Reshape:output:0*
T0*0
_output_shapes
:џџџџџџџџџ(МЁ
)2d_convolution/batch_norm/batchnorm/mul_1Mulinput_1+2d_convolution/batch_norm/batchnorm/mul:z:0*
T0*0
_output_shapes
:џџџџџџџџџ(МЩ
)2d_convolution/batch_norm/batchnorm/mul_2Mul/2d_convolution/batch_norm/moments/mean:output:0+2d_convolution/batch_norm/batchnorm/mul:z:0*
T0*0
_output_shapes
:џџџџџџџџџ(МЦ
'2d_convolution/batch_norm/batchnorm/subSub,2d_convolution/batch_norm/Reshape_1:output:0-2d_convolution/batch_norm/batchnorm/mul_2:z:0*
T0*0
_output_shapes
:џџџџџџџџџ(МЩ
)2d_convolution/batch_norm/batchnorm/add_1AddV2-2d_convolution/batch_norm/batchnorm/mul_1:z:0+2d_convolution/batch_norm/batchnorm/sub:z:0*
T0*0
_output_shapes
:џџџџџџџџџ(МБ
02d_convolution/conv2d_tanh/Conv2D/ReadVariableOpReadVariableOp8d_convolution_conv2d_tanh_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0ї
!2d_convolution/conv2d_tanh/Conv2DConv2D-2d_convolution/batch_norm/batchnorm/add_1:z:082d_convolution/conv2d_tanh/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ(М*
paddingSAME*
strides
Ї
12d_convolution/conv2d_tanh/BiasAdd/ReadVariableOpReadVariableOp9d_convolution_conv2d_tanh_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Я
"2d_convolution/conv2d_tanh/BiasAddBiasAdd*2d_convolution/conv2d_tanh/Conv2D:output:092d_convolution/conv2d_tanh/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ(М
2d_convolution/conv2d_tanh/TanhTanh+2d_convolution/conv2d_tanh/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ(МР
$2d_convolution/max_pool_2d_1/MaxPoolMaxPool#2d_convolution/conv2d_tanh/Tanh:y:0*/
_output_shapes
:џџџџџџџџџ^*
ksize
*
paddingSAME*
strides
Е
22d_convolution/conv2d_relu_1/Conv2D/ReadVariableOpReadVariableOp:d_convolution_conv2d_relu_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0њ
#2d_convolution/conv2d_relu_1/Conv2DConv2D-2d_convolution/max_pool_2d_1/MaxPool:output:0:2d_convolution/conv2d_relu_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ^*
paddingSAME*
strides
Ћ
32d_convolution/conv2d_relu_1/BiasAdd/ReadVariableOpReadVariableOp;d_convolution_conv2d_relu_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0д
$2d_convolution/conv2d_relu_1/BiasAddBiasAdd,2d_convolution/conv2d_relu_1/Conv2D:output:0;2d_convolution/conv2d_relu_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ^
!2d_convolution/conv2d_relu_1/ReluRelu-2d_convolution/conv2d_relu_1/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ^Ь
$2d_convolution/max_pool_2d_2/MaxPoolMaxPool/2d_convolution/conv2d_relu_1/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ
/*
ksize
*
paddingSAME*
strides
Е
22d_convolution/conv2d_relu_2/Conv2D/ReadVariableOpReadVariableOp:d_convolution_conv2d_relu_2_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0њ
#2d_convolution/conv2d_relu_2/Conv2DConv2D-2d_convolution/max_pool_2d_2/MaxPool:output:0:2d_convolution/conv2d_relu_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ
/*
paddingSAME*
strides
Ћ
32d_convolution/conv2d_relu_2/BiasAdd/ReadVariableOpReadVariableOp;d_convolution_conv2d_relu_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0д
$2d_convolution/conv2d_relu_2/BiasAddBiasAdd,2d_convolution/conv2d_relu_2/Conv2D:output:0;2d_convolution/conv2d_relu_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ
/
!2d_convolution/conv2d_relu_2/ReluRelu-2d_convolution/conv2d_relu_2/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ
/Ь
$2d_convolution/max_pool_2d_3/MaxPoolMaxPool/2d_convolution/conv2d_relu_2/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingSAME*
strides
Е
22d_convolution/conv2d_relu_3/Conv2D/ReadVariableOpReadVariableOp:d_convolution_conv2d_relu_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0њ
#2d_convolution/conv2d_relu_3/Conv2DConv2D-2d_convolution/max_pool_2d_3/MaxPool:output:0:2d_convolution/conv2d_relu_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingSAME*
strides
Ћ
32d_convolution/conv2d_relu_3/BiasAdd/ReadVariableOpReadVariableOp;d_convolution_conv2d_relu_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0д
$2d_convolution/conv2d_relu_3/BiasAddBiasAdd,2d_convolution/conv2d_relu_3/Conv2D:output:0;2d_convolution/conv2d_relu_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ 
!2d_convolution/conv2d_relu_3/ReluRelu-2d_convolution/conv2d_relu_3/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ Ь
$2d_convolution/max_pool_2d_4/MaxPoolMaxPool/2d_convolution/conv2d_relu_3/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ *
ksize
*
paddingSAME*
strides
Е
22d_convolution/conv2d_relu_4/Conv2D/ReadVariableOpReadVariableOp:d_convolution_conv2d_relu_4_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0њ
#2d_convolution/conv2d_relu_4/Conv2DConv2D-2d_convolution/max_pool_2d_4/MaxPool:output:0:2d_convolution/conv2d_relu_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ *
paddingSAME*
strides
Ћ
32d_convolution/conv2d_relu_4/BiasAdd/ReadVariableOpReadVariableOp;d_convolution_conv2d_relu_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0д
$2d_convolution/conv2d_relu_4/BiasAddBiasAdd,2d_convolution/conv2d_relu_4/Conv2D:output:0;2d_convolution/conv2d_relu_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ 
!2d_convolution/conv2d_relu_4/ReluRelu-2d_convolution/conv2d_relu_4/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ m
2d_convolution/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ  Д
2d_convolution/flatten/ReshapeReshape/2d_convolution/conv2d_relu_4/Relu:activations:0%2d_convolution/flatten/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ	
2d_convolution/dropout/IdentityIdentity'2d_convolution/flatten/Reshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ	
*2d_convolution/dense/MatMul/ReadVariableOpReadVariableOp2d_convolution_dense_matmul_readvariableop_resource*
_output_shapes
:		@*
dtype0Е
2d_convolution/dense/MatMulMatMul(2d_convolution/dropout/Identity:output:022d_convolution/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@
+2d_convolution/dense/BiasAdd/ReadVariableOpReadVariableOp3d_convolution_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Е
2d_convolution/dense/BiasAddBiasAdd%2d_convolution/dense/MatMul:product:032d_convolution/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@z
2d_convolution/dense/ReluRelu%2d_convolution/dense/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@
/2d_convolution/dense/ActivityRegularizer/L2LossL2Loss'2d_convolution/dense/Relu:activations:0*
T0*
_output_shapes
: s
.2d_convolution/dense/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o;Ч
,2d_convolution/dense/ActivityRegularizer/mulMul72d_convolution/dense/ActivityRegularizer/mul/x:output:082d_convolution/dense/ActivityRegularizer/L2Loss:output:0*
T0*
_output_shapes
: 
.2d_convolution/dense/ActivityRegularizer/ShapeShape'2d_convolution/dense/Relu:activations:0*
T0*
_output_shapes
:
<2d_convolution/dense/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
>2d_convolution/dense/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
>2d_convolution/dense/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
62d_convolution/dense/ActivityRegularizer/strided_sliceStridedSlice72d_convolution/dense/ActivityRegularizer/Shape:output:0E2d_convolution/dense/ActivityRegularizer/strided_slice/stack:output:0G2d_convolution/dense/ActivityRegularizer/strided_slice/stack_1:output:0G2d_convolution/dense/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskІ
-2d_convolution/dense/ActivityRegularizer/CastCast?2d_convolution/dense/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: С
02d_convolution/dense/ActivityRegularizer/truedivRealDiv02d_convolution/dense/ActivityRegularizer/mul:z:012d_convolution/dense/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: Ё
,2d_convolution/softmax/MatMul/ReadVariableOpReadVariableOp4d_convolution_softmax_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0И
2d_convolution/softmax/MatMulMatMul'2d_convolution/dense/Relu:activations:042d_convolution/softmax/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
-2d_convolution/softmax/BiasAdd/ReadVariableOpReadVariableOp5d_convolution_softmax_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Л
2d_convolution/softmax/BiasAddBiasAdd'2d_convolution/softmax/MatMul:product:052d_convolution/softmax/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
2d_convolution/softmax/SoftmaxSoftmax'2d_convolution/softmax/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџw
IdentityIdentity(2d_convolution/softmax/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџћ
NoOpNoOp1^2d_convolution/batch_norm/Reshape/ReadVariableOp3^2d_convolution/batch_norm/Reshape_1/ReadVariableOp4^2d_convolution/conv2d_relu_1/BiasAdd/ReadVariableOp3^2d_convolution/conv2d_relu_1/Conv2D/ReadVariableOp4^2d_convolution/conv2d_relu_2/BiasAdd/ReadVariableOp3^2d_convolution/conv2d_relu_2/Conv2D/ReadVariableOp4^2d_convolution/conv2d_relu_3/BiasAdd/ReadVariableOp3^2d_convolution/conv2d_relu_3/Conv2D/ReadVariableOp4^2d_convolution/conv2d_relu_4/BiasAdd/ReadVariableOp3^2d_convolution/conv2d_relu_4/Conv2D/ReadVariableOp2^2d_convolution/conv2d_tanh/BiasAdd/ReadVariableOp1^2d_convolution/conv2d_tanh/Conv2D/ReadVariableOp,^2d_convolution/dense/BiasAdd/ReadVariableOp+^2d_convolution/dense/MatMul/ReadVariableOp.^2d_convolution/softmax/BiasAdd/ReadVariableOp-^2d_convolution/softmax/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:џџџџџџџџџ(М: : : : : : : : : : : : : : : : 2d
02d_convolution/batch_norm/Reshape/ReadVariableOp02d_convolution/batch_norm/Reshape/ReadVariableOp2h
22d_convolution/batch_norm/Reshape_1/ReadVariableOp22d_convolution/batch_norm/Reshape_1/ReadVariableOp2j
32d_convolution/conv2d_relu_1/BiasAdd/ReadVariableOp32d_convolution/conv2d_relu_1/BiasAdd/ReadVariableOp2h
22d_convolution/conv2d_relu_1/Conv2D/ReadVariableOp22d_convolution/conv2d_relu_1/Conv2D/ReadVariableOp2j
32d_convolution/conv2d_relu_2/BiasAdd/ReadVariableOp32d_convolution/conv2d_relu_2/BiasAdd/ReadVariableOp2h
22d_convolution/conv2d_relu_2/Conv2D/ReadVariableOp22d_convolution/conv2d_relu_2/Conv2D/ReadVariableOp2j
32d_convolution/conv2d_relu_3/BiasAdd/ReadVariableOp32d_convolution/conv2d_relu_3/BiasAdd/ReadVariableOp2h
22d_convolution/conv2d_relu_3/Conv2D/ReadVariableOp22d_convolution/conv2d_relu_3/Conv2D/ReadVariableOp2j
32d_convolution/conv2d_relu_4/BiasAdd/ReadVariableOp32d_convolution/conv2d_relu_4/BiasAdd/ReadVariableOp2h
22d_convolution/conv2d_relu_4/Conv2D/ReadVariableOp22d_convolution/conv2d_relu_4/Conv2D/ReadVariableOp2f
12d_convolution/conv2d_tanh/BiasAdd/ReadVariableOp12d_convolution/conv2d_tanh/BiasAdd/ReadVariableOp2d
02d_convolution/conv2d_tanh/Conv2D/ReadVariableOp02d_convolution/conv2d_tanh/Conv2D/ReadVariableOp2Z
+2d_convolution/dense/BiasAdd/ReadVariableOp+2d_convolution/dense/BiasAdd/ReadVariableOp2X
*2d_convolution/dense/MatMul/ReadVariableOp*2d_convolution/dense/MatMul/ReadVariableOp2^
-2d_convolution/softmax/BiasAdd/ReadVariableOp-2d_convolution/softmax/BiasAdd/ReadVariableOp2\
,2d_convolution/softmax/MatMul/ReadVariableOp,2d_convolution/softmax/MatMul/ReadVariableOp:Y U
0
_output_shapes
:џџџџџџџџџ(М
!
_user_specified_name	input_1
є
a
(__inference_dropout_layer_call_fn_102235

inputs
identityЂStatefulPartitionedCallТ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_101365p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ	22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ	
 
_user_specified_nameinputs

e
I__inference_max_pool_2d_4_layer_call_and_return_conditional_losses_102194

inputs
identityЁ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ї
Ё
,__inference_conv2d_tanh_layer_call_fn_102083

inputs!
unknown:
	unknown_0:
identityЂStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ(М*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv2d_tanh_layer_call_and_return_conditional_losses_101140x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ(М`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ(М: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџ(М
 
_user_specified_nameinputs
ЙJ

J__inference_2d_convolution_layer_call_and_return_conditional_losses_101666
input_1 
batch_norm_101610:	М 
batch_norm_101612:	М,
conv2d_tanh_101615: 
conv2d_tanh_101617:.
conv2d_relu_1_101621:"
conv2d_relu_1_101623:.
conv2d_relu_2_101627:"
conv2d_relu_2_101629:.
conv2d_relu_3_101633: "
conv2d_relu_3_101635: .
conv2d_relu_4_101639:  "
conv2d_relu_4_101641: 
dense_101646:		@
dense_101648:@ 
softmax_101659:@
softmax_101661:
identity

identity_1Ђ"batch_norm/StatefulPartitionedCallЂ%conv2d_relu_1/StatefulPartitionedCallЂ%conv2d_relu_2/StatefulPartitionedCallЂ%conv2d_relu_3/StatefulPartitionedCallЂ%conv2d_relu_4/StatefulPartitionedCallЂ#conv2d_tanh/StatefulPartitionedCallЂdense/StatefulPartitionedCallЂsoftmax/StatefulPartitionedCall
"batch_norm/StatefulPartitionedCallStatefulPartitionedCallinput_1batch_norm_101610batch_norm_101612*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ(М*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_batch_norm_layer_call_and_return_conditional_losses_101123­
#conv2d_tanh/StatefulPartitionedCallStatefulPartitionedCall+batch_norm/StatefulPartitionedCall:output:0conv2d_tanh_101615conv2d_tanh_101617*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ(М*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv2d_tanh_layer_call_and_return_conditional_losses_101140ѓ
max_pool_2d_1/PartitionedCallPartitionedCall,conv2d_tanh/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ^* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_max_pool_2d_1_layer_call_and_return_conditional_losses_101043Џ
%conv2d_relu_1/StatefulPartitionedCallStatefulPartitionedCall&max_pool_2d_1/PartitionedCall:output:0conv2d_relu_1_101621conv2d_relu_1_101623*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ^*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv2d_relu_1_layer_call_and_return_conditional_losses_101158ѕ
max_pool_2d_2/PartitionedCallPartitionedCall.conv2d_relu_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ
/* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_max_pool_2d_2_layer_call_and_return_conditional_losses_101055Џ
%conv2d_relu_2/StatefulPartitionedCallStatefulPartitionedCall&max_pool_2d_2/PartitionedCall:output:0conv2d_relu_2_101627conv2d_relu_2_101629*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ
/*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv2d_relu_2_layer_call_and_return_conditional_losses_101176ѕ
max_pool_2d_3/PartitionedCallPartitionedCall.conv2d_relu_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_max_pool_2d_3_layer_call_and_return_conditional_losses_101067Џ
%conv2d_relu_3/StatefulPartitionedCallStatefulPartitionedCall&max_pool_2d_3/PartitionedCall:output:0conv2d_relu_3_101633conv2d_relu_3_101635*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv2d_relu_3_layer_call_and_return_conditional_losses_101194ѕ
max_pool_2d_4/PartitionedCallPartitionedCall.conv2d_relu_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_max_pool_2d_4_layer_call_and_return_conditional_losses_101079Џ
%conv2d_relu_4/StatefulPartitionedCallStatefulPartitionedCall&max_pool_2d_4/PartitionedCall:output:0conv2d_relu_4_101639conv2d_relu_4_101641*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv2d_relu_4_layer_call_and_return_conditional_losses_101212т
flatten/PartitionedCallPartitionedCall.conv2d_relu_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_101224д
dropout/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_101231
dense/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_101646dense_101648*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_101244Ф
)dense/ActivityRegularizer/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *6
f1R/
-__inference_dense_activity_regularizer_101090u
dense/ActivityRegularizer/ShapeShape&dense/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:w
-dense/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/dense/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/dense/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:г
'dense/ActivityRegularizer/strided_sliceStridedSlice(dense/ActivityRegularizer/Shape:output:06dense/ActivityRegularizer/strided_slice/stack:output:08dense/ActivityRegularizer/strided_slice/stack_1:output:08dense/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
dense/ActivityRegularizer/CastCast0dense/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: Ѕ
!dense/ActivityRegularizer/truedivRealDiv2dense/ActivityRegularizer/PartitionedCall:output:0"dense/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 
softmax/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0softmax_101659softmax_101661*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_softmax_layer_call_and_return_conditional_losses_101269w
IdentityIdentity(softmax/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџe

Identity_1Identity%dense/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: ѓ
NoOpNoOp#^batch_norm/StatefulPartitionedCall&^conv2d_relu_1/StatefulPartitionedCall&^conv2d_relu_2/StatefulPartitionedCall&^conv2d_relu_3/StatefulPartitionedCall&^conv2d_relu_4/StatefulPartitionedCall$^conv2d_tanh/StatefulPartitionedCall^dense/StatefulPartitionedCall ^softmax/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:џџџџџџџџџ(М: : : : : : : : : : : : : : : : 2H
"batch_norm/StatefulPartitionedCall"batch_norm/StatefulPartitionedCall2N
%conv2d_relu_1/StatefulPartitionedCall%conv2d_relu_1/StatefulPartitionedCall2N
%conv2d_relu_2/StatefulPartitionedCall%conv2d_relu_2/StatefulPartitionedCall2N
%conv2d_relu_3/StatefulPartitionedCall%conv2d_relu_3/StatefulPartitionedCall2N
%conv2d_relu_4/StatefulPartitionedCall%conv2d_relu_4/StatefulPartitionedCall2J
#conv2d_tanh/StatefulPartitionedCall#conv2d_tanh/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
softmax/StatefulPartitionedCallsoftmax/StatefulPartitionedCall:Y U
0
_output_shapes
:џџџџџџџџџ(М
!
_user_specified_name	input_1
жK
Љ
J__inference_2d_convolution_layer_call_and_return_conditional_losses_101533

inputs 
batch_norm_101477:	М 
batch_norm_101479:	М,
conv2d_tanh_101482: 
conv2d_tanh_101484:.
conv2d_relu_1_101488:"
conv2d_relu_1_101490:.
conv2d_relu_2_101494:"
conv2d_relu_2_101496:.
conv2d_relu_3_101500: "
conv2d_relu_3_101502: .
conv2d_relu_4_101506:  "
conv2d_relu_4_101508: 
dense_101513:		@
dense_101515:@ 
softmax_101526:@
softmax_101528:
identity

identity_1Ђ"batch_norm/StatefulPartitionedCallЂ%conv2d_relu_1/StatefulPartitionedCallЂ%conv2d_relu_2/StatefulPartitionedCallЂ%conv2d_relu_3/StatefulPartitionedCallЂ%conv2d_relu_4/StatefulPartitionedCallЂ#conv2d_tanh/StatefulPartitionedCallЂdense/StatefulPartitionedCallЂdropout/StatefulPartitionedCallЂsoftmax/StatefulPartitionedCall
"batch_norm/StatefulPartitionedCallStatefulPartitionedCallinputsbatch_norm_101477batch_norm_101479*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ(М*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_batch_norm_layer_call_and_return_conditional_losses_101123­
#conv2d_tanh/StatefulPartitionedCallStatefulPartitionedCall+batch_norm/StatefulPartitionedCall:output:0conv2d_tanh_101482conv2d_tanh_101484*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ(М*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv2d_tanh_layer_call_and_return_conditional_losses_101140ѓ
max_pool_2d_1/PartitionedCallPartitionedCall,conv2d_tanh/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ^* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_max_pool_2d_1_layer_call_and_return_conditional_losses_101043Џ
%conv2d_relu_1/StatefulPartitionedCallStatefulPartitionedCall&max_pool_2d_1/PartitionedCall:output:0conv2d_relu_1_101488conv2d_relu_1_101490*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ^*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv2d_relu_1_layer_call_and_return_conditional_losses_101158ѕ
max_pool_2d_2/PartitionedCallPartitionedCall.conv2d_relu_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ
/* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_max_pool_2d_2_layer_call_and_return_conditional_losses_101055Џ
%conv2d_relu_2/StatefulPartitionedCallStatefulPartitionedCall&max_pool_2d_2/PartitionedCall:output:0conv2d_relu_2_101494conv2d_relu_2_101496*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ
/*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv2d_relu_2_layer_call_and_return_conditional_losses_101176ѕ
max_pool_2d_3/PartitionedCallPartitionedCall.conv2d_relu_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_max_pool_2d_3_layer_call_and_return_conditional_losses_101067Џ
%conv2d_relu_3/StatefulPartitionedCallStatefulPartitionedCall&max_pool_2d_3/PartitionedCall:output:0conv2d_relu_3_101500conv2d_relu_3_101502*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv2d_relu_3_layer_call_and_return_conditional_losses_101194ѕ
max_pool_2d_4/PartitionedCallPartitionedCall.conv2d_relu_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_max_pool_2d_4_layer_call_and_return_conditional_losses_101079Џ
%conv2d_relu_4/StatefulPartitionedCallStatefulPartitionedCall&max_pool_2d_4/PartitionedCall:output:0conv2d_relu_4_101506conv2d_relu_4_101508*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_conv2d_relu_4_layer_call_and_return_conditional_losses_101212т
flatten/PartitionedCallPartitionedCall.conv2d_relu_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_101224ф
dropout/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_101365
dense/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_101513dense_101515*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_101244Ф
)dense/ActivityRegularizer/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *6
f1R/
-__inference_dense_activity_regularizer_101090u
dense/ActivityRegularizer/ShapeShape&dense/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:w
-dense/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/dense/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/dense/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:г
'dense/ActivityRegularizer/strided_sliceStridedSlice(dense/ActivityRegularizer/Shape:output:06dense/ActivityRegularizer/strided_slice/stack:output:08dense/ActivityRegularizer/strided_slice/stack_1:output:08dense/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
dense/ActivityRegularizer/CastCast0dense/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: Ѕ
!dense/ActivityRegularizer/truedivRealDiv2dense/ActivityRegularizer/PartitionedCall:output:0"dense/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 
softmax/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0softmax_101526softmax_101528*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_softmax_layer_call_and_return_conditional_losses_101269w
IdentityIdentity(softmax/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџe

Identity_1Identity%dense/ActivityRegularizer/truediv:z:0^NoOp*
T0*
_output_shapes
: 
NoOpNoOp#^batch_norm/StatefulPartitionedCall&^conv2d_relu_1/StatefulPartitionedCall&^conv2d_relu_2/StatefulPartitionedCall&^conv2d_relu_3/StatefulPartitionedCall&^conv2d_relu_4/StatefulPartitionedCall$^conv2d_tanh/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dropout/StatefulPartitionedCall ^softmax/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:џџџџџџџџџ(М: : : : : : : : : : : : : : : : 2H
"batch_norm/StatefulPartitionedCall"batch_norm/StatefulPartitionedCall2N
%conv2d_relu_1/StatefulPartitionedCall%conv2d_relu_1/StatefulPartitionedCall2N
%conv2d_relu_2/StatefulPartitionedCall%conv2d_relu_2/StatefulPartitionedCall2N
%conv2d_relu_3/StatefulPartitionedCall%conv2d_relu_3/StatefulPartitionedCall2N
%conv2d_relu_4/StatefulPartitionedCall%conv2d_relu_4/StatefulPartitionedCall2J
#conv2d_tanh/StatefulPartitionedCall#conv2d_tanh/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2B
softmax/StatefulPartitionedCallsoftmax/StatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџ(М
 
_user_specified_nameinputs

e
I__inference_max_pool_2d_3_layer_call_and_return_conditional_losses_102164

inputs
identityЁ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Х
_
C__inference_flatten_layer_call_and_return_conditional_losses_101224

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџ	Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ :W S
/
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs"Е	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Г
serving_default
D
input_19
serving_default_input_1:0џџџџџџџџџ(М;
softmax0
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:уд
ћ
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer-9
layer_with_weights-5
layer-10
layer-11
layer-12
layer_with_weights-6
layer-13
layer_with_weights-7
layer-14
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
Ф
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
axis
	 gamma
!beta"
_tf_keras_layer
н
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses

(kernel
)bias
 *_jit_compiled_convolution_op"
_tf_keras_layer
Ѕ
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses"
_tf_keras_layer
н
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses

7kernel
8bias
 9_jit_compiled_convolution_op"
_tf_keras_layer
Ѕ
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses"
_tf_keras_layer
н
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses

Fkernel
Gbias
 H_jit_compiled_convolution_op"
_tf_keras_layer
Ѕ
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses"
_tf_keras_layer
н
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses

Ukernel
Vbias
 W_jit_compiled_convolution_op"
_tf_keras_layer
Ѕ
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses"
_tf_keras_layer
н
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses

dkernel
ebias
 f_jit_compiled_convolution_op"
_tf_keras_layer
Ѕ
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses"
_tf_keras_layer
М
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses
s_random_generator"
_tf_keras_layer
Л
t	variables
utrainable_variables
vregularization_losses
w	keras_api
x__call__
*y&call_and_return_all_conditional_losses

zkernel
{bias"
_tf_keras_layer
П
|	variables
}trainable_variables
~regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias"
_tf_keras_layer

 0
!1
(2
)3
74
85
F6
G7
U8
V9
d10
e11
z12
{13
14
15"
trackable_list_wrapper

 0
!1
(2
)3
74
85
F6
G7
U8
V9
d10
e11
z12
{13
14
15"
trackable_list_wrapper
 "
trackable_list_wrapper
Я
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
љ
trace_0
trace_1
trace_2
trace_32
/__inference_2d_convolution_layer_call_fn_101313
/__inference_2d_convolution_layer_call_fn_101808
/__inference_2d_convolution_layer_call_fn_101846
/__inference_2d_convolution_layer_call_fn_101607П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1ztrace_2ztrace_3
х
trace_0
trace_1
trace_2
trace_32ђ
J__inference_2d_convolution_layer_call_and_return_conditional_losses_101939
J__inference_2d_convolution_layer_call_and_return_conditional_losses_102039
J__inference_2d_convolution_layer_call_and_return_conditional_losses_101666
J__inference_2d_convolution_layer_call_and_return_conditional_losses_101725П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1ztrace_2ztrace_3
ЬBЩ
!__inference__wrapped_model_101034input_1"
В
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 

	iter
beta_1
beta_2

decay
learning_rate m!m(m)m7m8mFmGmUmVmdmemzm{m	m	m v!v(v)v7v8vFv GvЁUvЂVvЃdvЄevЅzvІ{vЇ	vЈ	vЉ"
	optimizer
-
serving_default"
signature_map
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ё
trace_02в
+__inference_batch_norm_layer_call_fn_102048Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02э
F__inference_batch_norm_layer_call_and_return_conditional_losses_102074Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
 "
trackable_list_wrapper
:М2batch_norm/gamma
:М2batch_norm/beta
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
 metrics
 Ёlayer_regularization_losses
Ђlayer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses"
_generic_user_object
ђ
Ѓtrace_02г
,__inference_conv2d_tanh_layer_call_fn_102083Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЃtrace_0

Єtrace_02ю
G__inference_conv2d_tanh_layer_call_and_return_conditional_losses_102094Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЄtrace_0
,:*2conv2d_tanh/kernel
:2conv2d_tanh/bias
Д2БЎ
ЃВ
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
Ѕnon_trainable_variables
Іlayers
Їmetrics
 Јlayer_regularization_losses
Љlayer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses"
_generic_user_object
є
Њtrace_02е
.__inference_max_pool_2d_1_layer_call_fn_102099Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЊtrace_0

Ћtrace_02№
I__inference_max_pool_2d_1_layer_call_and_return_conditional_losses_102104Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЋtrace_0
.
70
81"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
В
Ќnon_trainable_variables
­layers
Ўmetrics
 Џlayer_regularization_losses
Аlayer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
є
Бtrace_02е
.__inference_conv2d_relu_1_layer_call_fn_102113Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zБtrace_0

Вtrace_02№
I__inference_conv2d_relu_1_layer_call_and_return_conditional_losses_102124Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zВtrace_0
.:,2conv2d_relu_1/kernel
 :2conv2d_relu_1/bias
Д2БЎ
ЃВ
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
Гnon_trainable_variables
Дlayers
Еmetrics
 Жlayer_regularization_losses
Зlayer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
є
Иtrace_02е
.__inference_max_pool_2d_2_layer_call_fn_102129Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zИtrace_0

Йtrace_02№
I__inference_max_pool_2d_2_layer_call_and_return_conditional_losses_102134Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЙtrace_0
.
F0
G1"
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
Кnon_trainable_variables
Лlayers
Мmetrics
 Нlayer_regularization_losses
Оlayer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
є
Пtrace_02е
.__inference_conv2d_relu_2_layer_call_fn_102143Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zПtrace_0

Рtrace_02№
I__inference_conv2d_relu_2_layer_call_and_return_conditional_losses_102154Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zРtrace_0
.:,2conv2d_relu_2/kernel
 :2conv2d_relu_2/bias
Д2БЎ
ЃВ
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
Сnon_trainable_variables
Тlayers
Уmetrics
 Фlayer_regularization_losses
Хlayer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses"
_generic_user_object
є
Цtrace_02е
.__inference_max_pool_2d_3_layer_call_fn_102159Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЦtrace_0

Чtrace_02№
I__inference_max_pool_2d_3_layer_call_and_return_conditional_losses_102164Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЧtrace_0
.
U0
V1"
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
Шnon_trainable_variables
Щlayers
Ъmetrics
 Ыlayer_regularization_losses
Ьlayer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
є
Эtrace_02е
.__inference_conv2d_relu_3_layer_call_fn_102173Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЭtrace_0

Юtrace_02№
I__inference_conv2d_relu_3_layer_call_and_return_conditional_losses_102184Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЮtrace_0
.:, 2conv2d_relu_3/kernel
 : 2conv2d_relu_3/bias
Д2БЎ
ЃВ
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
Яnon_trainable_variables
аlayers
бmetrics
 вlayer_regularization_losses
гlayer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
є
дtrace_02е
.__inference_max_pool_2d_4_layer_call_fn_102189Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zдtrace_0

еtrace_02№
I__inference_max_pool_2d_4_layer_call_and_return_conditional_losses_102194Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zеtrace_0
.
d0
e1"
trackable_list_wrapper
.
d0
e1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
жnon_trainable_variables
зlayers
иmetrics
 йlayer_regularization_losses
кlayer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
є
лtrace_02е
.__inference_conv2d_relu_4_layer_call_fn_102203Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zлtrace_0

мtrace_02№
I__inference_conv2d_relu_4_layer_call_and_return_conditional_losses_102214Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zмtrace_0
.:,  2conv2d_relu_4/kernel
 : 2conv2d_relu_4/bias
Д2БЎ
ЃВ
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
нnon_trainable_variables
оlayers
пmetrics
 рlayer_regularization_losses
сlayer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
ю
тtrace_02Я
(__inference_flatten_layer_call_fn_102219Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zтtrace_0

уtrace_02ъ
C__inference_flatten_layer_call_and_return_conditional_losses_102225Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zуtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
фnon_trainable_variables
хlayers
цmetrics
 чlayer_regularization_losses
шlayer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
Х
щtrace_0
ъtrace_12
(__inference_dropout_layer_call_fn_102230
(__inference_dropout_layer_call_fn_102235Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zщtrace_0zъtrace_1
ћ
ыtrace_0
ьtrace_12Р
C__inference_dropout_layer_call_and_return_conditional_losses_102240
C__inference_dropout_layer_call_and_return_conditional_losses_102252Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zыtrace_0zьtrace_1
"
_generic_user_object
.
z0
{1"
trackable_list_wrapper
.
z0
{1"
trackable_list_wrapper
 "
trackable_list_wrapper
б
эnon_trainable_variables
юlayers
яmetrics
 №layer_regularization_losses
ёlayer_metrics
t	variables
utrainable_variables
vregularization_losses
x__call__
ђactivity_regularizer_fn
*y&call_and_return_all_conditional_losses
'ѓ"call_and_return_conditional_losses"
_generic_user_object
ь
єtrace_02Э
&__inference_dense_layer_call_fn_102261Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zєtrace_0

ѕtrace_02ь
E__inference_dense_layer_call_and_return_all_conditional_losses_102272Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zѕtrace_0
:		@2dense/kernel
:@2
dense/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Е
іnon_trainable_variables
їlayers
јmetrics
 љlayer_regularization_losses
њlayer_metrics
|	variables
}trainable_variables
~regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ю
ћtrace_02Я
(__inference_softmax_layer_call_fn_102292Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zћtrace_0

ќtrace_02ъ
C__inference_softmax_layer_call_and_return_conditional_losses_102303Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zќtrace_0
 :@2softmax/kernel
:2softmax/bias
 "
trackable_list_wrapper

0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14"
trackable_list_wrapper
0
§0
ў1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
Bў
/__inference_2d_convolution_layer_call_fn_101313input_1"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B§
/__inference_2d_convolution_layer_call_fn_101808inputs"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B§
/__inference_2d_convolution_layer_call_fn_101846inputs"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Bў
/__inference_2d_convolution_layer_call_fn_101607input_1"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
J__inference_2d_convolution_layer_call_and_return_conditional_losses_101939inputs"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
J__inference_2d_convolution_layer_call_and_return_conditional_losses_102039inputs"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
J__inference_2d_convolution_layer_call_and_return_conditional_losses_101666input_1"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
J__inference_2d_convolution_layer_call_and_return_conditional_losses_101725input_1"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
ЫBШ
$__inference_signature_wrapper_101770input_1"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
пBм
+__inference_batch_norm_layer_call_fn_102048inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
њBї
F__inference_batch_norm_layer_call_and_return_conditional_losses_102074inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
рBн
,__inference_conv2d_tanh_layer_call_fn_102083inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ћBј
G__inference_conv2d_tanh_layer_call_and_return_conditional_losses_102094inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
тBп
.__inference_max_pool_2d_1_layer_call_fn_102099inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
§Bњ
I__inference_max_pool_2d_1_layer_call_and_return_conditional_losses_102104inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
тBп
.__inference_conv2d_relu_1_layer_call_fn_102113inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
§Bњ
I__inference_conv2d_relu_1_layer_call_and_return_conditional_losses_102124inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
тBп
.__inference_max_pool_2d_2_layer_call_fn_102129inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
§Bњ
I__inference_max_pool_2d_2_layer_call_and_return_conditional_losses_102134inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
тBп
.__inference_conv2d_relu_2_layer_call_fn_102143inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
§Bњ
I__inference_conv2d_relu_2_layer_call_and_return_conditional_losses_102154inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
тBп
.__inference_max_pool_2d_3_layer_call_fn_102159inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
§Bњ
I__inference_max_pool_2d_3_layer_call_and_return_conditional_losses_102164inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
тBп
.__inference_conv2d_relu_3_layer_call_fn_102173inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
§Bњ
I__inference_conv2d_relu_3_layer_call_and_return_conditional_losses_102184inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
тBп
.__inference_max_pool_2d_4_layer_call_fn_102189inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
§Bњ
I__inference_max_pool_2d_4_layer_call_and_return_conditional_losses_102194inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
тBп
.__inference_conv2d_relu_4_layer_call_fn_102203inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
§Bњ
I__inference_conv2d_relu_4_layer_call_and_return_conditional_losses_102214inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
мBй
(__inference_flatten_layer_call_fn_102219inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
їBє
C__inference_flatten_layer_call_and_return_conditional_losses_102225inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
эBъ
(__inference_dropout_layer_call_fn_102230inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
(__inference_dropout_layer_call_fn_102235inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
C__inference_dropout_layer_call_and_return_conditional_losses_102240inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
C__inference_dropout_layer_call_and_return_conditional_losses_102252inputs"Г
ЊВІ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
њ
џtrace_02л
-__inference_dense_activity_regularizer_101090Љ
В
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ
	zџtrace_0

trace_02ш
A__inference_dense_layer_call_and_return_conditional_losses_102283Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
кBз
&__inference_dense_layer_call_fn_102261inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
љBі
E__inference_dense_layer_call_and_return_all_conditional_losses_102272inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
мBй
(__inference_softmax_layer_call_fn_102292inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
їBє
C__inference_softmax_layer_call_and_return_conditional_losses_102303inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
R
	variables
	keras_api

total

count"
_tf_keras_metric
c
	variables
	keras_api

total

count

_fn_kwargs"
_tf_keras_metric
уBр
-__inference_dense_activity_regularizer_101090x"Љ
В
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ
	
ѕBђ
A__inference_dense_layer_call_and_return_conditional_losses_102283inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
$:"М2Adam/batch_norm/gamma/m
#:!М2Adam/batch_norm/beta/m
1:/2Adam/conv2d_tanh/kernel/m
#:!2Adam/conv2d_tanh/bias/m
3:12Adam/conv2d_relu_1/kernel/m
%:#2Adam/conv2d_relu_1/bias/m
3:12Adam/conv2d_relu_2/kernel/m
%:#2Adam/conv2d_relu_2/bias/m
3:1 2Adam/conv2d_relu_3/kernel/m
%:# 2Adam/conv2d_relu_3/bias/m
3:1  2Adam/conv2d_relu_4/kernel/m
%:# 2Adam/conv2d_relu_4/bias/m
$:"		@2Adam/dense/kernel/m
:@2Adam/dense/bias/m
%:#@2Adam/softmax/kernel/m
:2Adam/softmax/bias/m
$:"М2Adam/batch_norm/gamma/v
#:!М2Adam/batch_norm/beta/v
1:/2Adam/conv2d_tanh/kernel/v
#:!2Adam/conv2d_tanh/bias/v
3:12Adam/conv2d_relu_1/kernel/v
%:#2Adam/conv2d_relu_1/bias/v
3:12Adam/conv2d_relu_2/kernel/v
%:#2Adam/conv2d_relu_2/bias/v
3:1 2Adam/conv2d_relu_3/kernel/v
%:# 2Adam/conv2d_relu_3/bias/v
3:1  2Adam/conv2d_relu_4/kernel/v
%:# 2Adam/conv2d_relu_4/bias/v
$:"		@2Adam/dense/kernel/v
:@2Adam/dense/bias/v
%:#@2Adam/softmax/kernel/v
:2Adam/softmax/bias/vл
J__inference_2d_convolution_layer_call_and_return_conditional_losses_101666 !()78FGUVdez{AЂ>
7Ђ4
*'
input_1џџџџџџџџџ(М
p 

 
Њ "3Ђ0

0џџџџџџџџџ

	
1/0 л
J__inference_2d_convolution_layer_call_and_return_conditional_losses_101725 !()78FGUVdez{AЂ>
7Ђ4
*'
input_1џџџџџџџџџ(М
p

 
Њ "3Ђ0

0џџџџџџџџџ

	
1/0 к
J__inference_2d_convolution_layer_call_and_return_conditional_losses_101939 !()78FGUVdez{@Ђ=
6Ђ3
)&
inputsџџџџџџџџџ(М
p 

 
Њ "3Ђ0

0џџџџџџџџџ

	
1/0 к
J__inference_2d_convolution_layer_call_and_return_conditional_losses_102039 !()78FGUVdez{@Ђ=
6Ђ3
)&
inputsџџџџџџџџџ(М
p

 
Њ "3Ђ0

0џџџџџџџџџ

	
1/0 Є
/__inference_2d_convolution_layer_call_fn_101313q !()78FGUVdez{AЂ>
7Ђ4
*'
input_1џџџџџџџџџ(М
p 

 
Њ "џџџџџџџџџЄ
/__inference_2d_convolution_layer_call_fn_101607q !()78FGUVdez{AЂ>
7Ђ4
*'
input_1џџџџџџџџџ(М
p

 
Њ "џџџџџџџџџЃ
/__inference_2d_convolution_layer_call_fn_101808p !()78FGUVdez{@Ђ=
6Ђ3
)&
inputsџџџџџџџџџ(М
p 

 
Њ "џџџџџџџџџЃ
/__inference_2d_convolution_layer_call_fn_101846p !()78FGUVdez{@Ђ=
6Ђ3
)&
inputsџџџџџџџџџ(М
p

 
Њ "џџџџџџџџџЈ
!__inference__wrapped_model_101034 !()78FGUVdez{9Ђ6
/Ђ,
*'
input_1џџџџџџџџџ(М
Њ "1Њ.
,
softmax!
softmaxџџџџџџџџџИ
F__inference_batch_norm_layer_call_and_return_conditional_losses_102074n !8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ(М
Њ ".Ђ+
$!
0џџџџџџџџџ(М
 
+__inference_batch_norm_layer_call_fn_102048a !8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ(М
Њ "!џџџџџџџџџ(МЙ
I__inference_conv2d_relu_1_layer_call_and_return_conditional_losses_102124l787Ђ4
-Ђ*
(%
inputsџџџџџџџџџ^
Њ "-Ђ*
# 
0џџџџџџџџџ^
 
.__inference_conv2d_relu_1_layer_call_fn_102113_787Ђ4
-Ђ*
(%
inputsџџџџџџџџџ^
Њ " џџџџџџџџџ^Й
I__inference_conv2d_relu_2_layer_call_and_return_conditional_losses_102154lFG7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
/
Њ "-Ђ*
# 
0џџџџџџџџџ
/
 
.__inference_conv2d_relu_2_layer_call_fn_102143_FG7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
/
Њ " џџџџџџџџџ
/Й
I__inference_conv2d_relu_3_layer_call_and_return_conditional_losses_102184lUV7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ "-Ђ*
# 
0џџџџџџџџџ 
 
.__inference_conv2d_relu_3_layer_call_fn_102173_UV7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ
Њ " џџџџџџџџџ Й
I__inference_conv2d_relu_4_layer_call_and_return_conditional_losses_102214lde7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ 
Њ "-Ђ*
# 
0џџџџџџџџџ 
 
.__inference_conv2d_relu_4_layer_call_fn_102203_de7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ 
Њ " џџџџџџџџџ Й
G__inference_conv2d_tanh_layer_call_and_return_conditional_losses_102094n()8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ(М
Њ ".Ђ+
$!
0џџџџџџџџџ(М
 
,__inference_conv2d_tanh_layer_call_fn_102083a()8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ(М
Њ "!џџџџџџџџџ(МW
-__inference_dense_activity_regularizer_101090&Ђ
Ђ
	
x
Њ " Д
E__inference_dense_layer_call_and_return_all_conditional_losses_102272kz{0Ђ-
&Ђ#
!
inputsџџџџџџџџџ	
Њ "3Ђ0

0џџџџџџџџџ@

	
1/0 Ђ
A__inference_dense_layer_call_and_return_conditional_losses_102283]z{0Ђ-
&Ђ#
!
inputsџџџџџџџџџ	
Њ "%Ђ"

0џџџџџџџџџ@
 z
&__inference_dense_layer_call_fn_102261Pz{0Ђ-
&Ђ#
!
inputsџџџџџџџџџ	
Њ "џџџџџџџџџ@Ѕ
C__inference_dropout_layer_call_and_return_conditional_losses_102240^4Ђ1
*Ђ'
!
inputsџџџџџџџџџ	
p 
Њ "&Ђ#

0џџџџџџџџџ	
 Ѕ
C__inference_dropout_layer_call_and_return_conditional_losses_102252^4Ђ1
*Ђ'
!
inputsџџџџџџџџџ	
p
Њ "&Ђ#

0џџџџџџџџџ	
 }
(__inference_dropout_layer_call_fn_102230Q4Ђ1
*Ђ'
!
inputsџџџџџџџџџ	
p 
Њ "џџџџџџџџџ	}
(__inference_dropout_layer_call_fn_102235Q4Ђ1
*Ђ'
!
inputsџџџџџџџџџ	
p
Њ "џџџџџџџџџ	Ј
C__inference_flatten_layer_call_and_return_conditional_losses_102225a7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ 
Њ "&Ђ#

0џџџџџџџџџ	
 
(__inference_flatten_layer_call_fn_102219T7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ 
Њ "џџџџџџџџџ	ь
I__inference_max_pool_2d_1_layer_call_and_return_conditional_losses_102104RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ф
.__inference_max_pool_2d_1_layer_call_fn_102099RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџь
I__inference_max_pool_2d_2_layer_call_and_return_conditional_losses_102134RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ф
.__inference_max_pool_2d_2_layer_call_fn_102129RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџь
I__inference_max_pool_2d_3_layer_call_and_return_conditional_losses_102164RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ф
.__inference_max_pool_2d_3_layer_call_fn_102159RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџь
I__inference_max_pool_2d_4_layer_call_and_return_conditional_losses_102194RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ф
.__inference_max_pool_2d_4_layer_call_fn_102189RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџЖ
$__inference_signature_wrapper_101770 !()78FGUVdez{DЂA
Ђ 
:Њ7
5
input_1*'
input_1џџџџџџџџџ(М"1Њ.
,
softmax!
softmaxџџџџџџџџџЅ
C__inference_softmax_layer_call_and_return_conditional_losses_102303^/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "%Ђ"

0џџџџџџџџџ
 }
(__inference_softmax_layer_call_fn_102292Q/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "џџџџџџџџџ