# Project Journal

## Entry 001 — Project Founded

**Date:** 2026-09-26

### Decision

Begin a long-term project to build a self-owned AI from the ground up.

### Goals

- Learn the technology instead of treating it as a black box.
- Eventually train my own model.
- Build the surrounding AI system myself.
- Avoid mandatory subscriptions and unnecessary dependence on commercial AI APIs.
- Develop the project as a personal body of work that can grow over time.

### First Technical Principle

The first AI does not need to be powerful.

It needs to be understandable.

### Status

Phase 0 — Foundation

## Entry 002 — Development Environment Established

**Date:** 2026-09-26

### Environment

- Linux development environment
- Python 3.x
- Git repository

### Decision

Begin the first learning experiment using Python's standard library only.

No machine-learning framework will be used for Experiment 001.

### Reason

I want to see the learning process directly instead of hiding it behind a library.

### Status

Phase 0 — Foundation  
Preparing for Phase 1 — First Learning System

## Entry 003 — First Learning Experiment

**Date:** 2026-09-26

### Experiment

A one-parameter learning system was created using only Python's standard library.

The model:

prediction = weight × input

Training examples:

1 → 2  
2 → 4  
3 → 6  
4 → 8  
5 → 10

Initial weight:

0.5

Learning rate:

0.001

Training epochs:

50

### Result

Final weight:

1.9944572607243007

Test input:

7

Test prediction:

13.961200825070105

The value 7 was not included in the training examples.

### Observation

The system learned an approximation of the relationship y = 2x and generalized it to an unseen input.

### Lesson

A learning system can improve a parameter by measuring error and repeatedly adjusting that parameter.

This experiment introduced:

- prediction
- error
- loss
- gradient
- learning rate
- repeated training
- generalization
- evaluation

### Important Understanding

The model does not contain the rule "multiply by 2."

It contains a learned parameter that moved toward a value that minimizes its error on the examples.

## Entry 004 — Adding a Bias

**Date:** 2026-09-26

### Experiment

The model structure was expanded from:

prediction = weight × input

to:

prediction = weight × input + bias

This introduced a second learned parameter: bias.

Training examples:

1 → 3  
2 → 5  
3 → 7  
4 → 9  
5 → 11

The underlying relationship was y = 2x + 1.

### First Result

After 100 epochs:

- Weight: 2.1242365655296824
- Bias: 0.5447696447693232
- Test input: 7
- Prediction: 15.4144256034771

### Second Result

After increasing training to 1000 epochs:

- Weight: 2.026847653543838
- Bias: 0.9016324002141546
- Test input: 7
- Expected: 15
- Prediction: 15.089565975021019

### Observation

Increasing the number of training steps moved both learned parameters closer to the relationship represented by the training data.

### Lesson

A simple model can learn more than one parameter. Weight determines how strongly the input contributes, while bias shifts the prediction.

This is the first experiment in which the model behaves like a tiny neuron with a learned weight and bias.

## Entry 004 — Gradient Descent in Practice

**Date:** 2026-09-26

### Experiment

Tested repeated gradient descent updates using one training example:

x = 2
y = 5

Initial parameters:

w = 0.5
b = 0

Learning rate:

0.001

Training steps:

20

### Results

Step 1:

- Prediction: 1.0
- Loss: 16.0
- Weight: 0.516
- Bias: 0.008

Step 5:

- Prediction: 1.15761596
- Loss: 14.76391511084672

Step 10:

- Prediction: 1.3459310100654365
- Loss: 13.3522201832014

Step 20:

- Prediction: 1.695325504657653
- Loss: 10.920873520166197
- Weight: 0.7913488998444306
- Bias: 0.14567444992221532

### Observation

Repeated gradient descent caused the loss to decrease from 16.0 to 10.92 while the prediction moved from 1.0 toward the correct answer of 5.

### Lesson

Gradient descent repeatedly changes model parameters in the direction that reduces loss.

A small learning rate produces smaller parameter updates, making the process slower but more controlled.

## Entry 005 — Learning Rate Experiment

**Date:** 2026-09-26

### Experiment

Tested five learning rates using the same training example:

x = 2
y = 5

Initial parameters:

w = 0.5
b = 0

The learning rates tested were:

0.0001
0.001
0.01
0.1
0.21

### Results

#### Learning rate: 0.0001

After 20 steps:

- Prediction: 1.0753198605424035
- Loss: 15.403114197052899
- Weight: 0.5316978162727445
- Bias: 0.015848908136372256

Learning was very slow.

#### Learning rate: 0.001

After 20 steps:

- Prediction: 1.695325504657653
- Loss: 10.920873520166197
- Weight: 0.7913488998444306
- Bias: 0.14567444992221532

The model improved steadily but had not reached the target.

#### Learning rate: 0.01

After 20 steps:

- Prediction: 4.459659312930802
- Loss: 0.2919680581024125
- Weight: 1.905477352655089
- Bias: 0.7027386763275443

The model learned much faster and moved close to the target.

#### Learning rate: 0.1

The model reached the target:

Prediction = 5.0
Loss = 0.0

Final parameters:

- Weight: 2.1
- Bias: 0.8

#### Learning rate: 0.21

The model became unstable.

After 20 steps:

- Prediction: 29.46363617936579
- Loss: 598.4694951163748
- Weight: -8.663999918920945
- Bias: -4.581999959460473

### Observation

The learning rate controls how large each parameter update is.

A learning rate that is too small can make learning very slow.

A suitable learning rate can move the model toward the target efficiently.

A learning rate that is too large can cause the model to overshoot the target repeatedly and become unstable.

### Lesson

Gradient descent depends not only on the direction of the gradient, but also on the size of each step.

The learning rate determines that step size.

### Additional Observation

A negative prediction is not inherently an error. It simply means the model's current numerical output is below zero.

The problem occurs when the prediction moves farther from the target and the loss increases.

### Status

Experiment 004 complete.

## Entry 006 — Multiple Inputs

**Date:** 2026-09-27

### Experiment

Expanded the neuron from one input to two inputs.

The model structure became:

prediction = w1 × x1 + w2 × x2 + bias

Training data:

(1, 1) → 6
(2, 1) → 8
(1, 2) → 9
(3, 2) → 13
(2, 3) → 14

The underlying relationship was:

y = 2x1 + 3x2 + 1

The model was not given that relationship.

### Initial Parameters

w1 = 0.5
w2 = 0.5
bias = 0

Learning rate:

0.001

Training epochs:

1000

### Result

Final weight 1:

2.007437596601594

Final weight 2:

2.9896676735925904

Final bias:

1.005845009552969

### Unseen Test

Test inputs:

x1 = 4
x2 = 2

Expected:

15

Predicted:

15.014930743144525

### Observation

The neuron learned separate weights for separate inputs and combined them to produce its prediction.

The learned parameters were close to the relationship represented by the training data.

### Lesson

Each input has its own learned weight.

The weight determines how strongly that input contributes to the prediction.

With multiple inputs, the neuron can combine multiple signals before producing an output.

### Status

Experiment 005 complete.

## Entry 007 — Two-Neuron Layer

**Date:** 2026-09-27

### Experiment

Expanded from a single neuron to a layer containing two neurons.

Both neurons received the same two inputs:

x1
x2

Each neuron had its own two weights and bias.

The layer was trained to learn:

y1 = 2x1 + x2

y2 = x1 + 3x2

### Initial Parameters

All weights started at:

0.5

All biases started at:

0

Learning rate:

0.001

Training epochs:

2000

### Result — Neuron 1

Weight 1:

1.9809329330954377

Weight 2:

0.9810192332833941

Bias:

0.07573069152144327

Expected relationship:

y1 = 2x1 + x2

### Result — Neuron 2

Weight 1:

0.9721315321064297

Weight 2:

2.9711802993693683

Bias:

0.11283761404099299

Expected relationship:

y2 = x1 + 3x2

### Unseen Test

Test inputs:

x1 = 4
x2 = 2

Expected output 1:

10

Predicted output 1:

9.961500890469981

Expected output 2:

10

Predicted output 2:

9.943724341205447

### Observation

Two neurons can receive the same inputs while learning different relationships through separate weights and biases.

### Lesson

A layer allows multiple neurons to process the same information in different ways.

Each neuron has its own parameters and produces its own output.

### Next Direction

Explore activation functions and the role they play in allowing neural networks to represent nonlinear relationships.

### Status

Experiment 006 complete.

## Entry 008 — Activation Function: ReLU

**Date:** 2026-09-27

### Experiment

Introduced the ReLU activation function and observed how it transforms a neuron's raw output.

The raw neuron was:

z = wx + b

with:

w = 2
b = -3

The activation function was:

a = max(0, z)

### Results

Input -2:

z = -7
ReLU = 0

Input -1:

z = -5
ReLU = 0

Input 0:

z = -3
ReLU = 0

Input 1:

z = -1
ReLU = 0

Input 2:

z = 1
ReLU = 1

Input 3:

z = 3
ReLU = 3

Input 4:

z = 5
ReLU = 5

### Observation

ReLU passes positive values through unchanged and changes negative values to zero.

### Second Example

Using:

w = 3
b = -4

the raw outputs for inputs 1, 2, and 3 were:

-1
2
5

After ReLU:

0
2
5

### Lesson

An activation function transforms the output of a neuron.

ReLU itself does not learn parameters. It changes the forward signal and determines whether a gradient can pass backward.

For ReLU:

f(z) = max(0, z)

and its derivative is:

f'(z) = 0 when z < 0
f'(z) = 1 when z > 0

### Status

Experiment 007 complete.

## Entry 009 — Trainable ReLU

**Date:** 2026-09-27

### Experiment

Added ReLU to a trainable neuron.

The forward pass became:

z = weight × input + bias

prediction = max(0, z)

The gradient became dependent on the derivative of ReLU:

weight_gradient = 2 × error × input × ReLU_derivative(z)

bias_gradient = 2 × error × ReLU_derivative(z)

### Result

Using:

x = 2
y = 5
weight = 0.5
bias = 0
learning rate = 0.001

the neuron remained in the positive region during the 20 training steps.

The results matched the earlier gradient descent experiment because ReLU_derivative(z) remained 1.

### Observation

When z is positive, ReLU passes the value forward and its derivative is 1, allowing the gradient to pass backward.

When z is negative, ReLU outputs 0 and its derivative is 0, blocking the gradient from reaching the weight and bias.

### Lesson

An activation function affects both the forward output and the backward learning signal.

### Status

Experiment 008 complete.

## Entry 010 — ReLU Gradient Behavior

**Date:** 2026-09-27

### Experiment

Compared a positive and negative pre-activation value to observe how ReLU changes the gradient.

### Positive z

z = 1

Prediction = 1

Error = -4

Loss = 16

ReLU derivative = 1

Weight gradient = -16

Bias gradient = -8

### Negative z

z = -2

Prediction = 0

Error = -5

Loss = 25

ReLU derivative = 0

Weight gradient = 0

Bias gradient = 0

### Observation

A neuron can have a large prediction error while receiving a zero gradient when ReLU is inactive.

### Lesson

The size of the error alone does not determine the parameter update. The activation function can determine whether the gradient reaches the parameter.

### Status

Experiment 009 complete.

## Entry 011 — First Multi-Layer Network

**Date:** 2026-09-27

### Experiment

Built the first multi-layer neural network.

Architecture:

2 inputs → 2 ReLU hidden neurons → 1 output

The network was trained to learn XOR:

(0, 0) → 0
(0, 1) → 1
(1, 0) → 1
(1, 1) → 0

### Training

Learning rate:

0.05

Epochs:

1000

### Result

Final total loss:

6.409494854920721e-31

The final loss was effectively zero.

Predictions:

(0, 0) → 2.220446049250313e-16
(0, 1) → 0.9999999999999996
(1, 0) → 0.9999999999999996
(1, 1) → 4.440892098500626e-16

The values near zero are floating-point representations of values effectively equal to zero.

### Observation

The network successfully learned the XOR relationship using a hidden layer and ReLU activation.

### Lesson

A hidden layer combined with a nonlinear activation allows a neural network to represent relationships that a single linear neuron cannot.

This experiment also connected the forward pass and backward pass into one multi-layer training process.

The backward pass propagated gradients from the loss through the output neuron, through the ReLU activations, and into the hidden-layer parameters.

### Status

Experiment 010 complete.

## Entry 012 — Reusable Neuron

**Date:** 2026-09-27

### Experiment

Created a reusable Neuron class containing:

- weights
- bias
- ReLU
- ReLU derivative
- forward pass

The forward pass uses:

z = sum(weight × input) + bias

a = max(0, z)

### Test

Weights:

[0.5, 0.5]

Bias:

0.0

Inputs:

[2, 1]

### Result

Raw output:

1.5

ReLU output:

1.5

ReLU derivative:

1

### Lesson

The neuron can now be represented as a reusable component instead of rewriting its forward-pass logic in every experiment.

The neuron describes how an input is transformed. Training will remain a separate responsibility.

### Status

Experiment 011 complete.

## Entry 013 — Reusable Layer

**Date:** 2026-09-27

### Experiment

Created a reusable Layer class that contains multiple Neuron objects.

The layer sends the same inputs to each neuron and collects their raw and activated outputs.

### Test

Inputs:

[2, 1]

Neuron 1:

weights = [1.0, 2.0]
bias = 0.0

Neuron 2:

weights = [3.0, 1.0]
bias = 1.0

### Result

Raw outputs:

[4.0, 8.0]

Activated outputs:

[4.0, 8.0]

### Observation

The layer successfully processed the same inputs through multiple neurons and returned their outputs as a collection.

### Lesson

A neural-network layer can be represented as a reusable collection of neurons.

Each neuron has its own weights and bias while receiving the same input vector.

### Status

Experiment 012 complete.

## Entry 014 — Automatic Layer Construction

**Date:** 2026-09-27

### Experiment

Changed the Layer class so it can automatically create any requested number of neurons with the required number of input weights.

Test configuration:

- Inputs: 2
- Neurons: 3
- Random seed: 0
- Bias for each neuron: 0

### Parameter Count

Each neuron requires:

2 weights + 1 bias = 3 parameters

Three neurons therefore require:

3 × 3 = 9 parameters

### Test Input

[2, 1]

### Result

Neuron 1:

Weights = [0.6888437030500962, 0.515908805880605]

Raw output = 1.8935962119807974

ReLU output = 1.8935962119807974

Neuron 2:

Weights = [-0.15885683833831, -0.4821664994140733]

Raw output = -0.7998801760906933

ReLU output = 0

Neuron 3:

Weights = [0.02254944273721704, -0.19013172509917142]

Raw output = -0.14503283962473734

ReLU output = 0

### Observation

All neurons received the same input but produced different results because each neuron had its own independently initialized weights.

ReLU allowed one neuron to remain active while the other two produced zero outputs.

### Lesson

A layer can automatically create and manage multiple neurons.

Random initialization gives different neurons different starting parameters.

The layer can now be represented mathematically using weights, inputs, and biases as vectors and matrices.

### Status

Experiment 013 complete.


## Entry 015 — Training Loop

**Date:** 2026-09-27

### Experiment

Turned the manually derived forward pass, loss calculation, ReLU gradient, backpropagation, and gradient descent update into a repeated training loop.

The experiment used:

- 2 inputs
- 2 output neurons
- ReLU activation
- Mean squared error with a 1/2 factor
- Learning rate: 0.01
- Training steps: 100

Initial parameters:

W = [
[2.0, 1.0],
[3.0, 4.0]
]

b = [1.0, -2.0]

Input:

x = [5.0, 2.0]

Target:

y = [10.0, 20.0]

### Initial Forward Pass

The initial output was:

[13.0, 21.0]

Initial loss:

5.0

### Learning Process

The training loop repeatedly performed:

forward pass
→ loss
→ backpropagation
→ gradient calculation
→ gradient descent update

Using a learning rate of 0.01, the model quickly moved toward the target.

At step 10:

- Loss: approximately 0.0040
- Output: approximately [10.0847, 20.0282]

At step 20:

- Loss: approximately 0.0000 at the displayed precision
- Output: approximately [10.0024, 20.0008]

By step 30 and beyond, the printed output was effectively [10.0, 20.0].

### Observation

The model improved automatically when the same learning procedure was repeated.

The loss decreased from 5.0 to a value effectively zero at the displayed precision.

### Lesson

A neural network learns through repetition of a small set of operations:

1. Make a prediction.
2. Measure the error.
3. Calculate how the parameters contributed to that error.
4. Adjust the parameters.
5. Repeat.

This experiment turns the backpropagation math into an actual training loop.

### Important Understanding

The training loop is the mechanism that repeatedly applies the gradients.

The gradient determines the direction of the parameter update.

The learning rate determines the size of the update.

### Next Direction

Replace the manually named weights and biases in this experiment with the reusable Layer and Neuron components already developed in the project.

### Status

Experiment 014 complete.

## Entry 016 — Trainable Reusable Layer

**Date:** 2026-09-27

### Experiment

Extended the reusable Neuron and Layer components so they can participate in training.

The Neuron now stores:

- inputs from the forward pass
- raw output
- weight gradients
- bias gradient

The Neuron can now perform:

- forward pass
- backward pass
- parameter update

The Layer can now:

- forward inputs through all neurons
- propagate gradients backward through all neurons
- update all neuron parameters

### Test

Architecture:

2 inputs → 2 trainable neurons

Input:

[5.0, 2.0]

Target:

[10.0, 20.0]

Initial weights:

[
[2.0, 1.0],
[3.0, 4.0]
]

Initial biases:

[1.0, -2.0]

Learning rate:

0.01

Training steps:

100

### Result

Initial output:

[13.0, 21.0]

Initial loss:

5.0

At step 10:

- Loss: approximately 0.0040
- Output: approximately [10.0847, 20.0282]

At step 20:

- Output: approximately [10.0024, 20.0008]

By step 50, the output was effectively:

[10.0, 20.0]

### Observation

The reusable Layer produced the same learning behavior as the previous manually written training loop.

The difference is that the training process is now handled by reusable Neuron and Layer objects instead of manually naming every parameter.

### Lesson

Reusable neural-network components can contain both forward computation and the information required for backpropagation and parameter updates.

This is a major step toward building the network as a collection of reusable components rather than a collection of one-off experiments.

### Important Understanding

The Layer does not need to know the individual meaning of every weight.

It asks each Neuron to:

1. process its inputs,
2. calculate its gradients,
3. update its own parameters.

The Layer coordinates those operations across the neurons.

### Status

Experiment 015 complete.

## Entry 017 — Reusable Multi-Layer Network

**Date:** 2026-09-27

### Experiment

Built a multi-layer neural network using the reusable Neuron and Layer components.

Architecture:

2 inputs
→ 2 ReLU hidden neurons
→ 1 linear output neuron

This experiment replaced the manually named parameters from the earlier XOR experiment with reusable trainable Layer objects.

### Training Data

[0, 0] → 0
[0, 1] → 1
[1, 0] → 1
[1, 1] → 0

This is the XOR relationship.

### Training

Learning rate:

0.05

Epochs:

1000

The training loop performed:

forward pass
→ loss
→ output-layer backward pass
→ hidden-layer backward pass
→ parameter updates

### Results

Epoch 1:

Total loss ≈ 0.164122570751

Epoch 10:

Total loss ≈ 0.035717989313

Epoch 100:

Total loss ≈ 0.000920521315

Epoch 500:

Total loss ≈ 0.000000000041

Epoch 1000:

Total loss ≈ 0.000000000000

### Final XOR Results

Input [0, 0]:

Expected = 0.0
Predicted ≈ 0.000000000166

Input [0, 1]:

Expected = 1.0
Predicted ≈ 0.999999999923

Input [1, 0]:

Expected = 1.0
Predicted ≈ 0.999999999903

Input [1, 1]:

Expected = 0.0
Predicted ≈ 0.000000000001

The very small nonzero values are floating-point representations of values effectively equal to zero.

### Observation

The backward pass successfully propagated the learning signal from the output layer into the hidden layer.

The reusable Layer abstraction can therefore participate in a complete multi-layer training process.

### Lesson

A neural network can be constructed by connecting reusable layers.

Each layer performs its own forward computation and backward gradient calculation while passing information between layers.

This separates the network into reusable components instead of requiring every weight and gradient to be manually written.

### Important Understanding

Backpropagation is not limited to one layer.

The gradient can travel backward through multiple layers using the chain rule:

output loss
→ output layer
→ hidden layer
→ earlier layers

This experiment demonstrates that mechanism with the project's own code.

### Status

Experiment 016 complete.

## Entry 018 — Reusable Network

**Date:** 2026-09-27

### Experiment

Created a reusable Network class to coordinate multiple Layer objects.

The Network now handles:

- forward propagation through all layers
- backward propagation in reverse layer order
- parameter updates across all layers

The goal was to remove the need for each experiment to manually coordinate individual layers.

### Architecture

2 inputs
→ 2 ReLU hidden neurons
→ 1 linear output neuron

### Training Data

[0, 0] → 0
[0, 1] → 1
[1, 0] → 1
[1, 1] → 0

### Training

Learning rate:

0.05

Epochs:

1000

The training loop now uses:

network.forward()
→ loss
→ network.backward()
→ network.update()

### Results

Epoch 1:

Total loss ≈ 0.164122570751

Epoch 10:

Total loss ≈ 0.035717989313

Epoch 100:

Total loss ≈ 0.000920521315

Epoch 500:

Total loss ≈ 0.000000000041

Epoch 1000:

Total loss ≈ 0.000000000000

### Final XOR Results

Input [0, 0]:

Expected = 0.0
Predicted ≈ 0.000000000166

Input [0, 1]:

Expected = 1.0
Predicted ≈ 0.999999999923

Input [1, 0]:

Expected = 1.0
Predicted ≈ 0.999999999903

Input [1, 1]:

Expected = 0.0
Predicted ≈ 0.000000000001

### Observation

The reusable Network produced the same results as the previous multi-layer experiment.

The main change was architectural: the Network now coordinates the layers and hides the details of how many layers exist.

### Lesson

A neural network can be represented as a reusable sequence of layers.

Forward propagation moves information from the input toward the output.

Backward propagation moves gradients from the output toward the input.

The Network class provides the structure that connects those operations.

### Important Understanding

The network itself does not need to know the internal math of each neuron.

It coordinates the layers:

input
→ layer
→ layer
→ output

and then reverses that path during backpropagation.

### Status

Experiment 017 complete.

## Entry 019 — Automatic Network Construction

**Date:** 2026-09-27

### Experiment

Extended the reusable Network class so it can automatically construct its layers from a simple architecture description.

Instead of manually creating each Layer, the network can now be defined with:

number_of_inputs = 2
layer_sizes = [2, 1]
activations = ["relu", "linear"]

The Network creates the required layers and connects their input/output sizes automatically.

### Architecture

2 inputs
→ 2 ReLU hidden neurons
→ 1 linear output neuron

### Training Data

[0, 0] → 0
[0, 1] → 1
[1, 0] → 1
[1, 1] → 0

### First Run — Seed 0

The automatically constructed network did not successfully learn XOR.

Final loss:

0.350569675723

Final predictions:

[0, 0] → 0.683610867659947
[0, 1] → 0.683610867659947
[1, 0] → 0.683610867659947
[1, 1] → approximately 0

### Second Run — Seed 1

Changing the random seed produced a successful training run.

Epoch 1 loss:

0.203837789774

Epoch 10 loss:

0.079870619168

Epoch 100 loss:

0.000363064324

Epoch 500 loss:

0.000000000000

Epoch 1000 loss:

0.000000000000

Final predictions:

[0, 0] → approximately 0
[0, 1] → approximately 1
[1, 0] → approximately 1
[1, 1] → approximately 0

### Observation

The automatic Network construction worked in both runs.

The difference was the initial random parameters.

With seed 0, the network became stuck in a state where multiple inputs produced the same output and training stopped improving.

With seed 1, the same architecture and training process successfully learned XOR.

### Lesson

Network architecture and parameter initialization are separate concerns.

A correct architecture does not guarantee successful training from every random initialization.

Random initialization affects where optimization begins and can determine whether a small network successfully learns a particular problem.

### Important Understanding

The Network now separates three responsibilities:

1. Architecture — which layers and sizes exist.
2. Computation — forward and backward propagation.
3. Learning — updating parameters using gradients.

The architecture can now be described without manually constructing every layer.

### Status

Experiment 018 complete.

### Follow-up

Future experiments should investigate initialization more systematically rather than depending on a single seed.


## Entry 020 — Initialization Test

**Date:** 2026-09-27

### Experiment

Tested the effect of random parameter initialization by training the same automatically constructed XOR network with ten different random seeds.

Architecture:

2 inputs
→ 2 ReLU hidden neurons
→ 1 linear output neuron

Training data:

[0, 0] → 0
[0, 1] → 1
[1, 0] → 1
[1, 1] → 0

Learning rate:

0.05

Training epochs:

1000

Seeds tested:

0 through 9

### Results

Seed 0:

Loss ≈ 0.333763992254
Success = False

Seed 1:

Loss ≈ 0.000000000000
Success = True

Seed 2:

Loss ≈ 0.333763992254
Success = False

Seed 3:

Loss ≈ 0.500000863377
Success = False

Seed 4:

Loss ≈ 0.500000863377
Success = False

Seed 5:

Loss ≈ 0.000000000053
Success = True

Seed 6:

Loss ≈ 0.333763992254
Success = False

Seed 7:

Loss ≈ 0.333487085225
Success = False

Seed 8:

Loss ≈ 0.333472106898
Success = False

Seed 9:

Loss ≈ 0.333472106898
Success = False

The experiment treated a total loss below 1e-8 as successful.

### Additional Investigation

Seed 5 produced:

[0, 0] → approximately 0.00000887
[0, 1] → approximately 0.99999674
[1, 0] → approximately 0.99999597
[1, 1] → approximately 0.00000017

Its total loss was approximately 5.3e-11.

### Observation

The same network architecture and training procedure produced different results depending only on the initial random parameters.

Two of the ten tested seeds reached the experiment's success threshold within 1000 epochs.

Several other seeds became stuck at nonzero loss values.

### Lesson

Random initialization is an important part of neural-network training.

The architecture alone does not determine the training result. The starting parameter values can affect whether optimization reaches a useful solution within a given number of training steps.

### Important Understanding

A random seed is useful during development because it makes an experiment reproducible.

Testing multiple seeds is also useful because a single successful run does not show how robust the training process is.

### Status

Experiment 019 complete.

### Next Direction

Investigate initialization strategies and how they affect gradient flow before moving to larger networks.



## Entry 021 — Initialization Scale

**Date:** 2026-09-27

### Experiment

Tested how the scale of the initial weights affects training.

The same automatically constructed XOR network was used for every run:

2 inputs
→ 2 ReLU hidden neurons
→ 1 linear output neuron

Training data:

[0, 0] → 0
[0, 1] → 1
[1, 0] → 1
[1, 1] → 0

The random seed, learning rate, architecture, and training duration were held constant.

Learning rate:

0.05

Training epochs:

1000

The initial weights produced by the network were multiplied by different scale values.

### Results

Scale 0.0:

Loss ≈ 0.500000863377
Success = False

Scale 0.001:

Loss ≈ 0.000000000000
Success = True

Scale 0.01:

Loss ≈ 0.000000000000
Success = True

Scale 0.1:

Loss ≈ 0.000000000000
Success = True

Scale 1.0:

Loss ≈ 0.000000000000
Success = True

Scale 10.0:

Loss ≈ 0.500000863377
Success = False

Scale 100.0:

Loss ≈ 0.500000863377
Success = False

### Observation

The initialization scale affected whether the same network successfully learned XOR.

Zero initialization failed.

Very small through moderate initialization scales successfully learned the task.

Very large initialization scales failed under the same learning rate and training budget.

### Lesson

Initialization is not simply about making weights random.

The magnitude of the starting parameters also matters.

If all parameters begin at zero, neurons can remain identical and fail to learn different features.

If parameters begin too large, the same learning rate can produce unsuitable updates and prevent successful training.

### Important Understanding

Initialization interacts with optimization.

The starting values determine the initial activations and gradients, while the learning rate determines how far parameters move during each update.

A useful initialization strategy therefore needs to consider both the network architecture and the optimization process.

### Status

Experiment 020 complete.

### Next Direction

Investigate how initialization can be chosen systematically instead of selecting a scale manually.


## Entry 022 — He Initialization

**Date:** 2026-09-27

### Experiment

Compared the project's existing uniform random initialization with He initialization.

The same XOR network and training procedure were used across ten random seeds.

Architecture:

2 inputs
→ 2 ReLU hidden neurons
→ 1 linear output neuron

Training data:

[0, 0] → 0
[0, 1] → 1
[1, 0] → 1
[1, 1] → 0

Learning rate:

0.05

Training epochs:

1000

The He initialization was applied to the ReLU hidden layer.

The linear output layer was left with its existing initialization.

### He Initialization

For the ReLU hidden layer, the standard deviation was calculated as:

sqrt(2 / fan_in)

Weights were sampled from a zero-centered normal distribution using that standard deviation.

### Results

Uniform initialization:

Successful runs = 2/10

He initialization:

Successful runs = 3/10

### Observation

He initialization produced one additional successful run in this ten-seed test.

However, most runs still did not reach the success threshold.

The result therefore shows that initialization strategy can affect training behavior, but this experiment does not establish that He initialization is universally better.

### Lesson

Initialization strategies are designed around the behavior of the activation function and the number of inputs to a layer.

For ReLU networks, He initialization uses a scale based on the layer's input size.

This gives a more principled starting point than selecting an arbitrary weight scale.

### Important Understanding

The initialization experiment also highlighted an important distinction:

- ReLU hidden layers use one type of initialization strategy.
- A linear output layer may require different treatment.

Initialization should therefore be considered together with the layer's activation function and architecture.

### Status

Experiment 021 complete.

### Next Direction

Measure the actual activation and gradient values produced by different initialization strategies to understand why some initializations train successfully while others become stuck.

## Entry 023 — Activation and Gradient Measurements

### Question

Why do different initialization strategies produce different training behavior?

Experiment 022 measured the initial hidden-layer ReLU activations and gradients produced by uniform and He initialization across the same ten seeds used in Experiment 021.

### Results

Uniform initialization:

- Active activation fraction = 0.325
- Mean activation = 0.215116
- Mean absolute gradient = 0.111849
- Zero-gradient fraction = 0.750000
- Successful runs = 2/10

He initialization:

- Active activation fraction = 0.4375
- Mean activation = 0.439705
- Mean absolute gradient = 0.153237
- Zero-gradient fraction = 0.666667
- Successful runs = 3/10

### Observation

He initialization produced more active hidden-layer ReLU outputs on average.

It also produced a lower fraction of zero gradients and a higher mean absolute gradient than uniform initialization.

One uniform-initialization run had a completely inactive hidden layer:

- active activation fraction = 0.000
- zero-gradient fraction = 1.000
- final loss = 0.500000863377
- success = False

This provides a concrete example of a network becoming stuck because the ReLU hidden layer did not pass useful gradients during the measured samples.

### Interpretation

The measurements support the idea that initialization affects training indirectly through the distribution of activations and gradients.

For this network, He initialization produced a healthier initial distribution than the tested uniform initialization, but the difference was not large enough to make every run successful.

The experiment therefore explains part of the behavior observed in Experiment 021 without establishing that activation statistics alone determine whether training succeeds.

### Lesson

Initialization is not only about the initial size of the weights.

It also affects:

- which ReLU units are active,
- how much signal reaches later layers,
- how much gradient can propagate backward,
- and whether a unit can become inactive and remain difficult to train.

### Important Understanding

A useful way to study neural-network training is to connect three levels of evidence:

1. Initialization values
2. Activation and gradient behavior
3. Final training outcome

Experiment 022 connects the first and second levels and provides a measurable explanation for part of the third.

### Status

Experiment 022 complete.

### Next Direction

Test whether activation and gradient behavior changes over the course of training by measuring these values at multiple checkpoints rather than only at initialization.

## Experiment 023 — Training Activation Checkpoints

Date: 2026-09-27

### Question
Does initialization affect activation survival and gradient flow during training?

### Setup
Compared uniform initialization against He initialization using:
- XOR training data
- 2 input neurons
- 2 hidden ReLU neurons
- 1 linear output neuron
- learning rate 0.05
- 1000 epochs
- seeds 0–9
- checkpoints 0, 1, 10, 100, 500, 1000

### Findings
He initialization produced stronger overall training behavior than uniform initialization, but remained highly seed-dependent.

Uniform initialization produced complete activation collapse for some seeds. He initialization reduced the frequency and severity of complete collapse, but did not eliminate it.

Successful He seeds reached near-zero loss, while several seeds converged to loss plateaus around 0.25 or 0.333.

### Conclusion
Initialization is a major contributor to training stability, but initialization alone does not explain the seed-dependent failures.

Next question: determine whether individual hidden neurons die during training or whether failures can occur while neurons remain active.

---

## Experiment 024 — Individual Neuron Lifetime

Date: 2026-09-27

### Question
When He initialization fails, do individual hidden ReLU neurons die at initialization or during training?

### Setup
Used the same network and training procedure as Experiment 023, with He initialization only.

Measured each hidden neuron independently at epochs 0, 1, 10, 100, 500, and 1000:
- active fraction
- mean activation
- mean absolute gradient
- zero-gradient fraction
- activation pattern across the four XOR inputs
- weight norm

### Findings
Two distinct failure modes appeared.

1. Dead-neuron failure:
Some neurons were already completely inactive at epoch 0, while others became permanently inactive during training.

Examples:
- Seed 5: neuron 0 was dead at initialization.
- Seed 7: neuron 1 was dead at initialization.
- Seed 6: neuron 1 died by epoch 1.
- Seed 0: neuron 0 died between epochs 10 and 100.
- Seed 3: neuron 1 died between epochs 10 and 100.

2. Non-dead failure:
Seeds 1 and 9 did not lose both neurons, yet still converged to a loss plateau near 0.25.

### Conclusion
Neuron death is a real failure mechanism, but it is not the complete explanation for failed training.

There is another failure mode in which hidden neurons remain active but learn a representation that does not allow the final linear neuron to solve XOR.

Next question: inspect the hidden representation produced by successful and unsuccessful seeds.

---

## Experiment 025 — Hidden Representation

Date: 2026-09-27

### Question
When neurons survive, does the learned hidden representation determine whether XOR can be solved?

### Setup
Used the same He-initialized 2-2-1 network and training procedure.

At epochs 0, 1, 10, 100, 500, and 1000, recorded:
- hidden neuron activations for all four XOR inputs
- output weights
- output bias
- predictions

### Findings
Successful runs developed hidden representations that allowed the linear output neuron to separate the XOR cases.

Seed 2 eventually produced:
- [0,0] -> prediction 0
- [0,1] -> prediction 1
- [1,0] -> prediction 1
- [1,1] -> prediction 0

Seed 4 and seed 8 similarly reached essentially perfect XOR predictions.

Failed runs can retain active neurons without producing a useful separable representation.

For example, seed 1 retains active hidden units but eventually predicts approximately:
- [0,0] -> 0.5128
- [0,1] -> 0.5128
- [1,0] -> 1.0000
- [1,1] -> 0.0000

This produces a loss plateau because two inputs remain indistinguishable to the learned representation.

### Conclusion
The experiments now show that seed-dependent failure has at least two mechanisms:

1. ReLU neurons can become permanently inactive.
2. Neurons can remain active but converge to a hidden representation that cannot support the required XOR separation.

Experiment 026 should isolate the geometry of the hidden representation and determine what property distinguishes successful representations from failed ones.

---

## Experiment 026 — Representation Geometry

Date: 2026-09-27

### Question
Does the geometric separability of the hidden representation distinguish successful and failed XOR training?

### Setup
Used the same He-initialized 2-2-1 network from Experiments 024 and 025.

The four XOR inputs were mapped into the two-dimensional hidden ReLU representation.

The positive class consisted of:
- [0,1]
- [1,0]

The negative class consisted of:
- [0,0]
- [1,1]

Measured the distance between the two class line segments in hidden space. A nonzero distance indicates that the two classes are linearly separable.

### Findings

At epoch 1000, every successful seed had a nonzero separation gap:

- Seed 2: gap 0.565872044
- Seed 4: gap 0.477392712
- Seed 8: gap 0.527839013

Every failed seed had a gap of 0.

The timing of separability was also important.

Seed 0 began with a separable representation (gap 0.344778929) but lost separability by epoch 100 and eventually converged to loss approximately 0.33347.

Seed 6 also began separable (gap 0.072356197) but lost separability by epoch 1 and eventually converged to loss approximately 0.33349.

Conversely, successful seeds could begin nonseparable and develop separability during training.

Seed 2 became separable between epochs 100 and 500.

Seed 4 became separable between epochs 100 and 500.

Seed 8 became separable between epochs 10 and 100.

Seeds 1 and 9 remained nonseparable and converged to loss plateaus near 0.25.

### Conclusion
Final hidden-space separability strongly corresponds with successful XOR training in this experiment.

Initialization determines the starting geometry, but training dynamics can either create the required separation or destroy it.

The next question is when the separation transition occurs and what changes immediately before the network becomes separable or loses separability.


---

## Experiment 027 — Separability Transition Timeline

Date: 2026-09-27

### Question
At what point during training does the hidden representation become separable, or lose separability?

### Setup
Used the same He-initialized 2-2-1 XOR network from Experiments 024–026.

Instead of checking only selected checkpoints, hidden-space separability was evaluated after every training epoch from 1 through 1000.

Recorded every transition between:
- separable
- nonseparable

For every transition, the hidden representation and loss were recorded.

### Findings

Successful runs developed separability during training:

- Seed 8: nonseparable -> separable at epoch 80.
- Seed 2: nonseparable -> separable at epoch 131.
- Seed 4: nonseparable -> separable at epoch 146.

All three ultimately reached essentially zero loss.

Failed runs could also lose an initially separable representation:

- Seed 6: separable -> nonseparable at epoch 1.
- Seed 0: separable -> nonseparable at epoch 27.

These runs eventually converged to loss near 0.333.

Four other failed seeds (1, 3, 5, and 7) never crossed into separability during the 1000-epoch run.

Seed 9 also remained nonseparable throughout training and converged to a loss near 0.25.

### Conclusion
Training success is associated with entering and maintaining a linearly separable hidden representation.

The experiment also shows that separability is not determined solely by initialization. Training can create the required representation or destroy one that existed initially.

The next question is what changes immediately before these separability transitions.


---

## Experiment 028 — Transition Microscope

Date: 2026-09-27

### Question
What changes immediately before and after the hidden representation becomes separable or loses separability?

### Setup
Used He initialization with the same 2-2-1 XOR network.

Focused on the five seeds with meaningful separability transitions from Experiment 027:
- Seed 0: YES -> NO at epoch 27
- Seed 2: NO -> YES at epoch 131
- Seed 4: NO -> YES at epoch 146
- Seed 6: YES -> NO at epoch 1
- Seed 8: NO -> YES at epoch 80

For each transition, inspected a five-epoch window before and after the transition.

Recorded:
- loss
- hidden-space separation gap
- hidden activations for all four inputs
- output weights and bias
- mean hidden-neuron gradients
- zero-gradient fractions

### Findings

#### Seed 0 — separability lost

Before the transition, the hidden representation was still separable but the gap was shrinking:

- epoch 22: gap 0.084172239
- epoch 23: gap 0.065868940
- epoch 24: gap 0.047404318
- epoch 25: gap 0.029383488
- epoch 26: gap 0.011260454

At epoch 27 the gap reached zero.

The important change was neuron 0 becoming inactive for the [1,0] input:

- epoch 26: neuron 0 activation for [1,0] = 0.011262
- epoch 27: neuron 0 activation for [1,0] = 0.000000

At the transition, neuron 0's mean gradient became exactly zero and its zero-gradient fraction became 1.000.

Afterward, neuron 0 remained permanently inactive and the representation remained nonseparable.

#### Seed 6 — separability lost immediately

The network began with a separable representation with gap 0.072356197.

After the first training epoch, the [0,1] activation of neuron 1 reached zero:

- epoch 0: neuron 1 [0,1] = 0.073127
- epoch 1: neuron 1 [0,1] = 0.000000

Neuron 1's mean gradient became exactly zero and its zero-gradient fraction became 1.000.

The separation gap immediately dropped to zero and remained there.

#### Seed 2 — separability formed without neuron death

The network was nonseparable through epoch 130.

At epoch 131, it became separable with gap 0.004320596.

The hidden points were:

- [0,0] -> (0.000000, 0.015828)
- [0,1] -> (0.000000, 0.011241)
- [1,0] -> (1.416142, 0.515809)
- [1,1] -> (0.096308, 0.511223)

Both hidden neurons remained trainable. Mean gradients were nonzero for both neurons.

The separation gap then increased during subsequent epochs.

#### Seed 4 — separability formed through a geometric boundary crossing

The representation remained nonseparable through epoch 145.

At epoch 146, the gap became positive:

0.002305559

The transition occurred while both neurons remained active and trainable.

The hidden points moved only slightly, but their geometric ordering changed enough for the two class segments to stop intersecting.

The gap then increased over subsequent epochs.

#### Seed 8 — gradual geometric separation

The network remained nonseparable through epoch 79.

At epoch 80 the gap became positive:

0.004042234

No hidden neuron became permanently inactive.

Between epochs 75 and 80, the hidden representations moved continuously while the gap changed from zero to positive.

The gap then grew rapidly:

- epoch 80: 0.004042234
- epoch 81: 0.019419898
- epoch 82: 0.034943003
- epoch 83: 0.050594217
- epoch 84: 0.066354879
- epoch 85: 0.082205062

### Conclusion

Experiment 028 identifies two different transition mechanisms.

1. A separable representation can be destroyed when a ReLU neuron crosses zero for an important training input. When this happens, the corresponding gradient becomes permanently zero and the neuron can no longer move that input back into the active region.

2. A separable representation can also form without neuron death. In successful seeds 2, 4, and 8, the hidden points gradually move until the positive and negative class segments become geometrically separable.

Therefore, the important variable is not simply whether neurons are alive.

The deeper variable is whether training dynamics move the hidden representation toward or away from a geometry that the final linear neuron can separate.

Next question: determine whether the gradient direction itself predicts whether the hidden representation will become more separable or less separable.


---

## Experiment 029 — Gradient Geometry

Date: 2026-09-27

### Question
Which individual training examples move the hidden representation toward or away from linear separability?

### Setup
Used the same He-initialized 2-2-1 XOR network.

Focused on the informative transition seeds from Experiment 027:
- Seed 0
- Seed 2
- Seed 4
- Seed 6
- Seed 8

For the epochs surrounding each transition, each XOR training example was examined separately.

For every example, recorded:
- hidden-space separation gap before the update
- hidden-space separation gap after the update
- change in separation gap
- hidden-layer gradients for each neuron
- output-layer gradients
- hidden representation before and after the update

Training examples were processed in the existing order:
[0,0], [0,1], [1,0], [1,1].

### Findings

The four training examples exerted different and sometimes opposing effects on hidden-space geometry.

#### Seed 0

The representation was separable immediately before the failure.

At epoch 27, the [1,0] example produced the critical update:

- gap before: 0.011260454
- gap after: 0.000000000
- delta gap: -0.011260454

Neuron 0 changed its [1,0] hidden activation from approximately 0.011262 to 0.000000.

Its gradient then became exactly zero, leaving the neuron unable to recover that input.

#### Seed 6

The representation started separable.

At epoch 1, the [0,1] example produced:

- gap before: 0.072356197
- gap after: 0.000000000
- delta gap: -0.072356197

Neuron 1 changed its [0,1] activation from approximately 0.073127 to 0.000000.

Its gradient became exactly zero immediately afterward.

#### Seed 2

Near epoch 131, the [0,1] example created the first useful separation:

- gap before: 0
- gap after: 0.009122485

The following [1,0] and [1,1] examples produced competing changes, including negative delta-gap updates, but the representation remained separable.

The important point is that useful separation could be created while both hidden neurons remained trainable.

#### Seed 4

Near the transition, the [1,0] example was the update that created separation.

At epoch 145, [1,0] produced:

- gap before: 0
- gap after: 0.001440382

The subsequent [1,1] update removed that small gap.

At epoch 146, the [1,0] update produced a larger gap:

- gap before: 0
- gap after: 0.005315076

The [1,1] update again reduced the gap, but this time it remained positive:

- gap after [1,1]: 0.002305559

The network therefore crossed into the separable regime because the positive contribution from [1,0] exceeded the opposing update from [1,1].

#### Seed 8

Near epoch 79, the [1,0] example created a temporary separable representation:

- gap after [1,0]: 0.004898453

The following [1,1] update destroyed that gap:

- gap after [1,1]: 0

At epoch 80, the [1,0] update created a much larger gap:

- gap after [1,0]: 0.020289951

The following [1,1] update reduced it to:

- gap after [1,1]: 0.004042234

The representation therefore remained separable.

### Conclusion

Individual training examples can push the hidden representation in opposing geometric directions.

In the failed transitions for seeds 0 and 6, a single update moved a critical hidden activation across the ReLU boundary and permanently removed its gradient.

In successful transitions, particular examples created positive separation, while other examples sometimes reduced it without destroying it.

This suggests that the fixed training-example order may matter because the network parameters change after every example.

The next experiment should test whether changing or shuffling the presentation order changes the probability of reaching a separable hidden representation.


---

## Experiment 030 — Training Order Sensitivity

Date: 2026-09-27

### Question
Does the order in which XOR training examples are presented materially affect the seed-dependent training outcome?

### Setup
Used the same He-initialized 2-2-1 XOR network.

Compared four deterministic presentation orders while keeping architecture, initialization, learning rate, and number of epochs unchanged:

- fixed: [0,1,2,3]
- reverse: [3,2,1,0]
- odd-even: [1,3,0,2]
- even-odd: [0,2,1,3]

Ten seeds were tested for each order.

A successful run was defined as final loss below 1e-6 with a nonzero hidden-space separation gap.

### Findings

Every tested order produced 3 successful runs out of 10.

Final aggregate results:

- fixed: 3/10 successful, mean loss 0.216863172405
- reverse: 3/10 successful, mean loss 0.216864670238
- odd-even: 3/10 successful, mean loss 0.216890863108
- even-odd: 3/10 successful, mean loss 0.216861674573

The reverse order changed the identity of one successful seed. Seed 1 failed under the fixed order but succeeded under the reverse order.

Seeds 4 and 8 succeeded under every tested order, while most other seeds remained failures.

The final mean separation gaps were also very similar across orders.

### Conclusion

Presentation order can alter individual training trajectories, but the tested orders did not materially change the overall probability of successful training in this 10-seed experiment.

Therefore, the seed-dependent instability is unlikely to be explained primarily by the ordering of the four XOR samples.

The next question is whether the instability comes from applying parameter updates after every individual example at all.

Experiment 031 should compare online per-example updates against full-batch updates using the same initialization, learning rate, architecture, and training data.


---

## Experiment 031 — Online vs Full-Batch Updates

Date: 2026-09-27

### Question
Does applying gradients after every individual example versus once per full batch affect the seed-dependent XOR training outcome?

### Setup
Used the same He-initialized 2-2-1 XOR network.

Compared:

1. Online updates:
   - process one example
   - backpropagate
   - update parameters
   - repeat for all four examples

2. Full-batch updates:
   - compute gradients for all four examples at the same parameter state
   - average the gradients
   - perform one parameter update

The original batch implementation was first found to be invalid because `Neuron.backward()` overwrites stored gradients rather than accumulating them. The batch implementation was corrected to explicitly accumulate and average gradients before updating parameters.

Parameter-update counts were then equalized:

- online: 1000 epochs × 4 updates = 4000 parameter updates
- batch: 4000 epochs × 1 update = 4000 parameter updates

Ten He-initialized seeds were tested for each method.

### Findings

Online training:

- successful runs: 3/10
- mean final loss: 0.216863172405
- mean final separation gap: 0.157110377

Full-batch training:

- successful runs: 4/10
- mean final loss: 0.182302779168
- mean final separation gap: 0.242709185

Seeds 2, 4, and 8 succeeded under both methods.

Seed 1 failed under online training but succeeded under full-batch training.

Several failed seeds converged near the familiar loss plateau around 0.333333, while full-batch training produced a stronger hidden-space separation for several successful runs.

### Conclusion

Update granularity affects the optimization trajectory in this network.

With equalized parameter-update counts, full-batch training produced more successful runs and a larger mean final hidden-space separation gap than online training in this 10-seed experiment.

The result does not establish that full-batch training is generally superior. It establishes that the way Larry aggregates gradients is a meaningful experimental variable that can change whether a useful hidden representation emerges.

Next question: determine whether the difference comes from gradient averaging itself or from the different parameter trajectory produced by the two update rules.


---

## Experiment 032 — Batch Gradient Scaling

Date: 2026-09-27

### Question
Does batch gradient scaling affect the seed-dependent training outcome?

### Setup
Used the same He-initialized 2-2-1 XOR network.

Compared three full-batch update rules:

1. Average:
   sum the four example gradients and divide by 4.

2. Sum:
   sum the four example gradients without dividing by 4.

3. Sum-scaled:
   sum the four example gradients without dividing by 4, but divide the learning rate by 4.

All conditions used 4000 batch parameter updates across seeds 0–9.

The average and sum-scaled conditions therefore apply mathematically equivalent parameter updates.

### Findings

Average:
- successful runs: 4/10
- mean final loss: 0.182302779168
- mean final separation gap: 0.242709185

Sum-scaled:
- successful runs: 4/10
- mean final loss: 0.182302779168
- mean final separation gap: 0.242709185

The identical results confirm that the two implementations are equivalent.

Sum:
- successful runs: 5/10
- mean final loss: 0.166666666667
- mean final separation gap: 0.272730891

The unscaled sum condition caused seed 9 to reach essentially zero loss, while seed 9 remained at a higher-loss state under average and sum-scaled updates.

### Conclusion

The magnitude of the batch parameter update materially changes the training trajectory.

The average and sum-scaled controls produced identical results, confirming that the observed difference is caused by the effective update size rather than the implementation of gradient accumulation itself.

In this experiment, the larger unscaled batch step produced more successful runs than the averaged batch step.

Therefore, learning-rate scale is now a strong candidate explanation for some of the seed-dependent behavior observed earlier.

Next question: determine whether there is a reproducible relationship between learning rate and the probability of reaching a separable hidden representation.


---

## Experiment 033 — Learning Rate Stability Window

Date: 2026-09-27

### Question
How does learning-rate magnitude affect the formation of a usable hidden representation and successful XOR training?

### Setup
Used the corrected full-batch implementation from Experiment 031.

- He initialization
- 2-2-1 network
- XOR training data
- full-batch gradient averaging
- 4000 parameter updates
- seeds 0–9
- learning rates: 0.01, 0.025, 0.05, 0.10, 0.20

Two outcomes were tracked:

1. Hidden-space separability.
2. Near-zero final loss (< 1e-6) with nonzero separation.

### Findings

#### Learning rate 0.01

- separable runs: 2/10
- near-zero-loss runs: 0/10
- mean loss: 0.349609392756
- mean gap: 0.015550920

Seeds 2 and 4 developed nonzero separation but did not converge to near-zero loss.

#### Learning rate 0.025

- separable runs: 4/10
- near-zero-loss runs: 0/10
- mean loss: 0.222895230645
- mean gap: 0.160115760

Seeds 1, 2, 4, and 8 became separable, but their final losses remained above the near-zero threshold.

#### Learning rate 0.05

- separable runs: 5/10
- near-zero-loss runs: 4/10
- mean loss: 0.182302779168
- mean gap: 0.242709185

Seeds 1, 2, 4, 8, and 9 were separable. Seeds 1, 2, 4, and 8 reached near-zero loss.

#### Learning rate 0.10

- separable runs: 5/10
- near-zero-loss runs: 5/10
- mean loss: 0.166666666667
- mean gap: 0.272111421

Seeds 1, 2, 4, 8, and 9 reached near-zero loss and were separable.

#### Learning rate 0.20

- separable runs: 5/10
- near-zero-loss runs: 5/10
- mean loss: 0.166666666667
- mean gap: 0.272730891

The outcome was very similar to learning rate 0.10.

### Conclusion

Learning rate materially changes Larry's training trajectory.

At 0.01, some seeds formed separable representations but training did not reach near-zero loss.

Increasing the learning rate to 0.025 improved hidden-space separation and final loss, but still did not produce any near-zero-loss runs under the selected threshold.

At 0.05, five seeds became separable and four reached near-zero loss.

At 0.10 and 0.20, five seeds reached near-zero loss, with very similar aggregate behavior.

This suggests that, for the current full-batch setup, increasing the learning rate from 0.01 toward approximately 0.10 improves the ability to escape poor trajectories and form useful representations. The results between 0.10 and 0.20 are already very similar, so simply increasing the learning rate further may not explain the remaining seed dependence.

The next question should therefore examine what distinguishes the permanently failed seeds from the successful seeds after learning-rate effects have been accounted for.


## Experiment 034 — Early Trajectory Discriminator

### Question

Can the seeds that eventually solve XOR be distinguished from the permanently failed seeds early in training, after fixing the learning rate at a stable value?

### Setup

Used the corrected full-batch implementation from Experiment 031 and the learning rate selected from Experiment 033.

- He initialization
- 2-2-1 network
- XOR training data
- full-batch gradient averaging
- learning rate: 0.10
- 4000 parameter updates
- seeds 0–9
- checkpoints: epochs 0, 1, 2, 5, 10, 20, 50, 100, and 4000

At each checkpoint, measured:

- loss
- hidden-space separation gap
- hidden activation pattern
- mean hidden activation
- mean absolute hidden gradient
- zero-gradient fraction

Success remained defined as final loss < 1e-6 with nonzero hidden-space separation.

### Results

The final split was:

- successful seeds: 1, 2, 4, 8, 9
- failed seeds: 0, 3, 5, 6, 7
- success rate: 5/10

At initialization, the successful seeds were not simply the seeds with an immediately separable hidden representation. Seeds 1, 2, 4, 8, and 9 all had zero separation at epoch 0, while only seed 6 had a small positive gap among the eventual failures.

The clearest early discriminator was hidden-unit activity.

By epoch 50:

- successful seeds retained 6–7 active hidden-unit/example pairs
- failed seeds retained only 1–2 active pairs

By epoch 100:

- successful seeds retained 5–7 active hidden-unit/example pairs
- failed seeds retained only 1–2 active pairs

The failed trajectories therefore progressively lost hidden-unit activity while their separation gap remained zero. Seeds 0 and 6 also demonstrated explicit loss of an initially useful geometric state: seed 0's gap fell from 0.3448 at initialization to zero by epoch 100, while seed 6's initial gap of 0.0724 fell to zero by epoch 5.

The successful trajectories were different. Their hidden representations were not separable during the early checkpoints, but multiple hidden activations remained active while training continued. Their separation emerged later, reaching positive values only by the final checkpoint.

At epoch 4000, the successful seeds had:

- seed 1: gap 0.6441
- seed 2: gap 0.5722
- seed 4: gap 0.4756
- seed 8: gap 0.5293
- seed 9: gap 0.5000

All five reached near-zero loss.

### Conclusion

The remaining seed dependence is strongly associated with early hidden-unit activity rather than initial hidden-space separability alone.

Successful seeds preserved several active hidden responses through the early training trajectory, even while their hidden representations were still nonseparable. Failed seeds progressively reduced their active patterns to only 1–2 active hidden-unit/example pairs, after which their gradients weakened and the network converged to a loss of approximately 1/3 without developing a separable representation.

This suggests that preserving a sufficiently rich hidden representation early in training may be a prerequisite for later geometric separation and successful XOR learning.

The next experiment should isolate hidden-unit survival/activity as a causal variable rather than only measuring it as an outcome.

## Experiment 035 — Hidden Neuron Death Guard

### Question

Is complete hidden-neuron death itself responsible for the seed-dependent failure observed in Experiment 034?

### Setup

Compared the normal training process against an intervention that prevents a hidden ReLU neuron from becoming completely inactive.

- He initialization
- 2-2-1 network
- XOR training data
- full-batch gradient averaging
- learning rate: 0.10
- 4000 parameter updates
- seeds 0–9

The baseline used the standard training rule.

The guarded condition monitored the two hidden neurons after each update. If a hidden neuron was inactive on all four XOR examples, that neuron's weights and bias were restored to their values from immediately before the update.

Success remained defined as final loss < 1e-6 with nonzero hidden-space separation.

### Results

Baseline:

- successful seeds: 1, 2, 4, 8, 9
- success rate: 5/10

Guarded:

- successful seeds: 1, 2, 4, 8, 9
- success rate: 5/10

The intervention therefore did not increase the number of successful runs.

For the successful seeds, the guard was never triggered:

- seed 1: 0 interventions
- seed 2: 0 interventions
- seed 4: 0 interventions
- seed 8: 0 interventions
- seed 9: 0 interventions

For the failed seeds, the guard triggered repeatedly:

- seed 0: 3946 interventions
- seed 3: 3954 interventions
- seed 5: 4000 interventions
- seed 6: 3998 interventions
- seed 7: 4000 interventions

Seeds 5 and 7 became completely inactive immediately and remained so under the guard. Seeds 0, 3, and 6 were also prevented from permanently entering the exact all-dead state, but this did not produce successful learning.

The guarded results for seeds 0 and 6 retained small nonzero separation gaps, but their losses remained near 1/3 and therefore did not meet the success criterion.

### Conclusion

Preventing complete hidden-neuron death by itself did not improve XOR success.

The successful seeds never required the guard, while the failed seeds repeatedly attempted to enter a completely inactive hidden-neuron state. Blocking that state did not cause the failed trajectories to discover the useful hidden representation required for XOR.

This indicates that hidden-neuron death is likely associated with failed training but is not, by itself, the sole causal explanation for the remaining seed dependence.

The important distinction is between merely keeping a neuron numerically active and preserving a hidden representation that provides useful gradient information for separating the XOR classes.

The next experiment should therefore examine the quality and direction of the hidden gradients before neuron death, rather than treating neuron survival alone as the intervention target.

## Experiment 036 — Gradient Conflict and Support

### Question

Do opposing per-example hidden-layer gradients explain the seed-dependent failure observed in earlier experiments?

### Setup

Used the same training configuration as Experiments 033–035.

- He initialization
- 2-2-1 network
- XOR training data
- full-batch gradient averaging
- learning rate: 0.10
- 4000 parameter updates
- seeds 0–9
- checkpoints: epochs 0, 1, 2, 5, 10, 20, 50, 100, and 4000

For each checkpoint, the hidden-layer gradient was measured separately for each XOR training example.

Pairwise cosine similarity was used to measure gradient direction:

- positive cosine: aligned update directions
- negative cosine: opposing update directions

Because some ReLU examples produce zero hidden gradients, only nonzero gradient pairs were included in the cosine calculations.

The number of valid pairs was also recorded as a measure of gradient support.

### Results

The final split remained:

- successful seeds: 1, 2, 4, 8, 9
- failed seeds: 0, 3, 5, 6, 7
- success rate: 5/10

Gradient conflict alone did not explain success or failure.

Several successful seeds developed strongly negative mean cosine similarity:

- seed 1 reached approximately -0.26 by epochs 20–100
- seed 2 reached approximately -0.30 to -0.33
- seed 4 reached approximately -0.18 to -0.29
- seed 8 reached approximately -0.26 to -0.32
- seed 9 reached approximately -0.25 to -0.30

Therefore, successful learning can occur while individual examples push the hidden layer in substantially opposing directions.

The number of examples contributing usable hidden gradients was more strongly associated with the eventual outcome.

At epoch 5:

- all successful seeds had 6 valid gradient pairs
- seed 0 had 3
- seeds 3, 5, and 7 had 1
- seed 6 had 0

At epoch 20:

- all successful seeds had 6 valid gradient pairs
- seed 0 had 1
- seeds 3, 5, and 7 had 1
- seed 6 had 0

At epoch 50:

- all successful seeds had 6 valid gradient pairs
- seed 0 had 1
- seed 3 had 1
- seeds 5, 6, and 7 had 0

At epoch 100:

- all successful seeds had 6 valid gradient pairs
- seed 0 had 0
- seed 3 had 1
- seeds 5, 6, and 7 had 0

The successful trajectories therefore maintained broad hidden-gradient participation across the XOR dataset, while the failed trajectories rapidly became supported by only one or zero effective hidden-gradient examples.

### Conclusion

Per-example gradient conflict is not sufficient to explain Larry's seed dependence. Strongly opposing gradients occur in both successful and failed trajectories.

The more consistent distinction is gradient support.

Successful seeds preserve hidden gradients across many XOR examples during early training. Failed seeds rapidly lose that support, leaving the batch update determined by very few examples before eventually reaching a zero-gradient state.

This refines the earlier neuron-survival hypothesis. The important property may not be whether a hidden neuron remains numerically active, but whether the hidden layer continues receiving useful gradient information from a sufficiently broad portion of the training set.

The next experiment should test this activation/gradient-coverage hypothesis directly.

## Experiment 037 — Gradient Floor Rescue

### Question

Are failed seeds caused primarily by ReLU's zero derivative on negative hidden pre-activations?

### Setup

Compared the normal hidden ReLU against a controlled gradient-floor condition.

- He initialization
- 2-2-1 network
- XOR training data
- full-batch gradient averaging
- learning rate: 0.10
- 4000 parameter updates
- seeds 0–9

The baseline used the standard ReLU derivative:

- positive pre-activation: derivative 1
- nonpositive pre-activation: derivative 0

The gradient-floor condition kept the exact same ReLU forward function, but changed the hidden derivative for nonpositive pre-activations from 0 to 0.01.

This isolates the effect of removing the exact zero-gradient condition without changing the hidden activation values.

Success remained defined as final loss < 1e-6 with nonzero hidden-space separation.

### Results

Baseline:

- successful seeds: 1, 2, 4, 8, 9
- success rate: 5/10

Gradient floor:

- successful seeds: 1, 2, 4, 8, 9
- success rate: 5/10

The gradient floor changed the final hidden-space gap slightly for successful seeds but did not change which seeds succeeded.

Failed seeds remained failed:

- seed 0: loss 0.333335370, gap 0
- seed 3: loss 0.333335740, gap 0
- seed 5: loss 0.333359911, gap 0
- seed 6: loss 0.333345190, gap 0
- seed 7: loss 0.250019296, gap 0

Seed 7 showed the largest improvement in loss, decreasing from approximately 0.3333 to 0.2500, but it still failed to create a separable hidden representation.

All five successful seeds remained successful under the intervention.

### Conclusion

Eliminating exact zero hidden gradients is not sufficient to explain or prevent the seed-dependent failure.

The forward representation remained identical to ReLU, while only the negative-side hidden derivative was changed. Despite this intervention, none of the five failed seeds became successful.

This strengthens the conclusion from Experiments 035 and 036: the problem is not simply that gradients become exactly zero. Successful training appears to depend on whether the hidden representation develops in a useful direction and preserves enough structure across the training examples.

The improvement for seed 7 shows that a gradient floor can alter the trajectory, but the resulting representation still did not become separable.

The next experiment should investigate whether increasing hidden-layer capacity reduces the seed dependence by giving the optimizer more representational freedom.

## Experiment 038 — Hidden Capacity Sweep

### Question

Does increasing hidden-layer capacity reduce the seed dependence observed with the 2-neuron hidden layer?

### Setup

Varied only the number of hidden ReLU neurons.

- He initialization
- XOR training data
- one linear output neuron
- full-batch gradient averaging
- learning rate: 0.10
- 4000 parameter updates
- seeds 0–9
- hidden widths: 1, 2, 3, 4, and 6

Success was defined as final loss < 1e-6.

For width 2, hidden-space separation gap was also retained for comparison with earlier experiments.

### Results

The success rate increased substantially with hidden width:

| Hidden width | Successful runs | Mean final loss |
|---|---:|---:|
| 1 | 0/10 | 0.416666666667 |
| 2 | 5/10 | 0.166666666667 |
| 3 | 6/10 | 0.133333333333 |
| 4 | 7/10 | 0.083333333333 |
| 6 | 9/10 | 0.025000000000 |

Width 1 failed on every seed.

The original 2-neuron configuration retained the 5/10 success rate observed in Experiments 033–037.

Increasing the hidden width to 3 changed the successful seeds to:

- 1, 2, 4, 6, 8, 9

Width 4 succeeded on:

- 1, 2, 4, 6, 7, 8, 9

Width 6 succeeded on:

- 0, 1, 2, 3, 4, 6, 7, 8, 9

Only seed 5 failed at width 6, with final loss 0.25.

The successful width-2 runs continued to produce the same hidden-space gaps observed previously:

- seed 1: 0.644062744
- seed 2: 0.572161784
- seed 4: 0.475612787
- seed 8: 0.529266449
- seed 9: 0.500010441

### Conclusion

Hidden-layer capacity strongly affects the reliability of XOR learning under the current training setup.

A single hidden ReLU neuron could not solve XOR in any of the tested seeds. Two neurons were sufficient to solve XOR, but only 5/10 seeds converged successfully.

Increasing capacity progressively reduced seed dependence:

- width 3: 6/10
- width 4: 7/10
- width 6: 9/10

This indicates that the remaining failures are not explained solely by ReLU death or zero gradients. Additional hidden units provide more representational and optimization freedom, allowing more initializations to reach a useful solution.

However, increasing width also increases the number of trainable parameters. Therefore, this experiment establishes a strong association between capacity and reliability, but does not yet separate increased representational capacity from the optimization effects of having more parameters.

The next experiment should investigate whether the width improvement comes from additional representational degrees of freedom or simply from having more independent hidden units available during optimization.

## Experiment 039 — Effective Capacity by Pruning

### Question

Do successful width-6 networks actually require all six hidden neurons in their final learned representation?

### Setup

Used the width-6 configuration from Experiment 038.

- He initialization
- 6 hidden ReLU neurons
- one linear output neuron
- XOR training data
- full-batch gradient averaging
- learning rate: 0.10
- 4000 parameter updates
- seeds 0–9

After training, each successful network was tested with every possible subset of its six hidden neurons.

Pruning was performed by setting the output weight connected to a hidden neuron to zero while leaving the trained hidden-layer parameters unchanged.

For each seed, the smallest subset producing loss < 1e-6 was recorded.

This is a post-training compressibility test. It does not measure whether the narrower network could have reached the same solution if trained from scratch.

### Results

The width-6 network again succeeded on:

- 9/10 seeds

Seed 5 was the only failure.

Among the nine successful runs, the minimum number of contributing hidden neurons required after training was:

- seed 0: 3
- seed 1: 3
- seed 2: 3
- seed 3: 3
- seed 4: 5
- seed 6: 4
- seed 7: 4
- seed 8: 5
- seed 9: 5

Distribution:

- 3 neurons: 4 runs
- 4 neurons: 2 runs
- 5 neurons: 3 runs
- 6 neurons: 0 runs

Mean minimum contributing neurons: 3.89.

Therefore, none of the successful width-6 solutions required all six hidden neurons after training.

### Conclusion

The capacity improvement observed in Experiment 038 does not mean that six hidden neurons are required to represent the final XOR solutions.

Successful width-6 models could be compressed after training to between 3 and 5 contributing hidden neurons while retaining near-zero loss.

This suggests that additional hidden capacity primarily provides extra degrees of freedom during optimization rather than being fully required by the final solution.

The distinction is important: width 6 may make it easier for gradient descent to discover a useful representation, even when the resulting solution can later be represented with fewer neurons.

The post-training pruning result does not establish that a width-3 or width-4 network trained from scratch would reliably reproduce those solutions. Experiment 038 already showed that narrower networks have lower success rates.

The next experiment should test whether the extra capacity helps specifically by allowing multiple candidate hidden features to develop before some become unnecessary.

## Experiment 040 — Late Capacity Injection

### Question

Does additional hidden capacity need to be present from initialization, or can it rescue a width-2 trajectory after training has already begun?

### Setup

Started each run with the same 2-neuron hidden ReLU network used in Experiments 033–039.

- He initialization
- 2 initial hidden ReLU neurons
- one linear output neuron
- XOR training data
- full-batch gradient averaging
- learning rate: 0.10
- 4000 total parameter-update steps
- seeds 0–9

The width-2 control trained for all 4000 updates without modification.

For the intervention conditions, four additional hidden ReLU neurons were added at one of five points:

- epoch 0
- epoch 5
- epoch 20
- epoch 50
- epoch 100

The original two hidden neurons were not modified.

The four new hidden neurons received fresh He-initialized weights and zero bias.

Their output-layer weights were initialized to zero so that adding the neurons did not immediately alter the network's prediction at the injection point. They then participated in normal training after injection.

Success was defined as final loss < 1e-6.

### Results

The width-2 control reproduced the previous result:

- successful seeds: 1, 2, 4, 8, 9
- success rate: 5/10
- mean final loss: 0.166666666667

Every capacity-injection condition succeeded on all ten seeds:

| Capacity injection | Successful runs | Mean final loss |
|---|---:|---:|
| none, width-2 control | 5/10 | 0.166666666667 |
| epoch 0 | 10/10 | 0.000000000000 |
| epoch 5 | 10/10 | 0.000000000000 |
| epoch 20 | 10/10 | 0.000000000000 |
| epoch 50 | 10/10 | 0.000000000000 |
| epoch 100 | 10/10 | 0.000000000000 |

Most importantly, the five seeds that consistently failed with width 2 in earlier experiments were all rescued by the late-capacity intervention.

This includes seeds whose width-2 trajectories had already developed poor or nearly inactive hidden representations before the additional neurons were introduced.

### Conclusion

Additional hidden capacity does not need to be present from the beginning for the current XOR task.

Even after 100 full-batch updates of the width-2 network, adding four new hidden neurons caused every tested seed to reach zero loss.

This shows that the poor width-2 trajectories are reversible rather than permanently trapped states. The failure is therefore not simply the consequence of an irreversible optimization collapse.

The result also strengthens the capacity hypothesis from Experiment 038. Extra neurons appear to provide the optimization process with additional directions in parameter space that can recover from trajectories that fail with only two hidden neurons.

The experiment does not yet establish how many additional neurons are necessary, nor whether the rescue is caused by the added representational capacity itself or by having multiple newly initialized hidden features available.

The next experiment should determine the minimum amount of additional capacity required to rescue the failed width-2 trajectories.

## Experiment 041 — Minimum Rescue Capacity

### Question

How many additional hidden neurons are required to rescue the width-2 trajectories that fail under the original configuration?

### Setup

Started every run with the same width-2 hidden ReLU network used in Experiments 033–040.

- He initialization
- 2 initial hidden ReLU neurons
- one linear output neuron
- XOR training data
- full-batch gradient averaging
- learning rate: 0.10
- 4000 total parameter updates
- seeds 0–9
- capacity injection point: epoch 100
- additional neurons tested: 0, 1, 2, 3, and 4

At epoch 100, new hidden neurons were added without modifying the original two hidden neurons.

New neurons received fresh He initialization and zero bias.

Their output-layer weights were initialized to zero, so the network's predictions were unchanged immediately at the injection point.

Success was defined as final loss < 1e-6.

### Results

The width-2 control reproduced the previous result:

- successful seeds: 1, 2, 4, 8, 9
- success rate: 5/10
- mean final loss: 0.166666666667

Adding one hidden neuron at epoch 100 increased success to:

- successful seeds: 1, 2, 3, 4, 6, 8, 9
- success rate: 7/10
- mean final loss: 0.091666666667

Adding two hidden neurons increased success to:

- successful seeds: 0, 1, 2, 3, 4, 5, 6, 8, 9
- success rate: 9/10
- mean final loss: 0.025000000000

Adding three hidden neurons produced:

- successful seeds: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
- success rate: 10/10
- mean final loss: 0.000000000000

Adding four hidden neurons also produced:

- successful seeds: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
- success rate: 10/10
- mean final loss: 0.000000000000

The previously difficult seed 7 remained the only failure with two added neurons, reaching loss 0.25. Adding a third neuron rescued it.

### Results Summary

| Additional neurons at epoch 100 | Successful runs | Mean final loss |
|---|---:|---:|
| 0 | 5/10 | 0.166666666667 |
| 1 | 7/10 | 0.091666666667 |
| 2 | 9/10 | 0.025000000000 |
| 3 | 10/10 | 0.000000000000 |
| 4 | 10/10 | 0.000000000000 |

### Conclusion

The number of additional hidden neurons strongly affects the ability to rescue a failing width-2 trajectory.

At the fixed epoch-100 injection point:

- one additional neuron rescued two additional seeds
- two additional neurons rescued four additional seeds
- three additional neurons rescued all remaining failures
- a fourth additional neuron provided no further increase in success rate

This shows that the failed width-2 trajectories retain recoverable information even after 100 updates. The optimization does not require restarting from a new initialization; additional hidden capacity can supply enough new degrees of freedom for the remaining seeds to reach zero loss.

The result also provides a concrete capacity threshold for this specific experiment: adding three hidden neurons at epoch 100 was sufficient to achieve 10/10 success across seeds 0–9.

This threshold should not be interpreted as a universal requirement. It depends on the XOR task, network architecture, learning rate, initialization procedure, injection time, and deterministic initialization of the newly added neurons.

The next experiment should determine whether the rescue threshold depends on when the capacity is added, and whether the same small amount of added capacity can rescue trajectories much later in training.

## Experiment 042 — Capacity Injection Timing

### Question

Does the effectiveness of added hidden capacity depend strongly on when the capacity is introduced?

### Setup

Started every run with the same width-2 hidden ReLU network used in Experiments 040–041.

- He initialization
- 2 initial hidden ReLU neurons
- one linear output neuron
- XOR training data
- full-batch gradient averaging
- learning rate: 0.10
- 4000 total parameter updates
- seeds 0–9
- injection epochs: 0, 25, 50, 100, 250, 500, 1000, 2000, 3000
- additional neurons tested: 1 and 2

At the selected injection epoch, new hidden neurons were added with fresh He initialization and zero bias.

Their output-layer weights were initialized to zero so that the intervention did not immediately change the network prediction at the injection point.

The width-2 network with no injection served as the control.

Success was defined as final loss < 1e-6.

### Results

The width-2 control reproduced the previous result:

- successful seeds: 1, 2, 4, 8, 9
- success rate: 5/10
- mean final loss: 0.166666666667

Adding one hidden neuron produced:

| Injection epoch | Successful runs | Mean final loss |
|---|---:|---:|
| 0 | 7/10 | 0.091666666667 |
| 25 | 7/10 | 0.091666666667 |
| 50 | 7/10 | 0.091666666667 |
| 100 | 7/10 | 0.091666666667 |
| 250 | 6/10 | 0.116666666667 |
| 500 | 6/10 | 0.116666666667 |
| 1000 | 6/10 | 0.116666666667 |
| 2000 | 6/10 | 0.116666666667 |
| 3000 | 6/10 | 0.116666667002 |

Adding two hidden neurons produced:

| Injection epoch | Successful runs | Mean final loss |
|---|---:|---:|
| 0 | 9/10 | 0.025000000000 |
| 25 | 9/10 | 0.025000000000 |
| 50 | 9/10 | 0.025000000000 |
| 100 | 9/10 | 0.025000000000 |
| 250 | 9/10 | 0.025000000000 |
| 500 | 9/10 | 0.025000000000 |
| 1000 | 9/10 | 0.025000000000 |
| 2000 | 9/10 | 0.025000000000 |
| 3000 | 9/10 | 0.025000010022 |

The identity of the rescued seeds was also highly stable.

With one added neuron at epochs 0–100, successful seeds were:

- 1, 2, 3, 4, 6, 8, 9

At epochs 250 and later, seed 3 was no longer rescued, reducing success to 6/10.

With two added neurons, successful seeds were consistently:

- 0, 1, 2, 3, 4, 5, 6, 8, 9

Seed 7 remained the only failure for every two-neuron injection timing.

At epoch 3000, seed 3 reached loss 0.000000100217 with two added neurons, which still satisfied the selected success threshold.

### Conclusion

Capacity injection timing has a smaller effect than the amount of added capacity.

Two additional hidden neurons rescued 9/10 seeds at every tested injection point, including very late injections at epochs 1000, 2000, and 3000. This shows that useful new capacity can remain effective even after most of the original width-2 training trajectory has already unfolded.

One additional neuron was effective through epoch 100, producing 7/10 success, but its effectiveness decreased after epoch 250 to 6/10. This indicates that a single new degree of freedom can become insufficient once the original trajectory has progressed further.

Seed 7 remained resistant to two-neuron rescue at every tested timing and continued to require the three-neuron intervention identified in Experiment 041.

These results reinforce the distinction between capacity and timing. Additional hidden units provide recoverable optimization freedom even late in training, but the amount of added capacity required can depend on the state of the trajectory.

The next experiment should investigate why seed 7 specifically requires more added capacity than the other failed seeds.

## Experiment 043 — Rescue Initialization Sensitivity

### Question

When two new hidden neurons are added to rescue a failed width-2 trajectory, is the outcome determined by the amount of capacity alone, or does the initialization of those new neurons matter?

### Setup

Used the width-2 network and late-capacity procedure from Experiment 040.

- He initialization for the original width-2 network
- 2 initial hidden ReLU neurons
- one linear output neuron
- XOR training data
- full-batch gradient averaging
- learning rate: 0.10
- 4000 total parameter updates
- capacity injection at epoch 100
- 2 additional hidden ReLU neurons
- new output weights initialized to zero
- seeds 0–9

Only the deterministic initialization of the two newly added neurons was varied.

The original network initialization and entire pre-injection trajectory remained unchanged.

Initialization offsets tested:

0, 1, 2, 3, 4, 5, 10, and 100.

Success was defined as final loss < 1e-6.

### Results

The original width-2 control is 5/10 successful.

With two neurons added at epoch 100, the success rate depended on the initialization offset:

| Initialization offset | Successful runs | Mean final loss |
|---|---:|---:|
| 0 | 9/10 | 0.025000000000 |
| 1 | 9/10 | 0.025000000000 |
| 2 | 10/10 | 0.000000000000 |
| 3 | 9/10 | 0.025000000000 |
| 4 | 9/10 | 0.025000000000 |
| 5 | 10/10 | 0.000000000000 |
| 10 | 8/10 | 0.050000000000 |
| 100 | 9/10 | 0.025000000000 |

Seed 7, which was the only seed not rescued by two added neurons across all injection timings in Experiment 042, was rescued under five of the eight initialization offsets tested here:

- offset 1
- offset 2
- offset 4
- offset 5
- offset 100

It failed under:

- offset 0
- offset 3
- offset 10

Seed 3 also changed outcome depending on initialization, succeeding under most offsets but failing under offsets 1, 4, and 10.

The best tested offsets, 2 and 5, produced 10/10 success.

### Conclusion

Two additional hidden neurons are sufficient to rescue seed 7, but the result depends on how those new neurons are initialized.

This rules out the interpretation that seed 7 fundamentally requires three additional neurons. Experiment 041 showed that three neurons guarantee rescue under the tested initialization, while Experiment 043 shows that two neurons can also rescue seed 7 under suitable initializations.

The results demonstrate that newly injected capacity creates additional optimization opportunities, but those opportunities depend on the starting location of the new neurons in parameter space.

This also helps explain why the width-6 experiment had a much higher success rate: more independently initialized hidden neurons provide more chances for useful features to emerge.

The next experiment should examine the initial activation coverage of the newly added neurons and determine whether successful rescue can be predicted from how the new neurons respond to the four XOR examples immediately after injection.

## Experiment 044 — Rescue Activation Coverage

### Question

Can the success of a two-neuron rescue be predicted from how broadly the newly added neurons activate across the XOR training examples immediately after injection?

### Setup

Used the rescue configuration from Experiment 043.

- He initialization for the original width-2 network
- 2 initial hidden ReLU neurons
- one linear output neuron
- XOR training data
- full-batch gradient averaging
- learning rate: 0.10
- 4000 total parameter updates
- capacity injection at epoch 100
- 2 additional hidden ReLU neurons
- new output weights initialized to zero
- seeds 0–9
- 8 deterministic initialization offsets for the newly added neurons

Immediately after injection, the activation pattern of each new neuron was recorded over:

- [0,0]
- [0,1]
- [1,0]
- [1,1]

The number of XOR examples activated by either new neuron was then counted as activation coverage.

Success was defined as final loss < 1e-6.

### Results

Across the 10 seeds and 8 initialization offsets:

- successful rescue cases: 73/80
- failed rescue cases: 7/80

Activation coverage among successful and failed cases was:

| Examples covered by the two new neurons | Successful | Failed |
|---|---:|---:|
| 0/4 | 1 | 0 |
| 1/4 | 3 | 1 |
| 2/4 | 29 | 5 |
| 3/4 | 40 | 1 |

Coverage therefore showed an association with rescue, particularly because most successful cases covered 3 of the 4 examples.

However, coverage was not sufficient to predict success.

Examples of this include:

- several successful cases with only 2/4 coverage
- three successful cases with 1/4 coverage
- one successful case with 0/4 coverage
- one failed case with 3/4 coverage

The 0/4 successful case occurred for a seed that was already capable of solving the task from the original width-2 trajectory, so the newly added neurons were not required for that run.

The failed cases also showed that simply covering more examples does not guarantee a successful rescue.

### Conclusion

Initial activation coverage of the newly added neurons is related to rescue reliability but is not by itself the determining factor.

The strongest concentration of successful cases occurred at 3/4 coverage, but 2/4 coverage was also frequently successful. Because the [0,0] input produces zero pre-activation for newly added neurons initialized with zero bias, the meaningful variation is primarily which of the other XOR examples the new neurons activate on and how those patterns combine.

This indicates that the structure of the activation pattern may matter more than raw coverage count.

The next experiment should therefore compare the exact activation patterns of successful and failed rescue cases and determine whether particular feature patterns are consistently associated with successful recovery.

## Experiment 045 — Rescue Activation Pattern Analysis

### Question

Does the exact activation pattern of the two newly added hidden neurons predict whether a failed width-2 trajectory will be rescued?

### Setup

Used the rescue configuration from Experiments 043–044.

- He initialization for the original width-2 network
- 2 initial hidden ReLU neurons
- one linear output neuron
- XOR training data
- full-batch gradient averaging
- learning rate: 0.10
- 4000 total parameter updates
- capacity injection at epoch 100
- 2 additional hidden ReLU neurons
- new output weights initialized to zero
- seeds 0–9
- initialization offsets: 0, 1, 2, 3, 4, 5, 10, 100

Immediately after injection, each newly added neuron was represented by a four-bit activation pattern over:

- [0,0]
- [0,1]
- [1,0]
- [1,1]

Neuron order was canonicalized, so swapping the two new neurons produced the same pattern representation.

Success was defined as final loss < 1e-6.

### Results

Across all 80 seed/initialization combinations:

- successful rescue cases: 73/80
- failed rescue cases: 7/80

The most common successful patterns included:

- 0010|0111: 15/15 successful
- 0100|0111: 10/10 successful
- 0101|0111: 10/10 successful
- 0010|0100: 10/10 successful

Other patterns were less reliable.

For example:

- 0000|0011: 6/7 successful
- 0000|0100: 3/4 successful
- 0000|0101: 11/14 successful
- 0000|0111: 2/3 successful
- 0100|0101: 2/3 successful

There was also a successful case with:

- 0000|0000: 1/1 successful

The seven failures occurred under several different patterns:

- 0000|0101: three failures
- 0000|0011: one failure
- 0000|0100: one failure
- 0100|0101: one failure
- 0000|0111: one failure

### Conclusion

The exact initial activation pattern of the two new neurons is associated with rescue reliability, but it is not sufficient to determine the outcome.

Several patterns were perfectly successful in the tested sample, while other patterns produced both successful and failed runs.

The successful 0000|0000 case is especially important: newly added neurons did not need to activate any of the four examples immediately after injection in order for the network to eventually reach zero loss.

Therefore, initial activation coverage and exact activation pattern are not complete explanations of rescue.

The results indicate that the newly added neurons can acquire useful behavior during subsequent optimization even when their initial activation is limited or absent.

The next experiment should examine the first few updates after injection and measure how quickly the newly added neurons acquire nonzero output weights, activation coverage, and useful gradients.

## Experiment 046 — Rescue Early Dynamics

### Question

What happens during the first few updates after capacity is injected, and can early new-neuron dynamics distinguish successful rescues from failed rescues?

### Setup

Used the configuration from Experiments 043–045.

- He initialization for the original width-2 network
- 2 initial hidden ReLU neurons
- one linear output neuron
- XOR training data
- full-batch gradient averaging
- learning rate: 0.10
- 4000 total parameter updates
- capacity injection at epoch 100
- 2 additional hidden ReLU neurons
- zero initial output weights for the new neurons
- seeds 0–9
- initialization offsets: 0, 1, 2, 3, 4, 5, 10, 100

After injection, measurements were recorded at relative steps:

- 0
- 1
- 2
- 3
- 5
- 10
- 20

For the newly added neurons, the experiment measured:

- active examples per neuron
- mean absolute output weight
- mean absolute hidden gradient
- zero-gradient fraction

Success was defined as final loss < 1e-6.

### Results

Across the 80 seed/initialization combinations:

- successful rescues: 73
- failed rescues: 7

At the injection moment, new output weights were zero and consequently the newly added hidden neurons had zero hidden-layer gradient.

The successful and failed groups differed in initial activation support:

| Relative step | Successful mean active examples per new neuron | Failed mean active examples per new neuron |
|---|---:|---:|
| 0 | 1.575 | 1.071 |
| 1 | 1.575 | 1.071 |
| 2 | 2.226 | 1.571 |
| 3 | 2.055 | 1.214 |
| 5 | 2.048 | 1.214 |
| 10 | 2.089 | 1.214 |
| 20 | 2.110 | 1.143 |

The successful group also developed larger new-neuron output weights and hidden gradients during the first 20 updates.

At relative step 20:

- successful mean absolute new-neuron output weight: 0.057198
- failed mean absolute new-neuron output weight: 0.044954
- successful mean absolute new-neuron gradient: 0.011009
- failed mean absolute new-neuron gradient: 0.009282

The zero-gradient fraction showed a stronger difference:

- successful group: approximately 0.279
- failed group: approximately 0.476

### Failed Rescue Cases

The seven failed cases showed limited early support for the newly added neurons.

Examples included:

- seed 7, offset 0: activity remained approximately 2,0 across the first 20 updates
- seed 3, offset 1: activity remained approximately 0,2
- seed 7, offset 3: activity remained approximately 0,2
- seed 3, offset 4: activity remained approximately 2,0
- seed 3, offset 10: activity increased to approximately 2,2 but still failed
- seed 7, offset 10: activity remained limited to approximately 0,1–0,2
- seed 5, offset 100: activity remained approximately 3,0

These cases show that simply activating both neurons is not sufficient. For example, seed 3 with offset 10 eventually had both new neurons active on multiple examples but still failed.

### Conclusion

Successful rescue trajectories generally developed broader and stronger new-neuron participation during the first 20 updates after injection.

Compared with failed rescues, successful cases showed:

- more active examples per new neuron
- larger output weights
- larger hidden gradients
- a smaller fraction of zero hidden gradients

However, these measurements do not establish a single deterministic threshold. Some successful cases began with very limited activation, while at least one failed case reached relatively broad activation.

The results therefore identify early new-neuron participation as a useful correlate of successful rescue, but not yet a sufficient causal explanation.

The next experiment should test whether giving newly added neurons nonzero output connections at injection changes the result by allowing them to receive hidden-layer gradients immediately, while keeping their hidden initialization fixed.

## Experiment 047 — Nonzero Rescue Output Weights

### Question

Does initializing the newly added neurons with zero output-layer weights create a gradient bottleneck that reduces rescue success?

### Setup

Used the rescue configuration from Experiments 043–046.

- He initialization for the original width-2 network
- 2 initial hidden ReLU neurons
- one linear output neuron
- XOR training data
- full-batch gradient averaging
- learning rate: 0.10
- 4000 total parameter updates
- capacity injection at epoch 100
- 2 additional hidden ReLU neurons
- seeds 0–9
- initialization offsets: 0, 1, 2, 3, 4, 5, 10, 100

Compared two initial output-weight conditions for the newly added neurons:

- 0.00
- 0.01

The hidden-neuron initialization, original network trajectory, and all other training conditions were held fixed.

Each condition therefore contained 80 seed/initialization combinations.

Success was defined as final loss < 1e-6.

### Results

With zero new output weights:

- successful cases: 73/80
- mean final loss: 0.021875000000
- mean loss at injection: 0.421455397945

With output weights initialized to 0.01:

- successful cases: 73/80
- mean final loss: 0.021875000000
- mean loss at injection: 0.419786258131

The success count was identical for every initialization offset.

The 0.01 condition produced a slightly lower mean loss immediately after injection because the newly added neurons contributed a small nonzero output immediately, but this did not change final success rate or mean final loss.

### Conclusion

Initializing the newly added neurons with zero output-layer weights is not sufficient to explain the observed rescue failures.

Changing their initial output weights from 0.00 to 0.01 produced exactly the same success rate:

- 73/80 with zero output weights
- 73/80 with 0.01 output weights

The result indicates that the basic zero-output initialization is not the dominant bottleneck in the tested configuration.

However, only a small positive value was tested. This experiment therefore does not rule out larger output-weight initialization effects.

The stronger conclusion is that simply giving the newly added neurons a small immediate contribution does not materially change rescue reliability.

The next experiment should test whether the amount and direction of the new output connections matter, using substantially larger positive and negative initial values while keeping the hidden initialization fixed.

## Experiment 048 — Signed Rescue Output Initialization

### Question

Does the magnitude and sign of the newly added neurons' initial output weights affect rescue success?

### Setup

Used the rescue configuration from Experiments 043–047.

- He initialization for the original width-2 network
- 2 initial hidden ReLU neurons
- one linear output neuron
- XOR training data
- full-batch gradient averaging
- learning rate: 0.10
- 4000 total parameter updates
- capacity injection at epoch 100
- 2 additional hidden ReLU neurons
- seeds 0–9
- initialization offsets: 0, 1, 2, 3, 4, 5, 10, 100

The initial output weights of the two newly added neurons were set to the same tested value:

- -0.50
- -0.10
- 0.00
- +0.10
- +0.50

Each condition contained 80 seed/initialization combinations.

Success was defined as final loss < 1e-6.

### Results

| Initial output weight | Successful cases | Mean final loss | Mean loss at injection |
|---|---:|---:|---:|
| -0.50 | 55/80 | 0.087500000000 | 1.090635673320 |
| -0.10 | 69/80 | 0.035416666667 | 0.463413290714 |
| 0.00 | 73/80 | 0.021875000000 | 0.421455397945 |
| +0.10 | 73/80 | 0.021875000000 | 0.425436586328 |
| +0.50 | 72/80 | 0.026041666667 | 0.900752151391 |

The zero-output condition reproduced the Experiment 047 result of 73/80.

The +0.10 condition also produced 73/80.

The +0.50 condition produced 72/80, a small decrease.

Negative initialization was more disruptive:

- -0.10 produced 69/80
- -0.50 produced 55/80

The -0.50 condition also produced a substantially larger mean loss immediately after injection.

### Conclusion

The initial output connection of newly added neurons can affect rescue reliability, but the effect depends strongly on magnitude and sign.

A small positive output weight (+0.10) produced the same 73/80 success rate as zero initialization.

A larger positive value (+0.50) produced only a small change, decreasing success from 73/80 to 72/80.

Negative output weights were substantially more disruptive. At -0.50, success fell to 55/80 and the mean loss at injection increased considerably. At -0.10, success fell to 69/80.

This demonstrates that large signed output initialization changes the rescue trajectory materially. However, the experiment does not isolate the hidden-gradient mechanism by itself, because changing the output weights also changes the network's output and therefore the error signal at the injection point.

The main result is that the previously observed rescue behavior is robust to zero versus small positive output initialization, but not to sufficiently large signed initialization.

The next experiment should control the magnitude of the initial output contribution more carefully and determine whether the effect is caused by the initial prediction shift or by the resulting hidden-gradient magnitude.

## Experiment 049 — Balanced Rescue Output Weights

### Question

Does the relationship between the two newly added output weights matter when their absolute magnitude is fixed at 0.10?

### Setup

Used the rescue configuration from Experiments 043–048.

- He initialization for the original width-2 network
- 2 initial hidden ReLU neurons
- one linear output neuron
- XOR training data
- full-batch gradient averaging
- learning rate: 0.10
- 4000 total parameter updates
- capacity injection at epoch 100
- 2 additional hidden ReLU neurons
- seeds 0–9
- initialization offsets: 0, 1, 2, 3, 4, 5, 10, 100

Compared four output-weight configurations for the two newly added neurons:

- (0.00, 0.00)
- (+0.10, +0.10)
- (+0.10, -0.10)
- (-0.10, -0.10)

Each condition contained 80 seed/initialization combinations.

Success was defined as final loss < 1e-6.

### Results

| New output weights | Successful cases | Mean final loss | Mean loss at injection |
|---|---:|---:|---:|
| (0.00, 0.00) | 73/80 | 0.021875000000 | 0.421455397945 |
| (+0.10, +0.10) | 73/80 | 0.021875000000 | 0.425436586328 |
| (+0.10, -0.10) | 73/80 | 0.021875000000 | 0.439315532054 |
| (-0.10, -0.10) | 69/80 | 0.035416666667 | 0.463413290714 |

Zero, both-positive, and balanced-signed output weights all produced exactly the same success rate:

- 73/80

The balanced signed condition therefore did not improve rescue reliability despite giving the two new neurons opposite output directions.

Both-negative initialization reduced success to:

- 69/80

It also produced a higher mean loss immediately after injection.

### Conclusion

At an absolute magnitude of 0.10 per new output connection, the relative sign of the two new connections does not materially affect rescue when one connection is positive and the other is negative.

The three conditions:

- (0.00, 0.00)
- (+0.10, +0.10)
- (+0.10, -0.10)

all produced 73/80 successful rescues.

The both-negative condition performed somewhat worse at 69/80.

This reinforces the result from Experiment 048 that sufficiently negative output initialization can disrupt the rescue trajectory, while small positive or balanced signed connections are largely equivalent to zero initialization in this setup.

The result also weakens the hypothesis that rescue depends simply on introducing a particular sign pattern into the output layer.

The next experiment should return to the hidden representation and ask whether the new neurons' learned contributions, rather than their initial output-weight signs, predict successful rescue.

## Experiment 050 — Learned Rescue Contributions

### Question

After two hidden neurons are injected into a failing width-2 trajectory, what distinguishes the neurons that become useful from those that do not?

### Setup

Used the rescue configuration from Experiments 043–049.

- He initialization for the original width-2 network
- 2 initial hidden ReLU neurons
- one linear output neuron
- XOR training data
- full-batch gradient averaging
- learning rate: 0.10
- 4000 total parameter updates
- capacity injection at epoch 100
- 2 additional hidden ReLU neurons
- zero initial output weights for the new neurons
- seeds 0–9
- initialization offsets: 0, 1, 2, 3, 4, 5, 10, 100

Measurements were taken at relative steps:

- 0
- 1
- 5
- 20
- 100
- 500
- 3900

For each new hidden neuron, measured:

- absolute output weight
- number of active XOR examples
- mean activation
- contribution magnitude to the output

The contribution of a neuron was calculated as its output weight multiplied by its hidden activation. Success was defined as final loss < 1e-6.

### Results

Across the 80 seed/initialization combinations:

- successful rescues: 73
- failed rescues: 7

The successful group developed substantially larger contributions from the newly added neurons.

At relative step 20:

- successful mean absolute output weight: 0.057198
- failed mean absolute output weight: 0.044954
- successful mean contribution norm: 0.117597
- failed mean contribution norm: 0.082066

At relative step 100:

- successful mean absolute output weight: 0.224344
- failed mean absolute output weight: 0.151504
- successful mean contribution norm: 0.436153
- failed mean contribution norm: 0.284186

At relative step 500:

- successful mean absolute output weight: 0.589309
- failed mean absolute output weight: 0.306011
- successful mean contribution norm: 1.395823
- failed mean contribution norm: 0.619010

At the final checkpoint:

- successful mean absolute output weight: 0.627039
- failed mean absolute output weight: 0.318465
- successful mean contribution norm: 1.492687
- failed mean contribution norm: 0.678961

The successful group therefore produced approximately twice the new-neuron contribution magnitude of the failed group by the final checkpoint.

Activation support also differed.

At the injection point:

- successful mean active examples per new neuron: 1.575
- failed mean active examples per new neuron: 1.071

At relative step 20:

- successful: 2.110
- failed: 1.143

At relative step 500:

- successful: 1.877
- failed: 0.857

By the final checkpoint:

- successful: 1.562
- failed: 0.857

### Failed Cases

The individual failed trajectories show a recurring pattern in which one of the two newly added neurons contributes nothing or very little.

Examples:

- seed 7, offset 0: one new neuron had output weight 0.032481 while the other remained at 0.0 and inactive
- seed 3, offset 1: one new neuron remained inactive with zero output weight
- seed 7, offset 3: one new neuron remained inactive with zero output weight
- seed 3, offset 4: one new neuron remained inactive with zero output weight
- seed 7, offset 10: one new neuron remained inactive
- seed 5, offset 100: one new neuron remained inactive while the other developed a negative output weight

Seed 3 with offset 10 is an important exception: both new neurons developed nonzero output weights and both were active on two examples, yet the run still failed.

### Conclusion

Successful rescue is strongly associated with the newly added neurons becoming substantial contributors to the output during training.

Compared with failed rescues, successful cases developed:

- larger output weights
- broader activation support
- larger hidden-to-output contribution magnitude

The divergence appears early and grows over time. By relative step 20, the successful group already has higher output-weight magnitude and contribution norm, and the difference becomes much larger by steps 100–500.

The failed cases frequently contain one new neuron that remains inactive or has zero output weight. However, this is not a complete explanation because at least one failed case developed nonzero contributions from both new neurons.

Therefore, the important property is likely not simple neuron survival or activation coverage alone. It is whether the newly added features become sufficiently useful to the output objective during optimization.

This provides a stronger characterization of the rescue mechanism but remains observational. The correlation between contribution growth and successful learning does not establish whether large contributions cause success or emerge because the trajectory is already moving toward a solution.

The next experiment should manipulate the contribution of newly added neurons directly and test whether forcing or limiting their output influence changes rescue probability.

## Experiment 051 — Rescue Contribution Cap

### Question

Does limiting the output contribution of newly added hidden neurons reduce their ability to rescue a failing width-2 trajectory?

### Setup

Used the rescue configuration from Experiments 043–050.

- He initialization for the original width-2 network
- 2 initial hidden ReLU neurons
- one linear output neuron
- XOR training data
- full-batch gradient averaging
- learning rate: 0.10
- 4000 total parameter updates
- capacity injection at epoch 100
- 2 additional hidden ReLU neurons
- zero initial output weights for the new neurons
- seeds 0–9
- initialization offsets: 0, 1, 2, 3, 4, 5, 10, 100

The experiment tested maximum absolute output-weight caps on the two newly added neurons:

- 0.02
- 0.05
- 0.10
- 0.20
- uncapped control

After every training update following injection, the new output weights were clamped to the selected maximum absolute magnitude.

Success was defined as final loss < 1e-6.

### Results

| Output-weight cap | Successful cases | Mean final loss |
|---|---:|---:|
| 0.02 | 40/80 | 0.157221639967 |
| 0.05 | 40/80 | 0.130584835274 |
| 0.10 | 38/80 | 0.087053968266 |
| 0.20 | 34/80 | 0.034134855135 |
| uncapped | 73/80 | 0.021875000000 |

The uncapped result reproduced the previous 73/80 rescue rate.

The strongest cap, 0.02, reduced rescue success to exactly 40/80.

The 0.05 cap also produced 40/80 success.

The 0.10 cap produced 38/80.

The 0.20 cap produced 34/80.

Thus, imposing a contribution limit substantially reduced the ability of the newly added neurons to rescue failed trajectories.

Interestingly, increasing the cap from 0.02 to 0.20 did not produce a monotonic increase in success rate in this experiment. The uncapped condition was substantially better than every capped condition.

### Conclusion

The ability of newly added hidden neurons to grow substantial output connections is important for successful rescue.

When their output weights were capped after every update, rescue success fell sharply compared with the uncapped condition:

- uncapped: 73/80
- cap 0.20: 34/80
- cap 0.10: 38/80
- cap 0.05: 40/80
- cap 0.02: 40/80

This provides stronger causal evidence than the observational result in Experiment 050. Limiting the new neurons' output influence changes the final success rate substantially.

The result does not imply that larger output weights are always better. The non-monotonic ordering among the capped conditions shows that the relationship is more complicated than a simple magnitude threshold.

A likely interpretation is that successful rescue requires the new neurons to develop sufficiently strong task-relevant contributions during optimization. The uncapped model allows those contributions to grow as needed, while the capped models constrain that adaptation.

The next experiment should identify whether the important quantity is absolute output-weight magnitude itself or the actual task-relevant contribution produced by the hidden activations and output weights together.

## Experiment 052 — Task Contribution Cap

### Question

Is the actual task contribution of newly added hidden neurons important for successful rescue, independent of output-weight magnitude alone?

### Setup

Used the rescue configuration from Experiments 043–051.

- He initialization for the original width-2 network
- 2 initial hidden ReLU neurons
- one linear output neuron
- XOR training data
- full-batch gradient averaging
- learning rate: 0.10
- 4000 total parameter updates
- capacity injection at epoch 100
- 2 additional hidden ReLU neurons
- zero initial output weights for the new neurons
- seeds 0–9
- initialization offsets: 0, 1, 2, 3, 4, 5, 10, 100

For each newly added neuron, its contribution across the four XOR examples was defined as:

output weight × hidden activation

The contribution norm was calculated across the four examples.

After each training update, the output weight of each new neuron was rescaled when necessary so that its contribution norm did not exceed the selected cap.

Tested contribution caps:

- 0.10
- 0.25
- 0.50
- 1.00
- uncapped control

Each condition contained 80 seed/initialization combinations.

Success was defined as final loss < 1e-6.

### Results

| Contribution cap | Successful cases | Mean final loss |
|---|---:|---:|
| 0.10 | 38/80 | 0.150876694746 |
| 0.25 | 35/80 | 0.116419062522 |
| 0.50 | 32/80 | 0.069342987948 |
| 1.00 | 50/80 | 0.029742007557 |
| uncapped | 73/80 | 0.021875000000 |

The uncapped condition reproduced the previous 73/80 rescue rate.

Every contribution cap substantially reduced the success rate relative to the uncapped condition.

The strongest restriction, 0.10, produced 38/80 successful cases.

The 0.25 and 0.50 conditions produced 35/80 and 32/80 respectively.

Allowing a contribution norm up to 1.00 improved success to 50/80 but remained substantially below the uncapped result.

The capped conditions therefore demonstrate that restricting the effective output contribution of the new neurons can prevent otherwise successful rescue trajectories.

### Conclusion

The actual contribution made by newly added hidden neurons is important to successful rescue.

Limiting contribution magnitude after each update substantially reduced the number of successful runs compared with the uncapped condition:

- uncapped: 73/80
- cap 1.00: 50/80
- cap 0.50: 32/80
- cap 0.25: 35/80
- cap 0.10: 38/80

The relationship was not monotonic across the capped values. The 0.50 condition had the lowest success rate among the tested caps, while 0.10 and 0.25 produced slightly higher success rates. This indicates that the rescue process is not governed by a simple minimum contribution threshold.

The central result is that constraining the new neurons' effective influence on the output can prevent successful learning, providing causal evidence that sufficient learned contribution is part of the rescue mechanism.

This complements Experiment 051, where limiting output-weight magnitude also reduced rescue success. The present experiment shows that the relevant quantity is not only the raw output weight; the interaction between output weight and hidden activation also matters.

However, because the intervention is implemented by rescaling output weights, the experiment does not completely isolate contribution magnitude from changes to the gradient dynamics caused by that rescaling.

The next experiment should examine whether deliberately encouraging larger contributions early after injection improves rescue, rather than only testing what happens when contribution is restricted.

## Experiment 053 — Rescue Contribution Boost

### Question

Does accelerating the growth of the newly added neurons' output connections immediately after injection improve rescue success?

### Setup

Used the rescue configuration from Experiments 043–052.

- He initialization for the original width-2 network
- 2 initial hidden ReLU neurons
- one linear output neuron
- XOR training data
- full-batch gradient averaging
- learning rate: 0.10
- 4000 total parameter updates
- capacity injection at epoch 100
- 2 additional hidden ReLU neurons
- zero initial output weights for the new neurons
- seeds 0–9
- initialization offsets: 0, 1, 2, 3, 4, 5, 10, 100

The normal output-weight update for the two new neurons was multiplied by a boost factor during the first 20 updates after injection.

Tested boost factors:

- 1x control
- 2x
- 4x
- 8x

The remaining training after the first 20 post-injection updates was unchanged.

Each condition contained 80 seed/initialization combinations.

Success was defined as final loss < 1e-6.

### Results

All four conditions produced exactly the same outcome:

| Boost factor | Successful cases | Mean final loss |
|---|---:|---:|
| 1x | 73/80 | 0.021875000000 |
| 2x | 73/80 | 0.021875000000 |
| 4x | 73/80 | 0.021875000000 |
| 8x | 73/80 | 0.021875000000 |

Every initialization offset produced the same number of successful seeds under every boost factor.

The 8x condition therefore produced no measurable improvement in the selected success criterion.

### Conclusion

Accelerating the output-weight updates of the newly added neurons during the first 20 updates did not change rescue success in this experiment.

The success rate remained 73/80 for every tested boost factor, including an 8x increase.

This is another important narrowing result. Experiment 051 showed that restricting contribution can reduce rescue reliability, while Experiment 053 shows that simply pushing output-weight growth faster does not improve the already-high rescue rate.

Together, these results suggest that successful rescue requires sufficient freedom for useful contributions to develop, but additional acceleration alone does not guarantee a better trajectory.

The experiment also shows that the remaining seven failures are not explained simply by output-weight growth being too slow during the first 20 updates.

The next experiment should therefore investigate the hidden neurons' parameter updates themselves, rather than only their output connections.

## Experiment 054 — Hidden Feature Update Scaling

### Question

Does changing the update speed of newly added hidden neurons during the first 20 updates after injection affect rescue success?

### Setup

Used the rescue configuration from Experiments 043–053.

- He initialization for the original width-2 network
- 2 initial hidden ReLU neurons
- one linear output neuron
- XOR training data
- full-batch gradient averaging
- learning rate: 0.10
- 4000 total parameter updates
- capacity injection at epoch 100
- 2 additional hidden ReLU neurons
- zero initial output weights for the new neurons
- seeds 0–9
- initialization offsets: 0, 1, 2, 3, 4, 5, 10, 100

For the first 20 updates after injection, only the hidden-parameter updates of the two newly added neurons were scaled.

Tested scaling factors:

- 0.0x
- 1.0x
- 2.0x
- 4.0x

The output-layer updates remained normal.

Each condition contained 80 seed/initialization combinations.

Success was defined as final loss < 1e-6.

### Results

All four conditions produced exactly the same outcome:

| Hidden update factor | Successful cases | Mean final loss |
|---|---:|---:|
| 0.0x | 73/80 | 0.021875000000 |
| 1.0x | 73/80 | 0.021875000000 |
| 2.0x | 73/80 | 0.021875000000 |
| 4.0x | 73/80 | 0.021875000000 |

The 1.0x condition is the normal-training control and reproduced the previous 73/80 result.

Even freezing the newly added hidden neurons for the first 20 post-injection updates produced the same 73/80 success rate.

Increasing their hidden-parameter update magnitude to 2x or 4x also produced no change in success rate or mean final loss.

### Conclusion

The speed of hidden-feature learning during the first 20 updates after capacity injection did not affect the final rescue outcome in this experiment.

The most restrictive condition, 0.0x, froze the new hidden parameters during those first 20 updates, yet achieved the same success rate as the normal 1.0x condition.

Likewise, accelerating the hidden updates to 2x or 4x produced no measurable change.

This is a useful null result when combined with Experiments 051–053:

- limiting eventual output contribution can reduce rescue success
- accelerating output-weight growth does not improve rescue
- limiting or accelerating early hidden-feature updates does not change rescue

These results suggest that the decisive factor is not simply how quickly the new parameters move immediately after injection. The more important factor may be whether the new neurons eventually discover a useful task-aligned configuration over the subsequent training trajectory.

The next experiment should therefore examine the direction of the learned hidden-feature changes rather than their update magnitude alone.

## Experiment 055 — Hidden Feature Direction

### Question

Does the direction and magnitude of hidden-feature movement after capacity injection distinguish successful rescues from failed rescues?

### Setup

Used the rescue configuration from Experiments 043–054.

- He initialization for the original width-2 network
- 2 initial hidden ReLU neurons
- one linear output neuron
- XOR training data
- full-batch gradient averaging
- learning rate: 0.10
- 4000 total parameter updates
- capacity injection at epoch 100
- 2 additional hidden ReLU neurons
- zero initial output weights for the new neurons
- seeds 0–9
- initialization offsets: 0, 1, 2, 3, 4, 5, 10, 100

For each newly added hidden neuron, tracked its hidden weight vector relative to its value at injection.

Measurements were taken at relative steps:

- 0
- 1
- 5
- 20
- 100
- 500
- 3900

Measured:

- cosine similarity between initial and current hidden-weight vectors
- Euclidean weight displacement
- absolute bias change

Success was defined as final loss < 1e-6.

### Results

Across the 80 seed/initialization combinations:

- successful rescues: 73
- failed rescues: 7

At the first few updates, the successful and failed groups were nearly identical.

By relative step 20:

- successful mean displacement: 0.005928
- failed mean displacement: 0.004911
- successful mean bias change: 0.001602
- failed mean bias change: 0.000601

By relative step 100:

- successful mean displacement: 0.090818
- failed mean displacement: 0.061836
- successful mean cosine similarity: 0.994261
- failed mean cosine similarity: 0.996702

By relative step 500:

- successful mean displacement: 0.366420
- failed mean displacement: 0.181424
- successful mean cosine similarity: 0.952493
- failed mean cosine similarity: 0.971431

At the final checkpoint:

- successful mean displacement: 0.402689
- failed mean displacement: 0.189233
- successful mean cosine similarity: 0.939436
- failed mean cosine similarity: 0.971040
- successful mean bias change: 0.063797
- failed mean bias change: 0.022425

Thus, successful rescue trajectories moved the newly added hidden features substantially farther from their injected initialization than failed trajectories.

### Failed Cases

The individual failures show that large movement alone is not sufficient.

For example:

- seed 3, offset 4 reached displacement 0.396630 at step 500 and cosine 0.882785, yet still failed
- seed 5, offset 100 reached bias change 0.146822, yet still failed

Other failed cases showed much smaller movement, such as:

- seed 3, offset 1: displacement 0.057178 at step 500
- seed 7, offset 0: displacement 0.177999
- seed 7, offset 3: displacement 0.189473

### Conclusion

Successful rescue trajectories generally required substantially more movement of the newly added hidden features than failed trajectories.

The divergence became clear during the first few hundred post-injection updates. By step 500, successful cases had approximately twice the mean hidden-weight displacement of failed cases.

Cosine similarity also decreased more strongly in successful trajectories, indicating that their hidden features moved farther in parameter space from their initial directions.

However, direction and displacement alone were not sufficient to predict success. At least one failed case moved farther and rotated more strongly than the successful-group average.

Therefore, the important property is not simply how much a newly added hidden feature changes. The feature must change in a way that becomes useful to the XOR objective.

This strengthens the hypothesis that rescue depends on task-relevant hidden-feature formation rather than raw parameter-update magnitude.

The next experiment should measure whether the learned hidden-weight vectors become aligned with specific XOR decision directions or input differences associated with the target labels.

## Experiment 056 — Task-Aligned Hidden Features

### Question

Do newly added hidden neurons become useful for rescue by developing activation patterns that distinguish the XOR positive and negative classes?

### Setup

Used the rescue configuration from Experiments 043–055.

- He initialization for the original width-2 network
- 2 initial hidden ReLU neurons
- one linear output neuron
- XOR training data
- full-batch gradient averaging
- learning rate: 0.10
- 4000 total parameter updates
- capacity injection at epoch 100
- 2 additional hidden ReLU neurons
- zero initial output weights for the new neurons
- seeds 0–9
- initialization offsets: 0, 1, 2, 3, 4, 5, 10, 100

The XOR positive class was:

- [0,1]
- [1,0]

The negative class was:

- [0,0]
- [1,1]

For each new neuron, measured the difference between its mean activation on the positive and negative classes:

positive mean - negative mean

The signed contribution contrast was then calculated by multiplying this contrast by the neuron's output weight.

Measurements were taken at relative steps:

- 0
- 1
- 5
- 20
- 100
- 500
- 3900

Success was defined as final loss < 1e-6.

### Results

Across the 80 seed/initialization combinations:

- successful rescues: 73
- failed rescues: 7

At the injection point:

- successful mean absolute activation contrast: 0.237771
- failed mean absolute activation contrast: 0.170996

By relative step 20:

- successful mean signed contribution contrast: 0.027023
- failed mean signed contribution contrast: 0.026344

By relative step 100:

- successful mean signed contribution contrast: 0.125827
- failed mean signed contribution contrast: 0.104029

By relative step 500:

- successful mean signed contribution contrast: 0.439675
- failed mean signed contribution contrast: 0.229867

At the final checkpoint:

- successful mean signed contribution contrast: 0.478300
- failed mean signed contribution contrast: 0.234112

The successful group therefore developed substantially stronger task-aligned contributions, especially after the first 100 post-injection updates.

### Failed Cases

The individual failures show that task alignment alone is not sufficient.

For example:

- seed 7, offset 3 reached signed contribution contrast 0.288305 at step 500 but still failed
- seed 3, offset 4 reached 0.370730 but still failed
- seed 3, offset 10 reached 0.460494 but still failed

These failed cases demonstrate that a new neuron can become positively aligned with the XOR class structure and still fail to produce a complete solution.

Other failures had much weaker task alignment, including:

- seed 3, offset 1: 0.007008
- seed 7, offset 0: 0.102815
- seed 7, offset 10: 0.319939
- seed 5, offset 100: 0.059778

### Conclusion

Successful rescue trajectories generally develop stronger task-aligned hidden contributions than failed trajectories.

This provides a more informative signal than raw hidden-weight displacement alone. By step 500, the successful group had a mean signed task contribution contrast of approximately 0.440 compared with approximately 0.230 for failed cases.

However, task alignment is not sufficient by itself. Several failed trajectories developed substantial positive-vs-negative contribution contrast but still converged to a loss near 0.25.

Therefore, successful rescue appears to require more than a single useful hidden feature. A hidden neuron can point in the correct class-separating direction without providing enough complementary structure to represent the complete XOR mapping.

This suggests that the interaction between the newly added neurons is important. The next experiment should examine whether successful rescue depends on the two new features providing complementary rather than redundant class-separating information.

## Experiment 057 — Hidden Feature Complementarity

### Question

Does successful rescue depend on the two newly added hidden neurons providing complementary rather than redundant features?

### Setup

Used the rescue configuration from Experiments 043–056.

- He initialization for the original width-2 network
- 2 initial hidden ReLU neurons
- one linear output neuron
- XOR training data
- full-batch gradient averaging
- learning rate: 0.10
- 4000 total parameter updates
- capacity injection at epoch 100
- 2 additional hidden ReLU neurons
- zero initial output weights for the new neurons
- seeds 0–9
- initialization offsets: 0, 1, 2, 3, 4, 5, 10, 100

At relative steps 0, 20, 100, 500, and 3900, the two new hidden neurons were compared using their four-example activation vectors.

Measured:

- cosine similarity between the two activation vectors
- independence, defined as 1 - cosine^2
- absolute determinant of the positive/negative class-response matrix
- combined class-contrast magnitude

Success was defined as final loss < 1e-6.

### Results

Across the 80 seed/initialization combinations:

- successful rescues: 73
- failed rescues: 7

At the injection point:

- successful mean activation-vector cosine: 0.212542
- failed mean cosine: 0.139491
- successful mean independence: 0.565860
- failed mean independence: 0.006652
- successful mean determinant magnitude: 0.118325
- failed mean determinant magnitude: 0.003004

By relative step 100:

- successful mean independence: 0.578864
- failed mean independence: 0.000000
- successful mean determinant magnitude: 0.139700
- failed mean determinant magnitude: approximately 0

By relative step 500:

- successful mean cosine: 0.165757
- failed mean cosine: 0.142857
- successful mean independence: 0.594244
- failed mean independence: 0.000000
- successful mean determinant magnitude: 0.174206
- failed mean determinant magnitude: 0.000000

At the final checkpoint:

- successful mean cosine: 0.160700
- failed mean cosine: 0.142857
- successful mean independence: 0.596909
- failed mean independence: 0.000000
- successful mean determinant magnitude: 0.171516
- failed mean determinant magnitude: 0.000000

The successful trajectories therefore maintained two substantially distinct activation directions, while the failed trajectories became redundant or effectively one-dimensional.

### Failed Cases

All seven failed cases had zero determinant magnitude at step 500.

Most failures had one new neuron completely inactive, producing an effectively one-feature rescue attempt.

The failures included:

- seed 7, offset 0: one active new feature and one inactive feature
- seed 3, offset 1: one inactive feature
- seed 7, offset 3: one inactive feature
- seed 3, offset 4: one inactive feature
- seed 3, offset 10: both features active but cosine similarity reached 1.0, indicating redundant activation vectors
- seed 7, offset 10: one inactive feature
- seed 5, offset 100: one inactive feature

The seed 3, offset 10 case is especially informative because its combined class contrast was relatively large, yet the two features were redundant and the run still failed.

### Conclusion

Successful rescue is strongly associated with complementary hidden features.

The successful trajectories developed and maintained two distinct activation directions, while every failed case was effectively reduced to a one-dimensional hidden representation by the step-500 complementarity measures.

This is a stronger structural explanation than raw activation count, gradient magnitude, or parameter displacement.

The result also explains why a single strongly task-aligned feature can still fail: XOR requires a representation with enough independent structure for the linear output layer to distinguish both positive examples from both negative examples.

However, this experiment is observational. It shows a strong association between feature complementarity and successful rescue but does not establish that complementarity itself causes success.

The next experiment should intervene on the initial relationship between the two new hidden features and test whether deliberately complementary initialization improves rescue reliability.

## Experiment 058 — Complementary Initialization

### Question

Does deliberately making the two newly added hidden-weight vectors orthogonal improve rescue reliability by encouraging complementary features?

### Setup

Used the rescue configuration from Experiments 043–057.

- He initialization for the original width-2 network
- 2 initial hidden ReLU neurons
- one linear output neuron
- XOR training data
- full-batch gradient averaging
- learning rate: 0.10
- 4000 total parameter updates
- capacity injection at epoch 100
- 2 additional hidden ReLU neurons
- zero initial output weights for the new neurons
- seeds 0–9
- initialization offsets: 0, 1, 2, 3, 4, 5, 10, 100

Two initialization conditions were compared:

1. Standard independent He initialization for the two new hidden neurons.
2. Orthogonal initialization in which the two new hidden-weight vectors were constructed to be perpendicular while preserving their individual weight norms.

The new output weights were zero in both conditions.

Each condition contained 80 seed/initialization combinations.

Success was defined as final loss < 1e-6.

### Results

| Initialization | Successful cases | Mean final loss |
|---|---:|---:|
| Standard | 73/80 | 0.021875000000 |
| Orthogonal hidden weights | 68/80 | 0.040625001049 |

The standard condition reproduced the previous 73/80 rescue rate.

The orthogonal hidden-weight condition produced only 68/80 successful rescues.

The orthogonal construction therefore did not improve reliability and instead reduced the observed success rate by 5 cases out of 80.

The measured activation-vector statistics immediately after injection were:

| Initialization | Mean activation-vector cosine | Mean independence | Mean determinant |
|---|---:|---:|---:|
| Standard | 0.206150 | 0.879430 | 0.108234 |
| Orthogonal hidden weights | 0.211525 | 0.884595 | 0.085659 |

Although the hidden weight vectors were exactly orthogonal in the intervention condition, their activation vectors were not. The activation representation depends on the input data, ReLU thresholding, and zero bias, so orthogonal parameter vectors do not imply orthogonal or complementary feature responses on the XOR examples.

### Conclusion

Forcing the two newly added hidden-weight vectors to be orthogonal did not improve rescue reliability.

The standard initialization succeeded on 73/80 cases, while the orthogonal initialization succeeded on 68/80.

This is an important distinction between parameter-space geometry and feature-space geometry. Orthogonal hidden-weight vectors do not guarantee useful complementary responses to the actual training data.

The experiment therefore does not support the hypothesis that simple weight-vector orthogonality is sufficient to create better rescue features.

The results from Experiment 057 remain more informative: successful rescue was associated with complementary activation behavior on the XOR examples, not merely geometric separation of the underlying weight vectors.

The next experiment should therefore construct complementarity in the actual activation patterns on the XOR inputs rather than imposing orthogonality on the raw hidden-weight vectors.

## Experiment 059 — Input-Space Complementarity

### Question

Does deliberately constructing complementary hidden features in the actual XOR input space improve rescue reliability?

### Setup

Used the rescue configuration from Experiments 043–058.

- He initialization for the original width-2 network
- 2 initial hidden ReLU neurons
- one linear output neuron
- XOR training data
- full-batch gradient averaging
- learning rate: 0.10
- 4000 total parameter updates
- capacity injection at epoch 100
- 2 additional hidden ReLU neurons
- zero initial output weights for the new neurons
- seeds 0–9
- initialization offsets: 0, 1, 2, 3, 4, 5, 10, 100

Two initialization conditions were compared:

1. Standard independent He initialization for the two new hidden neurons.
2. Input-space complementary initialization designed directly around the XOR training examples.

For the complementary condition, the two hidden weight vectors were constructed as:

- neuron 1: [a, -a]
- neuron 2: [-a, a]

with zero biases.

The scale `a` was matched to the average norm of the corresponding standard He-generated weight vectors so that the comparison preserved a comparable initialization scale while changing the feature directions.

On the four XOR inputs `[00, 01, 10, 11]`, this construction produces the activation patterns:

- neuron 1: `0010`
- neuron 2: `0100`

Thus each new neuron responds to one of the two positive XOR examples while remaining inactive on the two negative examples.

Each condition contained 80 seed/initialization combinations.

Success was defined as final loss < 1e-6.

### Results

| Initialization | Successful cases | Mean final loss |
|---|---:|---:|
| Standard He pair | 73/80 | 0.021875000000 |
| Input-space complementary features | 80/80 | 0.000000000000 |

The standard condition reproduced the previous 73/80 rescue rate.

The input-space complementary condition succeeded on all 80 combinations:

- offset 0: 10/10
- offset 1: 10/10
- offset 2: 10/10
- offset 3: 10/10
- offset 4: 10/10
- offset 5: 10/10
- offset 10: 10/10
- offset 100: 10/10

The complementary condition produced the same canonical activation pattern in every run:

`0010|0100`

with:

- success = 80
- failure = 0

### Conclusion

Deliberately constructing complementary features in the actual XOR input space produced a complete rescue in all 80 tested combinations, compared with 73/80 under standard initialization.

This provides experimental support for the hypothesis developed in Experiments 057 and 058: useful complementarity is a property of the features produced on the task inputs, not simply a property of geometric relationships between raw hidden-weight vectors.

Experiment 058 showed that forcing raw hidden-weight vectors to be orthogonal did not improve rescue reliability. Experiment 059 instead directly constructed two complementary XOR feature responses and increased the observed rescue rate from 91.25% to 100%.

The intervention is stronger evidence than the observational results in Experiment 057 because the feature relationship was deliberately manipulated before training.

However, this result is specific to the XOR task, this network architecture, this injection point, and the tested initialization scale. It does not establish that the exact `0010|0100` construction is generally optimal for other tasks.

The next experiment should separate the effect of exact XOR-targeted feature patterns from the broader idea of complementarity by testing alternative complementary feature constructions and determining which structural properties are actually necessary for reliable rescue.

## Experiment 060 — Complementarity Generalization

### Question

Does the perfect rescue observed in Experiment 059 depend on the exact `0010|0100` activation pattern, or does a broader form of hidden-feature diversity improve rescue reliability?

### Setup

Used the rescue configuration from Experiments 043–059.

- He initialization for the original width-2 network
- 2 initial hidden ReLU neurons
- one linear output neuron
- XOR training data
- full-batch gradient averaging
- learning rate: 0.10
- 4000 total parameter updates
- capacity injection at epoch 100
- 2 additional hidden ReLU neurons
- zero initial output weights for the new neurons
- seeds 0–9
- initialization offsets: 0, 1, 2, 3, 4, 5, 10, 100

Six new-feature initialization conditions were compared:

1. Standard independent He initialization
2. The Experiment 059 target-selective construction
3. An asymmetric version of the target-selective construction
4. An axis-pair construction using separate input dimensions
5. A hinge-style decomposition
6. A redundant pair using the same feature twice

Each condition contained 80 seed/initialization combinations.

Success was defined as final loss < 1e-6.

### Results

| Condition | Successful cases | Mean final loss |
|---|---:|---:|
| Standard He | 73/80 | 0.021875000000 |
| Target-selective | 80/80 | 0.000000000000 |
| Asymmetric-selective | 80/80 | 0.000000000000 |
| Axis-pair | 80/80 | 0.000000000000 |
| Hinge-decomposition | 56/80 | 0.076016616307 |
| Redundant | 56/80 | 0.091666666667 |

The target-selective condition reproduced the Experiment 059 result of 80/80.

The asymmetric-selective condition also reached 80/80, showing that the exact activation magnitudes are not required as long as the same selective support pattern is preserved.

The axis-pair condition produced a different activation structure:

`0011|0101`

and also achieved 80/80.

The redundant condition produced:

`0010|0010`

and achieved only 56/80.

The hinge condition produced:

`0000|0111`

where one newly added neuron was inactive on all four XOR examples, and also achieved only 56/80.

### Observation

The Experiment 059 activation pattern is not uniquely responsible for perfect rescue.

Multiple distinct feature constructions achieved 80/80 success.

The strongest contrast was between the successful axis-pair construction and the unsuccessful redundant construction. The axis pair created two different active response patterns, while the redundant condition created two identical patterns.

The hinge condition provided one active feature and one inactive feature and also failed at the same 56/80 rate observed for the redundant condition.

### Interpretation

These results strengthen the hypothesis that useful hidden-feature diversity is more important than reproducing one exact hand-crafted XOR representation.

However, the experiment does not establish that any pair of distinct activation patterns is sufficient.

The axis-pair features do not themselves form a complete linearly separable XOR representation. Their successful interaction with the existing hidden representation shows that newly added features can help optimization without individually solving the task.

The failed hinge and redundant conditions suggest that adding capacity without sufficiently distinct useful responses does not provide the same rescue benefit.

### Lesson

Feature complementarity should be understood in terms of the responses produced on the actual task inputs.

Parameter-space differences alone are not enough.

The exact successful pattern from Experiment 059 is one example of a broader phenomenon: adding distinct hidden responses can provide new directions for the optimization process.

### Status

Experiment 060 complete.

### Next Direction

Systematically vary the relationship between the two new hidden activation patterns to determine whether rescue reliability depends on feature diversity, activation coverage, overlap, or some minimum amount of independence between the new features.

## Experiment 061 — Feature Overlap Sweep

### Question

What structural relationship between newly added hidden features best explains rescue reliability?

Experiment 060 showed that multiple distinct feature constructions could produce perfect rescue while redundant and effectively one-feature constructions performed worse.

Experiment 061 therefore varied the amount of activation overlap between the two newly added hidden features while keeping their weight norms matched.

### Setup

Used the rescue configuration established in Experiments 043–060.

- He initialization for the original width-2 network
- 2 initial hidden ReLU neurons
- one linear output neuron
- XOR training data
- full-batch gradient averaging
- learning rate: 0.10
- 4000 total training epochs
- capacity injection at epoch 100
- 2 additional hidden ReLU neurons
- zero initial output weights for the new neurons
- seeds 0–9
- initialization offsets: 0, 1, 2, 3, 4, 5, 10, 100

Eight feature-pair conditions were compared.

For each pair, the binary activation pattern across the XOR examples was measured.

The following quantities were recorded:

- activation overlap
- activation union
- Jaccard overlap
- total active-example coverage
- final training success

Each condition contained 80 seed/initialization combinations.

Success was defined as final loss < 1e-6.

### Results

| Feature pair | Overlap | Union | Jaccard | Active coverage | Successful |
|---|---:|---:|---:|---:|---:|
| 0010\|0010 | 1 | 1 | 1.000 | 2 | 56/80 |
| 0011\|0011 | 2 | 2 | 1.000 | 4 | 57/80 |
| 0010\|0111 | 1 | 3 | 0.333 | 4 | 64/80 |
| 0011\|0111 | 2 | 3 | 0.667 | 5 | 80/80 |
| 0011\|0101 | 1 | 3 | 0.333 | 4 | 80/80 |
| 0010\|0100 | 0 | 2 | 0.000 | 2 | 80/80 |
| 0001\|0101 | 1 | 2 | 0.500 | 3 | 63/80 |
| 0000\|0111 | 0 | 3 | 0.000 | 3 | 56/80 |

### Observation

Activation overlap alone does not explain rescue reliability.

Completely redundant features performed poorly:

- `0010|0010` → 56/80
- `0011|0011` → 57/80

However, reducing overlap did not automatically guarantee success.

For example:

- `0010|0111` had Jaccard overlap 0.333 but succeeded only 64/80.
- `0001|0101` had Jaccard overlap 0.500 but succeeded only 63/80.

In contrast:

- `0011|0111` succeeded 80/80.
- `0011|0101` succeeded 80/80.
- `0010|0100` succeeded 80/80.

The two zero-overlap conditions also behaved very differently:

- `0010|0100` → 80/80
- `0000|0111` → 56/80

The difference is that the first gives separate active responses to the two positive XOR examples, while the second contains one completely inactive feature.

### Interpretation

The results rule out Jaccard overlap as a sufficient explanation of rescue success.

Feature overlap is informative because highly redundant features reduce the amount of new information supplied by the injected neurons, but overlap must be considered together with which training examples are represented.

The identity of the active XOR examples appears to matter.

Successful feature pairs tend to provide useful responses covering both positive XOR examples without collapsing the new representation into a redundant or effectively one-feature system.

The failed `0000|0111` condition demonstrates that zero overlap by itself is not sufficient. One feature can be completely inactive while the other remains active on several examples, leaving insufficient new structure for reliable rescue.

### Lesson

Feature complementarity is not a single scalar property.

Two hidden features can have:

- low overlap but poor usefulness,
- high overlap but still provide some useful structure,
- or low overlap while covering the task in a way that reliably improves optimization.

The actual identity of the examples represented by each feature matters.

### Important Understanding

The investigation has moved from asking whether two features are different to asking what information those features provide about the task.

For XOR, the positive examples are:

- [0,1]
- [1,0]

while the negative examples are:

- [0,0]
- [1,1]

A feature pair may therefore need to be evaluated by its coverage and relationship to these task classes, not only by geometric independence.

### Status

Experiment 061 complete.

### Next Direction

Systematically vary which XOR examples are activated by the two new features and determine whether successful rescue is predicted by positive-class coverage, negative-class exclusion, or the combination of both.

## Experiment 062 — Task Coverage Matrix

### Question

Which task examples must newly injected hidden features cover for rescue to become reliable?

Experiment 061 showed that activation overlap alone did not explain rescue success. Experiment 062 therefore fixed the first new feature at `0010` and systematically varied the second feature across the 14 activation patterns realizable by a single affine ReLU unit on the four XOR inputs.

The goal was to separate positive-class coverage, negative-class coverage, overlap, and the exact identity of the represented examples.

### Setup

Used the rescue configuration established in Experiments 043–061.

- He initialization for the original width-2 network
- 2 initial hidden ReLU neurons
- one linear output neuron
- XOR training data
- full-batch gradient averaging
- learning rate: 0.10
- 4000 total training epochs
- capacity injection at epoch 100
- 2 additional hidden ReLU neurons
- zero initial output weights for the new neurons
- seeds 0–9
- initialization offsets: 0, 1, 2, 3, 4, 5, 10, 100

The first injected feature was fixed at:

`0010`

The second injected feature was varied across all linearly separable activation patterns realizable by a single affine ReLU unit on the four XOR inputs.

Feature weight norms were matched across conditions.

Each condition contained 80 seed/initialization combinations.

Success was defined as final loss < 1e-6.

### Results

| Second feature | Combined pattern | Overlap | Union | Jaccard | Positive coverage | Negative coverage | Successful |
|---|---|---:|---:|---:|---:|---:|---:|
| 0000 | 0010\|0000 | 0 | 1 | 0.000 | 1 | 0 | 56/80 |
| 0001 | 0010\|0001 | 0 | 2 | 0.000 | 1 | 1 | 63/80 |
| 0010 | 0010\|0010 | 1 | 1 | 1.000 | 1 | 0 | 56/80 |
| 0011 | 0010\|0011 | 1 | 2 | 0.500 | 1 | 1 | 76/80 |
| 0100 | 0010\|0100 | 0 | 2 | 0.000 | 2 | 0 | 80/80 |
| 0101 | 0010\|0101 | 0 | 3 | 0.000 | 2 | 1 | 80/80 |
| 0111 | 0010\|0111 | 1 | 3 | 0.333 | 2 | 1 | 64/80 |
| 1000 | 0010\|1000 | 0 | 2 | 0.000 | 1 | 1 | 64/80 |
| 1010 | 0010\|1010 | 1 | 2 | 0.500 | 1 | 1 | 64/80 |
| 1011 | 0010\|1011 | 1 | 3 | 0.333 | 1 | 2 | 80/80 |
| 1100 | 0010\|1100 | 0 | 3 | 0.000 | 2 | 1 | 64/80 |
| 1101 | 0010\|1101 | 0 | 4 | 0.000 | 2 | 2 | 80/80 |
| 1110 | 0010\|1110 | 1 | 3 | 0.333 | 2 | 1 | 64/80 |
| 1111 | 0010\|1111 | 1 | 4 | 0.250 | 2 | 2 | 80/80 |

### Observation

The strongest result is that positive-class coverage is important, but it is not sufficient by itself.

Features covering both positive XOR examples sometimes produced perfect rescue:

- `0010|0100` → 80/80
- `0010|0101` → 80/80
- `0010|1101` → 80/80
- `0010|1111` → 80/80

However, other conditions that also covered both positive examples were less reliable:

- `0010|0111` → 64/80
- `0010|1100` → 64/80
- `0010|1110` → 64/80

Therefore, simply covering both positive examples does not guarantee rescue.

Likewise, covering only one positive example does not always fail:

- `0010|0011` → 76/80
- `0010|1011` → 80/80

This means positive-class coverage is informative, but not a complete explanation.

### Interpretation

Experiment 062 rules out a simple task-coverage rule such as:

> Rescue succeeds whenever both positive XOR examples are represented.

It also rules out the idea that negative-class coverage is simply harmful. Some successful conditions included one or two negative examples:

- `0010|0101` → 80/80
- `0010|1011` → 80/80
- `0010|1101` → 80/80
- `0010|1111` → 80/80

The exact identity of the examples represented by a feature pair matters, but the binary activation pattern still does not fully predict the outcome.

An important remaining variable is the magnitude of the activations, not just whether they are zero or nonzero.

Two feature pairs can have similar binary task coverage while producing different activation values and different optimization dynamics.

The successful and failed conditions therefore suggest that rescue depends on both:

- which training examples the new features respond to
- how strongly they respond to those examples

### Lesson

Feature complementarity is a task-dependent representation property rather than a simple overlap score or binary coverage count.

For XOR, knowing that a feature is active on an example is not enough. The magnitude of the feature response may determine whether the newly injected representation provides a sufficiently useful optimization direction.

The investigation should therefore move from binary activation patterns to activation margins and feature strength.

### Important Understanding

The investigation has progressed through three levels:

1. Are the new features different?
2. Which XOR examples do the new features represent?
3. How strongly do the new features represent those examples?

Experiment 061 addressed the first question.

Experiment 062 addressed the second question.

The next experiment should isolate the third.

### Status

Experiment 062 complete.

### Next Direction

Experiment 063 should hold the binary activation pattern fixed while systematically varying the activation magnitude or margin of the injected features.

The goal is to determine whether rescue reliability changes continuously with feature strength and whether a minimum useful activation margin exists.

## Experiment 063 — Feature Strength Sweep

### Question

How does the magnitude of newly injected hidden features affect rescue reliability?

Experiment 062 showed that binary task coverage alone did not fully explain rescue success. Experiment 063 therefore held the exact binary activation pattern fixed at `0010|0100` and varied only the strength of the injected features.

The goal was to determine whether rescue reliability changes with activation magnitude.

### Setup

Used the rescue configuration established in Experiments 043–062.

- He initialization for the original width-2 network
- 2 initial hidden ReLU neurons
- one linear output neuron
- XOR training data
- full-batch gradient averaging
- learning rate: 0.10
- 4000 total training epochs
- capacity injection at epoch 100
- 2 additional hidden ReLU neurons
- zero initial output weights for the new neurons
- seeds 0–9
- initialization offsets: 0, 1, 2, 3, 4, 5, 10, 100

The two injected features were fixed to the complementary input-space construction:

- first feature: `0010`
- second feature: `0100`

The entire affine feature vectors were multiplied by a strength factor.

Tested strengths:

- 0.01
- 0.025
- 0.05
- 0.10
- 0.25
- 0.50
- 1.00
- 2.00
- 4.00

Because each feature was multiplied by a positive scalar, the binary activation pattern remained unchanged.

Each strength contained 80 seed/initialization combinations.

Success was defined as final loss < 1e-6.

### Results

| Feature strength | Positive activation | Binary pattern | Successful |
|---:|---:|---|---:|
| 0.010 | 0.007212 | 0010\|0100 | 80/80 |
| 0.025 | 0.018031 | 0010\|0100 | 80/80 |
| 0.050 | 0.036062 | 0010\|0100 | 80/80 |
| 0.100 | 0.072124 | 0010\|0100 | 80/80 |
| 0.250 | 0.180310 | 0010\|0100 | 80/80 |
| 0.500 | 0.360620 | 0010\|0100 | 80/80 |
| 1.000 | 0.721239 | 0010\|0100 | 80/80 |
| 2.000 | 1.442479 | 0010\|0100 | 80/80 |
| 4.000 | 2.884958 | 0010\|0100 | 80/80 |

All 9 strength conditions achieved:

- 80/80 successful runs
- final mean loss = 0.000000000000

Total runs:

720

Total successful runs:

720/720

### Observation

Feature strength had no effect on eventual rescue reliability across the tested range.

The weakest condition, strength `0.01`, produced only approximately `0.007212` activation on the relevant positive example, yet still achieved 80/80 successful rescues.

Increasing feature strength by a factor of 400, from `0.01` to `4.00`, did not change the final success rate.

The binary feature pattern also remained identical in every condition:

`0010|0100`

### Interpretation

Experiment 063 provides evidence that the exact activation magnitude of the injected complementary features is not the determining factor for eventual rescue under this training budget.

The important distinction is that this experiment measured only the final outcome after 4000 training epochs.

A weak feature may still provide a useful optimization direction but require more training updates before its contribution becomes substantial.

Therefore, the result rules out a simple claim such as:

> Rescue requires a large activation margin.

It does not yet rule out the possibility that feature strength affects the speed of rescue.

### Lesson

A representation can be extremely weak in magnitude and still provide enough useful structure for gradient descent to eventually recover the XOR task.

This strengthens the evidence that the structural identity of the feature response matters more than its raw activation magnitude.

However, eventual success and convergence speed are different questions.

### Important Understanding

The investigation has now tested:

1. Feature difference
2. Task coverage
3. Feature activation strength

Binary pattern remained fixed while strength changed, and eventual success remained constant.

The remaining question is whether strength changes the trajectory even when the final destination is the same.

### Status

Experiment 063 complete.

### Next Direction

Experiment 064 should keep the `0010|0100` feature pattern fixed while varying feature strength over a wider range and measuring convergence speed.

The primary metric should be the first epoch at which loss falls below several thresholds, rather than only final success.

This will determine whether weak injected features learn more slowly and whether a practical minimum feature strength exists for rapid rescue.

## Experiment 064 — Feature Strength Convergence

### Question

Does feature strength affect how quickly a complementary injected representation rescues a failing trajectory, even when eventual rescue remains reliable?

Experiment 063 showed that the fixed `0010|0100` feature pair produced 80/80 successful rescues across a wide range of feature strengths. Experiment 064 therefore extended the range to much weaker features and measured convergence speed rather than final success alone.

### Setup

Used the same rescue configuration as Experiment 063.

- He initialization for the original width-2 network
- 2 initial hidden ReLU neurons
- one linear output neuron
- XOR training data
- full-batch gradient averaging
- learning rate: 0.10
- 4000 total training epochs
- capacity injection at epoch 100
- 2 additional hidden ReLU neurons
- zero initial output weights for the new neurons
- seeds 0–9
- initialization offsets: 0, 1, 2, 3, 4, 5, 10, 100

The injected feature pattern was fixed at:

`0010|0100`

Only feature strength was varied.

Tested strengths:

- 0.000001
- 0.00001
- 0.0001
- 0.001
- 0.003
- 0.01
- 0.03
- 0.10
- 1.00
- 4.00

Each strength contained 80 seed/initialization combinations.

Convergence was measured after injection using the first post-injection update at which total loss fell below:

- 1e-2
- 1e-4
- 1e-6

Success remained defined as final loss < 1e-6.

### Results

| Feature strength | Positive activation | Successful | Mean steps to 1e-2 | Mean steps to 1e-4 | Mean steps to 1e-6 |
|---:|---:|---:|---:|---:|---:|
| 0.000001 | 0.000000721 | 80/80 | 1231.56 | 1446.00 | 1651.22 |
| 0.000010 | 0.000007212 | 80/80 | 1116.89 | 1328.55 | 1530.97 |
| 0.000100 | 0.000072124 | 80/80 | 981.86 | 1195.46 | 1399.36 |
| 0.001000 | 0.000721239 | 80/80 | 823.98 | 1030.74 | 1228.09 |
| 0.003000 | 0.002163718 | 80/80 | 742.42 | 939.76 | 1127.86 |
| 0.010000 | 0.007212395 | 80/80 | 640.50 | 824.73 | 1002.83 |
| 0.030000 | 0.021637184 | 80/80 | 539.79 | 719.76 | 894.91 |
| 0.100000 | 0.072123945 | 80/80 | 426.94 | 606.56 | 782.27 |
| 1.000000 | 0.721239451 | 80/80 | 187.05 | 348.86 | 516.74 |
| 4.000000 | 2.884957804 | 80/80 | 61.96 | 201.91 | 367.30 |

All 10 strength conditions achieved:

- 80/80 successful runs

Total:

720/720 successful runs.

### Observation

Feature strength strongly affected convergence speed while leaving eventual rescue unchanged.

The weakest tested feature strength, `0.000001`, still rescued every run, but required an average of 1651.22 post-injection updates to reach loss below 1e-6.

The strongest tested feature strength, `4.0`, also rescued every run and required only 367.30 updates on average.

The convergence-time relationship was consistent across all three loss thresholds.

At the 1e-2 threshold:

- strength `0.000001`: 1231.56 steps
- strength `4.0`: 61.96 steps

At the 1e-4 threshold:

- strength `0.000001`: 1446.00 steps
- strength `4.0`: 201.91 steps

At the 1e-6 threshold:

- strength `0.000001`: 1651.22 steps
- strength `4.0`: 367.30 steps

### Interpretation

Experiment 064 separates two effects that were previously difficult to distinguish.

Feature structure appears to determine whether the injected representation provides a useful rescue direction.

Feature strength affects how quickly that direction becomes influential enough to drive the network toward the solution.

The results do not support the existence of a minimum feature strength required for eventual rescue within the tested range. Even an activation of approximately `0.000000721` was eventually sufficient for all 80 seed/initialization combinations.

However, weak features are substantially slower.

Increasing strength from `0.000001` to `4.0` reduced the mean time to the 1e-6 threshold by approximately a factor of 4.5.

### Lesson

A weak but structurally useful feature can eventually produce the same solution as a strong feature.

The difference is optimization efficiency rather than final representational capability.

This suggests that the magnitude of a useful feature may act more like a learning-rate multiplier for the newly available direction than like a hard requirement for representation.

### Important Understanding

The investigation now has a clearer hierarchy:

1. **Feature structure** — determines whether the injected neurons provide useful complementary information.
2. **Feature strength** — determines how quickly that information becomes effective during optimization.
3. **Training budget** — determines whether a weak but useful feature has enough time to produce the desired result.

Experiment 063 showed that strength did not change eventual rescue across the original tested range.

Experiment 064 showed that strength strongly changes convergence speed when the range is extended much lower.

### Status

Experiment 064 complete.

### Next Direction

Experiment 065 should test feature strength against training budget directly.

The goal is to determine whether weak features that eventually succeed simply require more optimization time, and whether there is a practical boundary where increasing the training budget can compensate for extremely weak injected features.

The experiment should vary both:

- feature strength
- total post-injection training budget

while keeping the `0010|0100` feature structure fixed.

## Experiment 065 — Strength × Training Budget

### Question

Can additional training time compensate for very weak but structurally useful injected features?

Experiment 064 showed that feature strength strongly affected convergence speed while every tested strength eventually rescued all 80 runs. Experiment 065 tests whether additional optimization time can compensate for extremely weak features.

### Setup

The feature structure was fixed at `0010|0100`.

The network started with width 2 and received 2 additional hidden neurons at epoch 100.

Training used:

- He initialization
- full-batch training
- learning rate `0.10`
- seeds `0–9`
- initialization offsets `[0, 1, 2, 3, 4, 5, 10, 100]`
- zero initial output weights for the injected neurons
- success threshold: loss < `1e-6`

Feature strengths tested:

`1e-8`, `1e-7`, `1e-6`, `1e-5`, `1e-4`, `1e-3`, `1e-2`, `0.1`, `1.0`, `4.0`

Post-injection training budgets tested:

`100`, `250`, `500`, `1000`, `2000`, `3900` updates.

Each strength/seed/initialization combination was trained once through the full 3900-update budget, with loss recorded at each selected budget. Therefore the 60 strength × budget cells are measurements from 800 underlying trajectories rather than 4800 independent runs.

### Results

| Feature strength | 100 | 250 | 500 | 1000 | 2000 | 3900 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00000001 | 0/80 | 0/80 | 0/80 | 0/80 | 34/80 | 80/80 |
| 0.00000010 | 0/80 | 0/80 | 0/80 | 0/80 | 63/80 | 80/80 |
| 0.00000100 | 0/80 | 0/80 | 0/80 | 0/80 | 64/80 | 80/80 |
| 0.00001000 | 0/80 | 0/80 | 0/80 | 0/80 | 64/80 | 80/80 |
| 0.00010000 | 0/80 | 0/80 | 0/80 | 0/80 | 77/80 | 80/80 |
| 0.00100000 | 0/80 | 0/80 | 0/80 | 0/80 | 80/80 | 80/80 |
| 0.01000000 | 0/80 | 0/80 | 0/80 | 54/80 | 80/80 | 80/80 |
| 0.10000000 | 0/80 | 0/80 | 0/80 | 71/80 | 80/80 | 80/80 |
| 1.00000000 | 0/80 | 0/80 | 45/80 | 80/80 | 80/80 | 80/80 |
| 4.00000000 | 0/80 | 27/80 | 64/80 | 74/80 | 80/80 | 80/80 |

### Observation

Training budget clearly compensated for weak feature strength.

The weakest feature tested, `1e-8`, produced no successful runs through 1000 post-injection updates, but reached 34/80 by 2000 updates and 80/80 by 3900 updates.

At strength `1e-3`, all 80 runs succeeded within 2000 updates.

At strength `0.01`, 54/80 succeeded by 1000 updates and all 80 succeeded by 2000.

At strength `1.0`, 45/80 succeeded by 500 updates and all 80 succeeded by 1000.

At strength `4.0`, 27/80 succeeded by 250 updates and 64/80 by 500 updates.

At the full 3900-update budget, every tested feature strength achieved 80/80 success.

### Interpretation

Experiment 065 directly supports the idea that a sufficiently weak but structurally useful feature can be rescued by giving optimization more time.

There was no tested feature strength that remained permanently unsuccessful when the full 3900-update budget was available.

The main tradeoff is:

**weaker feature → slower optimization → larger training budget required**

**stronger feature → faster optimization → smaller training budget required**

The effect is not perfectly monotonic at every intermediate budget. For example, strength `4.0` reached 74/80 at 1000 updates while strength `1.0` reached 80/80. This shows that increasing feature magnitude does not guarantee a strictly better optimization trajectory at every fixed training budget.

### Mean Loss

The mean-loss results showed the same overall relationship.

For strength `1e-8`:

- 100 updates: `0.401794947`
- 250 updates: `0.367901137`
- 500 updates: `0.266490770`
- 1000 updates: `0.213608871`
- 2000 updates: `0.030108976`
- 3900 updates: approximately `0`

For strength `1.0`:

- 100 updates: `0.112220870`
- 250 updates: `0.004186262`
- 500 updates: `0.000022448`
- 1000 updates: `0.000000017`
- 2000 updates: approximately `0`
- 3900 updates: approximately `0`

For strength `4.0`:

- 100 updates: `0.003781465`
- 250 updates: `0.000139592`
- 500 updates: `0.000010615`
- 1000 updates: `0.000000147`
- 2000 updates: approximately `0`
- 3900 updates: approximately `0`

### Lesson

Feature structure and feature strength play different roles.

The complementary `0010|0100` structure makes the injected representation useful. Strength determines how quickly that useful direction can influence optimization.

Training budget can compensate for extremely weak strength, but the cost is slower convergence.

A weak feature is therefore not necessarily a bad feature. It may simply require more optimization steps before its contribution becomes effective.

### Important Understanding

The investigation now suggests three connected factors:

1. **Feature structure** determines whether the injected neurons provide useful complementary information.
2. **Feature strength** determines how quickly that information influences learning.
3. **Training budget** determines whether optimization has enough time to exploit a weak but useful feature.

Experiments 063 and 064 established the strength/convergence relationship.

Experiment 065 adds the missing budget dimension and shows that additional training can compensate for extremely weak but useful injected features.

### Status

Experiment 065 complete.

### Next Direction

The next experiments should test whether the same strength-versus-budget relationship holds when the injected features are less ideal than the fixed `0010|0100` pair, and whether feature quality, feature strength, and optimization time interact.

## Experiment 066 — Structure × Strength × Training Budget

### Question

Can additional training time compensate for weaker feature structure, in the same way that it can compensate for weak feature strength?

Experiment 065 showed that extremely weak but useful features can eventually converge when given enough optimization time. Experiment 066 tests whether additional training can similarly overcome structurally weaker feature pairs.

### Setup

Three fixed feature structures were tested:

- `0010|0100` — complementary baseline
- `0010|0111` — partial-overlap structure
- `0010|0010` — redundant structure

Feature strengths:

`0.001`, `0.01`, `0.1`, `1.0`, `4.0`

Post-injection training budgets:

`500`, `1000`, `2000`, `3900` updates

All other conditions remained fixed:

- width 2 before injection
- 2 hidden neurons added at epoch 100
- He initialization
- full-batch training
- learning rate `0.10`
- seeds `0–9`
- initialization offsets `[0, 1, 2, 3, 4, 5, 10, 100]`
- new output weights initialized to zero
- success threshold: loss < `1e-6`

Each structure/strength/seed/initialization combination was trained once through 3900 updates, with losses recorded at each budget.

### Results

#### `0010|0100` — complementary

| Strength | 500 | 1000 | 2000 | 3900 |
|---:|---:|---:|---:|---:|
| 0.001 | 0/80 | 0/80 | 80/80 | 80/80 |
| 0.010 | 0/80 | 54/80 | 80/80 | 80/80 |
| 0.100 | 0/80 | 71/80 | 80/80 | 80/80 |
| 1.000 | 45/80 | 80/80 | 80/80 | 80/80 |
| 4.000 | 64/80 | 74/80 | 80/80 | 80/80 |

#### `0010|0111` — partial overlap

| Strength | 500 | 1000 | 2000 | 3900 |
|---:|---:|---:|---:|---:|
| 0.001 | 0/80 | 0/80 | 55/80 | 55/80 |
| 0.010 | 0/80 | 23/80 | 48/80 | 48/80 |
| 0.100 | 0/80 | 38/80 | 55/80 | 55/80 |
| 1.000 | 6/80 | 55/80 | 56/80 | 56/80 |
| 4.000 | 44/80 | 57/80 | 63/80 | 63/80 |

#### `0010|0010` — redundant

| Strength | 500 | 1000 | 2000 | 3900 |
|---:|---:|---:|---:|---:|
| 0.001 | 0/80 | 1/80 | 48/80 | 48/80 |
| 0.010 | 0/80 | 24/80 | 48/80 | 48/80 |
| 0.100 | 0/80 | 34/80 | 55/80 | 55/80 |
| 1.000 | 18/80 | 45/80 | 56/80 | 56/80 |
| 4.000 | 34/80 | 48/80 | 56/80 | 56/80 |

### Observation

The complementary structure was the only structure that eventually reached 80/80 at every tested feature strength.

The partial-overlap structure reached a maximum of 63/80.

The redundant structure reached a maximum of 56/80.

Most importantly, the weaker structures showed little or no improvement between 2000 and 3900 updates. Additional training time helped them optimize, but did not remove their final performance ceiling.

### Mean Loss

The complementary structure converged to approximately zero loss.

The partial-overlap structure plateaued at nonzero mean loss:

- strength `0.001`: `0.078125000`
- strength `0.010`: `0.100000000`
- strength `0.100`: `0.078125000`
- strength `1.000`: `0.075000000`
- strength `4.000`: `0.053125000`

The redundant structure also plateaued:

- strength `0.001`: `0.116666667`
- strength `0.010`: `0.116666667`
- strength `0.100`: `0.094791667`
- strength `1.000`: `0.091666667`
- strength `4.000`: `0.091666667`

### Interpretation

Experiment 066 establishes a boundary on what training time can compensate for.

Experiment 065 showed that weak versions of a useful feature pair can eventually reach the desired solution when given enough optimization time.

Experiment 066 shows that this does not hold for structurally inadequate feature pairs.

For the complementary `0010|0100` structure, strength and training budget mainly affected convergence speed.

For `0010|0111` and `0010|0010`, increasing strength and training budget improved results at intermediate checkpoints, but the network eventually reached structure-dependent ceilings.

The strongest complementary condition reached 80/80.

The strongest partial-overlap condition reached only 63/80.

The strongest redundant condition reached only 56/80.

A stronger version of a poor representation therefore remains limited by the representation itself.

### Lesson

The hierarchy from the previous experiments is becoming clearer:

1. **Feature structure** determines the available representational directions and can impose a hard performance ceiling.
2. **Feature strength** determines how quickly those directions influence optimization.
3. **Training budget** determines how much time optimization has to exploit the available directions.

Strength and training time can compensate for weak magnitude.

They cannot reliably compensate for missing or redundant structure.

### Important Understanding

Experiments 059–062 established that complementary feature structure is strongly associated with successful rescue.

Experiments 063–064 showed that feature strength primarily affects convergence speed within a useful structure.

Experiment 065 showed that additional training can compensate for extremely weak but useful features.

Experiment 066 shows the opposite boundary: additional training cannot fully compensate for structurally inadequate features.

This suggests that representational quality comes before optimization efficiency. A useful direction must exist before strength and training time can exploit it.

### Status

Experiment 066 complete.

### Next Direction

The next experiment should investigate which measurable property of feature structure predicts the performance ceiling most precisely.

Candidates include feature rank, target alignment, negative-example contamination, and the geometry of the represented outputs.

## Experiment 067 — Representation vs. Optimization

### Question

When a rescue succeeds or fails, is the limiting factor the representation itself, or the ability of gradient descent to exploit that representation?

Experiment 066 showed that stronger features and longer training could not fully overcome structurally weaker feature pairs. Experiment 067 measures the representational capacity available immediately at injection, before the newly added output weights are trained.

### Setup

The same three structures from Experiment 066 were tested:

- `0010|0100` — complementary
- `0010|0111` — partial overlap
- `0010|0010` — redundant

Feature strengths:

`0.001`, `0.01`, `0.1`, `1.0`, `4.0`

For every run, the hidden representation at the injection point was held fixed and an analytic best linear output readout was calculated.

The readout included an intercept and the available hidden activations.

This produces a representation-capacity measurement independent of the network's gradient-descent path.

The same run was then trained for 3900 post-injection updates.

There were 80 runs for each structure/strength combination.

Success was defined as loss < `1e-6`.

### Definitions

**Readout success** means the fixed hidden representation at injection could already fit XOR with a linear output layer.

**Readout → fail** means the representation could already fit XOR, but ordinary training did not reach the success threshold.

**New solution** means the representation could not initially fit XOR with a linear readout, but training changed the hidden representation enough for the final network to succeed.

### Results

#### `0010|0100` — complementary

| Strength | Readout | Final | Readout → Fail | New Solution |
|---:|---:|---:|---:|---:|
| 0.001 | 80/80 | 80/80 | 0 | 0 |
| 0.010 | 80/80 | 80/80 | 0 | 0 |
| 0.100 | 80/80 | 80/80 | 0 | 0 |
| 1.000 | 80/80 | 80/80 | 0 | 0 |
| 4.000 | 80/80 | 80/80 | 0 | 0 |

The complementary structure was already linearly sufficient in every run at injection.

Every one of those runs also successfully trained.

#### `0010|0111` — partial overlap

| Strength | Readout | Final | Readout → Fail | New Solution |
|---:|---:|---:|---:|---:|
| 0.001 | 64/80 | 55/80 | 16 | 7 |
| 0.010 | 64/80 | 48/80 | 16 | 0 |
| 0.100 | 64/80 | 55/80 | 9 | 0 |
| 1.000 | 64/80 | 56/80 | 8 | 0 |
| 4.000 | 64/80 | 63/80 | 1 | 0 |

The representation was immediately sufficient in 64/80 runs regardless of strength.

Increasing strength improved the ability of training to exploit that representation, but did not change the initial representational count.

At strength `4.0`, only one run had a sufficient representation at injection but failed to reach the final success threshold.

At strength `0.001`, 7 runs created a successful representation during training even though their initial representation was insufficient.

#### `0010|0010` — redundant

| Strength | Readout | Final | Readout → Fail | New Solution |
|---:|---:|---:|---:|---:|
| 0.001 | 48/80 | 48/80 | 0 | 0 |
| 0.010 | 48/80 | 48/80 | 0 | 0 |
| 0.100 | 48/80 | 55/80 | 0 | 7 |
| 1.000 | 48/80 | 56/80 | 0 | 8 |
| 4.000 | 48/80 | 56/80 | 0 | 8 |

The redundant structure was immediately sufficient in only 48/80 runs.

Unlike the complementary structure, increasing feature strength did not improve the initial representational capacity.

Training did create new successful representations in some cases at higher strengths, but the final ceiling remained 56/80.

### Observation

The strongest result is that the initial readout counts were determined by feature structure rather than feature strength.

For `0010|0100`:

- readout success remained 80/80 at every strength.

For `0010|0111`:

- readout success remained 64/80 at every strength.

For `0010|0010`:

- readout success remained 48/80 at every strength.

This is expected because positive scaling changes feature magnitude without changing the underlying activation pattern. The representational geometry at the pattern level therefore remains unchanged, while optimization speed can still change substantially.

### Representation vs. Optimization

Experiment 067 separates two different failure modes.

**Representational failure**

A subset of runs begins with a hidden representation that cannot linearly realize XOR.

Examples:

- `0010|0111`: 16/80 runs initially insufficient
- `0010|0010`: 32/80 runs initially insufficient

Additional training can sometimes create a new representation, but not reliably enough to reach the complementary structure's performance.

**Optimization failure**

A run can begin with a representation that is already sufficient, yet training can still fail.

This occurred most clearly for `0010|0111`.

At strength `0.001`, 64 runs had a sufficient representation at injection, but 16 of those did not reach final success.

At strength `4.0`, only 1 such run failed.

This demonstrates that having the right representation does not automatically guarantee that gradient descent will exploit it successfully.

### Important Result

The experiment produces a useful decomposition:

**Representation determines what solution is available.**

**Optimization determines whether training reaches that solution.**

For the complementary structure, both conditions were satisfied in every run.

For the partial-overlap structure, some runs had sufficient representation but still failed optimization.

For the redundant structure, the initial representation was insufficient in many runs, and training could only create new successful representations in a limited subset.

### Interpretation

Experiments 065 and 066 showed that strength and training budget influence convergence, but structure can impose a final ceiling.

Experiment 067 now shows why.

Feature strength does not fundamentally change the binary structure of a positive ReLU feature. It scales the available activations, but does not create new activation patterns.

Therefore:

- structure changes representational capacity
- strength changes optimization dynamics
- training budget gives optimization more opportunity to exploit the representation

A stronger feature can make an existing useful representation easier to learn from, but it does not automatically make a poor representation expressive enough.

### Lesson

A neural network can fail for two very different reasons.

It can fail because the available representation cannot express the target sufficiently.

Or it can fail because the representation is sufficient, but optimization does not find the correct output parameters.

Those failure modes should not be treated as the same problem.

### Important Understanding

The investigation now supports a four-stage picture:

1. **Feature structure** determines the representational possibilities.
2. **Feature strength** determines how strongly those possibilities influence gradients.
3. **Training budget** determines how much time optimization has to exploit them.
4. **Optimization trajectory** determines whether a sufficient representation actually becomes a successful trained solution.

Experiment 067 provides a direct measurement separating the first and fourth stages.

### Status

Experiment 067 complete.

### Next Direction

The next experiment should measure how the hidden representation changes during training in runs that begin representation-sufficient versus representation-insufficient.

The goal is to determine whether successful training mainly preserves an already-good representation, refines it, or constructs a substantially different one.

## Experiment 068 — Representation Trajectory

### Question

When a rescue succeeds or fails, does training preserve an already-useful hidden representation, refine it, or construct a new useful representation?

Experiment 067 showed that representation sufficiency and optimization success are not identical. Some runs began with a hidden representation that could already solve XOR with an optimal linear readout but still failed under ordinary training.

Experiment 068 therefore measured the hidden representation repeatedly during post-injection training.

The goal was to determine whether successful and failed trajectories follow different representational paths.

### Setup

Used the same three feature structures from Experiments 063–067:

* `0010|0100` — complementary
* `0010|0111` — partial overlap
* `0010|0010` — redundant

Feature strengths:

`0.001`, `0.01`, `0.1`, `1.0`, `4.0`

Training conditions remained:

* width 2 before injection
* 2 hidden neurons added at epoch 100
* He initialization for the original network
* full-batch training
* learning rate `0.10`
* 4000 total training epochs
* 3900 post-injection updates
* seeds `0–9`
* initialization offsets `[0, 1, 2, 3, 4, 5, 10, 100]`
* zero initial output weights for injected neurons
* success threshold: loss < `1e-6`

There were:

* 10 seeds × 8 initialization offsets = 80 runs per condition
* 15 structure/strength conditions
* 1200 total runs

### Representation Measurements

The hidden representation was measured at post-injection steps:

`0, 1, 5, 20, 100, 500, 1000, 2000, 3900`

At each checkpoint, the experiment measured:

* best possible analytic linear readout loss
* cosine similarity between the injection-time and current hidden activation vectors
* number of hidden features whose binary activation pattern changed
* total binary activation-pattern Hamming distance

The readout measurement asks:

> Could the current hidden representation solve XOR if the output layer were chosen optimally?

The cosine and pattern measurements ask:

> How much has the representation moved away from the representation present at injection?

The representation measurements include the full hidden layer, not only the two injected neurons.

### Outcome Groups

Runs were separated into four categories:

* **S→S** — representation initially sufficient and final training successful
* **S→F** — representation initially sufficient but final training failed
* **I→S** — representation initially insufficient but training constructed a sufficient representation
* **I→F** — representation initially insufficient and final training failed

This separates representational limitations from optimization failures.

---

### Results — Complementary `0010|0100`

All five strength conditions produced:

**80/80 S→S**

There were:

* 0 S→F
* 0 I→S
* 0 I→F

The representation was already linearly sufficient at injection for every run.

The representation nevertheless changed during optimization.

At strength `0.001`:

* mean feature cosine: `1.000000 → 0.948371`
* final mean pattern changes: approximately `1.9`
* final mean pattern Hamming distance: approximately `2.06`

At strength `4.0`:

* mean feature cosine: `1.000000 → 0.997430`
* final mean pattern changes: approximately `0.425`
* final mean pattern Hamming distance: approximately `0.425`

Thus, weak useful features allowed substantially more representational movement, while stronger useful features produced a more stable trajectory.

Despite this movement, all complementary runs retained or recovered a linearly sufficient representation and reached zero loss.

### Interpretation

A successful representation does not have to remain identical.

Optimization can refine a useful representation while preserving enough structure to solve the task.

Feature strength appears to influence how much the representation moves while maintaining success.

---

### Results — Partial-Overlap `0010|0111`

The outcome depended strongly on feature strength.

| Strength | S→S | S→F | I→S | I→F |
| -------: | --: | --: | --: | --: |
|  `0.001` |  48 |  16 |   7 |   9 |
|  `0.010` |  48 |  16 |   0 |  16 |
|  `0.100` |  55 |   9 |   0 |  16 |
|  `1.000` |  56 |   8 |   0 |  16 |
|  `4.000` |  63 |   1 |   0 |  16 |

The most important result is the existence of **S→F** trajectories.

For example, at strength `0.001`, 16 runs began with:

`readout_loss = 0`

but eventually reached:

`readout_loss = 0.25`

while the network failed to solve XOR.

The hidden representation moved substantially during these failures.

At strength `0.001`, the S→F group ended with:

* mean feature cosine ≈ `0.8765`
* approximately `3` hidden features changed binary pattern
* approximately `4` total pattern-bit changes

At strength `4.0`, only one S→F case remained, and its final representation still moved substantially.

### Interpretation

This provides direct evidence that a network can begin with an adequate representation and then move into an inadequate one during ordinary optimization.

Therefore:

> Representation sufficiency at initialization does not guarantee a successful training trajectory.

Optimization can destroy a representation that was already capable of solving the task.

Feature strength reduced the frequency of this failure, making the initially sufficient representation easier for training to preserve or exploit.

---

### Results — Representation Construction

The strongest direct evidence for representation construction occurred in the partial-overlap condition at strength `0.001`.

Seven runs began representation-insufficient:

`readout_loss ≈ 0.321428571`

but eventually became linearly sufficient.

Their trajectory was approximately:

`0.321 → 0.267 → 0.252 → 0.0278 → 0`

The seven successful cases reached representation sufficiency at:

* 1 run by step 1000
* 6 runs by step 2000

Representation change accompanied this transition.

At step 3900:

* mean feature cosine ≈ `0.9039`
* mean pattern changes ≈ `1.0`
* mean Hamming distance ≈ `2.0`

### Interpretation

Training can construct a representation that was not initially capable of solving the task.

This is genuine representational change rather than merely output-layer optimization.

The network first lacked a linearly sufficient hidden representation, then changed its hidden features until a linear readout became sufficient.

---

### Results — Redundant `0010|0010`

The redundant condition showed no S→F cases.

| Strength | S→S | S→F | I→S | I→F |
| -------: | --: | --: | --: | --: |
|  `0.001` |  48 |   0 |   0 |  32 |
|  `0.010` |  48 |   0 |   0 |  32 |
|  `0.100` |  48 |   0 |   7 |  25 |
|  `1.000` |  48 |   0 |   8 |  24 |
|  `4.000` |  48 |   0 |   8 |  24 |

The redundant structure began with a sufficient representation in exactly 48/80 cases at every strength.

At higher strengths, training sometimes constructed a sufficient representation:

* strength `0.1`: 7 I→S
* strength `1.0`: 8 I→S
* strength `4.0`: 8 I→S

However, the remaining insufficient cases stayed at nonzero readout loss.

This matches the structure-dependent ceiling observed in Experiment 066.

### Interpretation

The redundant structure can sometimes be improved by training, but optimization does not consistently create the complementary structure needed for complete rescue.

Training can search for a better representation, but the trajectory may remain trapped in an insufficient structural configuration.

---

### Major Findings

Experiment 068 establishes four important behaviors.

**1. Successful optimization can refine a sufficient representation.**

The complementary `0010|0100` runs changed substantially in some conditions but remained capable of solving XOR.

**2. Optimization can destroy a sufficient representation.**

The partial-overlap S→F cases began with a perfect analytic readout but ended with an insufficient representation and final loss near `0.25`.

**3. Optimization can construct a sufficient representation.**

The partial-overlap and redundant I→S cases began with insufficient representations and eventually developed representations with zero analytic readout loss.

**4. Some trajectories never construct enough structure.**

The I→F cases changed somewhat but settled into nonzero representation loss, consistent with a structure-dependent performance ceiling.

---

### Important Understanding

Experiment 067 established that representation and optimization are distinct.

Experiment 068 shows that the representation itself is dynamic.

The hidden representation is not simply a fixed substrate on which the output layer learns.

During optimization, the network can:

**preserve → refine → damage → reconstruct**

its internal representation.

This leads to a more precise picture:

1. Feature structure determines the representational possibilities available to the network.
2. Feature strength influences how strongly those representations participate in optimization.
3. Optimization moves the hidden representation through parameter space.
4. That movement can improve representational sufficiency.
5. The same movement can also destroy representational sufficiency.
6. Training success therefore depends not only on whether a good representation exists, but on whether optimization follows a trajectory that preserves or reaches it.

### Conclusion

Experiment 068 provides the strongest evidence so far that representation and optimization are coupled dynamically rather than being independent stages.

A network can fail because:

* the representation is initially insufficient,
* optimization fails to construct a sufficient representation,
* or optimization moves an initially sufficient representation into an insufficient one.

The last case is especially important because it demonstrates that optimization itself can create failure even when the required representation is already present.

The complementary structure avoided all of these failure modes in the tested conditions.

### Status

**Experiment 068 complete.**

### Next Direction

Experiment 069 should intervene directly on representation movement.

At injection, split the network into two identical trajectories:

* **normal training:** continue updating hidden and output parameters
* **frozen counterfactual:** preserve the exact injection-time hidden representation and assign the analytically optimal linear output readout

The frozen condition removes output-layer optimization as a confounding factor.

The central question is:

> When an initially sufficient representation exists, does preventing hidden-representation drift prevent the S→F failure?

A successful frozen counterfactual would show that the representation present at injection remained sufficient and that normal hidden-layer movement contributed to the failure.

## Experiment 069 — Frozen Representation

### Question

Does hidden-representation drift cause training failure when a sufficient representation already exists?

Experiment 068 showed that hidden representations are dynamic.

Some trajectories began with representations that were already sufficient to solve XOR but later lost that property.

Other trajectories began with insufficient representations and later constructed sufficient ones.

Experiment 069 directly intervened on representation movement to determine whether this drift contributes causally to optimization failure.

### Methodological Correction

The first version of Experiment 069 froze the hidden layer but still attempted to train the output layer from its original zero initialization.

This introduced a confounding factor.

A frozen representation could have zero analytic readout loss while ordinary gradient descent still failed to discover the required output weights.

Therefore, that version could not cleanly test whether representation drift itself caused failure.

The experiment was corrected.

The corrected frozen condition uses the **analytic best linear output readout available at injection**.

This removes output-layer optimization from the frozen comparison.

### Setup

After epoch 100 and injection of the two new hidden neurons, the exact same network state is copied into two trajectories.

#### Normal trajectory

The hidden and output layers continue ordinary gradient-descent training.

#### Frozen counterfactual

The hidden representation is held exactly at its injection state.

The output layer is immediately assigned the analytically optimal linear readout for that representation.

The frozen branch therefore answers:

> What would happen if the hidden representation present at injection were preserved perfectly and the output layer were given the best possible solution?

The remaining conditions are unchanged:

- width 2 before injection
- 2 hidden neurons added at epoch 100
- He initialization
- XOR training data
- full-batch training
- learning rate `0.10`
- 4000 total training epochs
- 3900 post-injection updates
- seeds `0–9`
- initialization offsets `[0, 1, 2, 3, 4, 5, 10, 100]`
- success threshold: loss < `1e-6`

Feature structures:

- `0010|0100` — complementary
- `0010|0111` — partial overlap
- `0010|0010` — redundant

Feature strengths:

`0.001`, `0.01`, `0.1`, `1.0`, `4.0`

Each condition contained 80 seed/initialization combinations.

### Key Measurement

The most important value is:

**normal_failed → frozen_success**

This means:

- the representation was already sufficient at injection
- normal training failed
- the identical injection representation remained solvable when hidden drift was prevented

This provides a causal test of representation drift.

### Results — Complementary `0010|0100`

All five strengths produced:

| Strength | Initially sufficient | Normal success | Frozen success | Normal failed → frozen success |
|---:|---:|---:|---:|---:|
| 0.001 | 80/80 | 80/80 | 80/80 | 0 |
| 0.010 | 80/80 | 80/80 | 80/80 | 0 |
| 0.100 | 80/80 | 80/80 | 80/80 | 0 |
| 1.000 | 80/80 | 80/80 | 80/80 | 0 |
| 4.000 | 80/80 | 80/80 | 80/80 | 0 |

The complementary representation was already sufficient for every run and remained perfectly usable under the frozen counterfactual.

Normal training also succeeded in every run.

### Results — Partial-Overlap `0010|0111`

The injection representation was sufficient in exactly `64/80` runs at every strength.

The frozen counterfactual succeeded in all of those 64 cases.

Normal training produced fewer successes.

| Strength | Initially sufficient | Normal success | Frozen success | Normal failed → frozen success |
|---:|---:|---:|---:|---:|
| 0.001 | 64/80 | 55/80 | 64/80 | **16** |
| 0.010 | 64/80 | 48/80 | 64/80 | **16** |
| 0.100 | 64/80 | 55/80 | 64/80 | **9** |
| 1.000 | 64/80 | 56/80 | 64/80 | **8** |
| 4.000 | 64/80 | 63/80 | 64/80 | **1** |

Across the five strengths:

- initially sufficient cases: `320`
- normal failures among them: `50`
- frozen failures among them: `0`
- normal failures rescued by freezing: `50`

### Observation

There were **50 cases** in which the representation was already sufficient at injection, normal training failed, and the identical representation remained successful under the frozen counterfactual.

This is the clean causal result Experiment 069 was designed to test.

The effect was strongest at weak feature strengths.

At strength `0.001`, 16 of 64 initially sufficient runs failed under normal training but succeeded under the frozen counterfactual.

At strength `4.0`, only 1 such case remained.

### Interpretation

These results provide direct evidence that hidden-representation drift contributes to optimization failure.

The required representation already existed at injection.

Normal optimization nevertheless moved the hidden representation into a state that could no longer solve the task.

Preventing that movement preserved the solution.

Therefore:

> Some optimization failures are caused not by insufficient representational capacity, but by optimization moving away from an already-sufficient representation.

### Results — Representation Construction

The frozen counterfactual produced zero successful cases from initially insufficient representations.

This is expected because the hidden representation cannot change in the frozen branch.

Normal training, however, successfully constructed new sufficient representations in some initially insufficient cases.

Successful I→S cases included:

- `0010|0111`, strength `0.001`: `7`
- `0010|0010`, strength `0.100`: `7`
- `0010|0010`, strength `1.000`: `8`
- `0010|0010`, strength `4.000`: `8`

Total:

**30 successful trajectories required hidden-representation change.**

### Major Findings

Experiment 069 establishes two opposing roles for representation movement.

**1. Representation movement can be necessary.**

Some initially insufficient representations only became successful because hidden-layer training changed the representation.

**2. Representation movement can be harmful.**

Fifty initially sufficient trajectories failed under normal training but remained solvable when their injection representations were preserved.

Therefore:

> Representation movement is both the mechanism for discovering new solutions and a possible source of losing existing solutions.

### Important Understanding

Experiments 067–069 now form a coherent sequence.

**Experiment 067**

Separated representation capacity from ordinary optimization.

**Experiment 068**

Showed that the hidden representation changes dynamically and can either improve or damage representational sufficiency.

**Experiment 069**

Intervened on that movement and showed that preserving an already-sufficient representation prevents the corresponding normal-training failures.

This leads to a more precise model of learning:

1. Representation determines what solutions are available.
2. Optimization changes the representation.
3. Representation change can construct missing structure.
4. Representation change can destroy useful structure.
5. Feature strength affects how easily useful structure is exploited.
6. Successful learning requires both finding useful representations and navigating representation space without unnecessarily losing them.

### Conclusion

Experiment 069 provides causal evidence that hidden-representation drift contributes to optimization failure.

For the partial-overlap structure, 50 initially sufficient trajectories failed under normal training while the same injection representations remained perfectly solvable under the frozen counterfactual.

At the same time, 30 initially insufficient trajectories succeeded only through normal hidden-representation change.

The resulting picture is:

> **When the representation is insufficient, movement can be necessary.**

> **When the representation is already sufficient, movement can be harmful.**

The problem is therefore no longer simply whether gradient descent can find a useful representation.

The deeper question is:

> **How does optimization determine when to preserve an existing representation and when to change it?**

### Status

**Experiment 069 complete.**

### Next Direction

Experiment 070 should test whether useful representation learning can be preserved while harmful representation drift is reduced.

A natural intervention is to allow normal hidden-layer training while adding a penalty that encourages the hidden representation to remain close to its injection-time state.

This would test whether representation stability can improve optimization reliability without completely freezing representation learning.

## Experiment 070 — Representation Stability

### Question

Can hidden-representation stability reduce harmful representation drift without completely preventing useful representation learning?

Experiment 069 showed that two things can both be true:

* hidden-representation movement can be necessary when the representation is initially insufficient
* hidden-representation movement can be harmful when the representation is already sufficient

Experiment 070 tested whether a stability penalty could discourage harmful movement while still allowing useful movement.

The intervention was applied in activation space rather than parameter space.

### Setup

The experiment used the same three feature structures from Experiments 063–069:

* `0010|0100` — complementary
* `0010|0111` — partial overlap
* `0010|0010` — redundant

Feature strengths:

`0.001`, `0.01`, `0.1`, `1.0`, `4.0`

Representation-stability penalty strengths:

`λ = 0`, `0.001`, `0.01`, `0.1`, `1.0`, `10.0`

Each condition contained:

* 10 seeds
* 8 initialization offsets
* 80 runs per condition

The total sweep contained:

`3 structures × 5 strengths × 6 λ values × 80 runs = 7200 runs`

The penalty was:

`task_loss + λ × mean(0.5 × (hidden_activation - injection_activation)^2)`

The reference representation was the hidden activation state immediately after feature injection.

The interpretation of λ was:

* `λ = 0` — ordinary training
* larger λ — increasingly stronger pressure to remain near the injection-time representation
* very large λ — increasingly similar to freezing the representation

### Important Methodological Limitation

This first 070 run did **not** initialize the output layer to the analytically optimal readout at injection.

The output layer therefore continued learning from its existing state while the hidden representation was simultaneously being constrained.

This introduces a confounding factor:

> A stability penalty may make optimization harder even when the representation itself remains useful.

Therefore these results should not be treated as the final causal test of representation stability.

The experiment is still useful because it shows how the proposed stability objective behaves, but a cleaner rerun is required.

---

### Results — Final Training Success

| Structure | Strength | λ=0 | λ=.001 | λ=.01 | λ=.1 | λ=1 | λ=10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0010|0100` | 0.001 | 80 | 40 | 0 | 0 | 0 | 0 |
| `0010|0100` | 0.01 | 80 | 48 | 0 | 0 | 0 | 0 |
| `0010|0100` | 0.1 | 80 | 66 | 0 | 0 | 0 | 0 |
| `0010|0100` | 1.0 | 80 | 79 | 2 | 10 | 36 | 60 |
| `0010|0100` | 4.0 | 80 | 80 | 69 | 72 | 72 | 72 |
| `0010|0111` | 0.001 | 55 | 16 | 0 | 0 | 0 | 0 |
| `0010|0111` | 0.01 | 48 | 24 | 0 | 0 | 0 | 0 |
| `0010|0111` | 0.1 | 55 | 26 | 0 | 0 | 0 | 0 |
| `0010|0111` | 1.0 | 56 | 53 | 1 | 0 | 14 | 27 |
| `0010|0111` | 4.0 | 63 | 48 | 15 | 15 | 44 | 35 |
| `0010|0010` | 0.001 | 48 | 16 | 0 | 0 | 0 | 0 |
| `0010|0010` | 0.01 | 48 | 24 | 0 | 0 | 0 | 0 |
| `0010|0010` | 0.1 | 55 | 31 | 0 | 0 | 0 | 0 |
| `0010|0010` | 1.0 | 56 | 40 | 1 | 0 | 23 | 23 |
| `0010|0010` | 4.0 | 56 | 44 | 8 | 0 | 32 | 24 |

### Observation

The stability penalty consistently reduced representational movement.

For example, complementary `0010|0100` at strength `0.001` moved from mean feature cosine `0.948371` at `λ=0` to `0.998714` at `λ=10`, while pattern changes fell from `1.900` to `0.225`.

However, greater representational stability did not automatically produce better task optimization.

The same condition went from:

`80/80 success at λ=0`

to:

`0/80 success at λ=10`

The hidden representation became more stable while actual training performance became worse.

### Observation — Initially Sufficient Representations

The stability penalty did not generally rescue initially sufficient representations.

For partial overlap at strength `0.001`:

* λ=0: `55/80`
* λ=0.001: `16/80`
* λ≥0.01: `0/80`

For redundant structure at strength `0.1`:

* λ=0: `55/80`
* λ=0.001: `31/80`
* λ≥0.01: `0/80`

This means that simply preserving the injection representation was not enough to make ordinary training succeed.

### Observation — Initially Insufficient Representations

The penalty also suppressed successful representation construction.

For redundant `0010|0010` at strength `1.0`:

* λ=0: `8` initially insufficient runs became successful
* λ=0.001: `8` remained successful
* λ=0.01: `0`
* λ=0.1: `0`
* λ=1.0: `0`
* λ=10: `0`

For partial overlap, there were no initially insufficient → successful cases for any λ in this run except the λ=0 baseline behavior inherited from ordinary optimization at the weakest strength.

### Interpretation

The proposed stability penalty successfully controlled the variable it was designed to control:

> increasing λ reduced hidden-representation movement.

But reducing movement was not equivalent to improving learning.

The penalty constrained both kinds of movement:

`useful movement`

and

`harmful movement`

The network therefore lost some ability to construct new representations while also not receiving a reliable optimization benefit on representations that were already sufficient.

This suggests:

> **Representation drift is not inherently bad. The important problem is whether the direction of drift is useful or harmful.**

A global penalty cannot distinguish between those two cases.

### Important Understanding

Experiments 067–070 now give a progressively more precise picture.

**Experiment 067**

A representation can be sufficient even when ordinary training fails.

**Experiment 068**

The representation changes during optimization.

**Experiment 069**

Some initially sufficient representations remain solvable when hidden movement is completely prevented, while some initially insufficient cases require hidden movement to succeed.

**Experiment 070**

Simply penalizing all representation movement does not solve the problem.

The important distinction is therefore not:

`movement vs. no movement`

but:

`useful movement vs. harmful movement`

### Conclusion

The first 070 sweep does **not** support the idea that global representation stabilization improves training.

Instead, it shows that the stability penalty introduces a new tradeoff:

* stronger stability produces less representation drift
* stronger stability can also interfere with optimization
* useful representation construction can be suppressed
* preserving representation geometry alone does not guarantee successful gradient-based training

The main lesson is:

> **The goal should not be to keep the representation fixed. The goal should be to prevent harmful representation changes while preserving useful ones.**

### Status

**Experiment 070 exploratory run complete.**

The result is not yet considered the final 070 causal test because the output layer was not initialized to the analytic best readout at injection.

### Next Direction

Run a controlled version of Experiment 070 in which every λ condition receives the same analytically optimal output readout immediately after injection.

This removes output-layer optimization as a confound and isolates the effect of the representation-stability penalty itself.

The key question becomes:

> When the output layer is already optimal for the injection representation, does selectively constraining hidden-representation movement improve or reduce the ability to maintain or discover a solution?

### Controlled Rerun — Analytic Output Readout

The exploratory 070 run revealed an output-layer confound.

The stability penalty was originally applied while the output layer still had to learn from its existing state.

To isolate the effect of representation stability, the experiment was rerun with the output layer initialized immediately after injection to the analytically optimal linear readout for the injection-time hidden representation.

This is the primary controlled result for Experiment 070.

### Controlled Setup

The λ sweep remained:

`0`, `0.001`, `0.01`, `0.1`, `1.0`, `10.0`

Every λ condition began from the same injection state.

The only intervention was the representation-stability penalty:

`task_loss + λ × mean(0.5 × (hidden_activation - injection_activation)^2)`

The output layer was first set to the analytic best readout and then ordinary hidden/output training continued.

This removes output-readout discovery as a confounding factor while testing the stability penalty.

### Controlled Results — Complementary `0010|0100`

| Strength | λ=0 | λ=.001 | λ=.01 | λ=.1 | λ=1 | λ=10 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.001 | 78 | 72 | 72 | 72 | 72 | 72 |
| 0.010 | 77 | 77 | 77 | 77 | 77 | 77 |
| 0.100 | 78 | 74 | 74 | 74 | 74 | 74 |
| 1.000 | 80 | 80 | 80 | 80 | 80 | 80 |
| 4.000 | 80 | 80 | 80 | 80 | 80 | 80 |

Success counts are out of 80 runs.

At strengths `1.0` and `4.0`, the injection representation was so effective that every run remained successful and showed zero measured representation movement.

At weaker strengths, adding stability did not improve success and often reduced it.

### Controlled Results — Partial-Overlap `0010|0111`

| Strength | λ=0 | λ=.001 | λ=.01 | λ=.1 | λ=1 | λ=10 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.001 | 30 | 15 | 15 | 15 | 15 | 15 |
| 0.010 | 22 | 17 | 17 | 17 | 17 | 17 |
| 0.100 | 26 | 21 | 17 | 17 | 17 | 17 |
| 1.000 | 51 | 49 | 45 | 45 | 44 | 35 |
| 4.000 | 57 | 54 | 49 | 48 | 48 | 39 |

No initially insufficient runs became successful in the controlled partial-overlap sweep.

The stability penalty therefore did not create a useful middle ground between representation preservation and representation construction.

### Controlled Results — Redundant `0010|0010`

| Strength | λ=0 | λ=.001 | λ=.01 | λ=.1 | λ=1 | λ=10 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.001 | 32 | 28 | 15 | 15 | 23 | 23 |
| 0.010 | 27 | 27 | 15 | 15 | 23 | 23 |
| 0.100 | 35 | 35 | 15 | 15 | 23 | 23 |
| 1.000 | 40 | 40 | 32 | 32 | 32 | 29 |
| 4.000 | 40 | 40 | 32 | 32 | 32 | 32 |

At strengths `0.001–0.100`, some initially insufficient → successful trajectories occurred only at λ=0 or λ=0.001.

At stronger penalties, those representation-construction trajectories disappeared.

### Controlled Observation

The stability penalty successfully reduced representation movement.

However:

> Less representation movement did not produce more successful learning.

The penalty restricted both useful and harmful hidden movement.

The controlled result therefore rejects the idea that a global representation-stability penalty is sufficient to improve optimization.

### Major New Finding

The most important difference between the exploratory and controlled runs is not the stability penalty.

It is the output-layer initialization.

In the controlled run, the output layer begins at the analytically optimal readout for the injection representation.

Under this condition, initially insufficient representations frequently remain insufficient.

For example, in the controlled partial-overlap sweep, every strength and λ condition reports:

`initially_insufficient: success = 0`

This differs from the earlier ordinary-training behavior, where some insufficient representations became sufficient through hidden-layer movement.

### Interpretation

Output-layer state is therefore not merely a downstream detail.

The output error signal influences whether hidden representations receive pressure to change.

Starting from an output readout that is already optimal for the existing representation can remove or greatly reduce the gradient signal responsible for constructing a new representation.

This means:

> **Representation learning depends on the state of the readout that is driving its gradients.**

The relationship between representation and optimization is therefore even more coupled than Experiment 069 suggested.

The hidden representation is not optimized independently of the output layer.

### Conclusion

The controlled Experiment 070 result is:

> **A global penalty against representation drift does not improve learning reliability.**

The deeper discovery is:

> **Changing the output initialization changes the hidden representation trajectory.**

This creates a new experimental question that is more fundamental than simply finding a better stability coefficient:

> **How does the strength of the output readout at injection control subsequent representation learning?**

### Status

**Experiment 070 complete.**

The exploratory run and controlled rerun are both preserved.

The controlled rerun is the primary result because it removes output-readout discovery as a confound.

### Next Direction

Experiment 071 should vary the amount of analytic output initialization while keeping the hidden injection state identical.

Use interpolation between the original zero output and the analytic best output:

`α = 0, 0.25, 0.50, 0.75, 1.00`

Measure:

* final task success
* initially sufficient → failed cases
* initially insufficient → successful cases
* hidden representation movement
* final analytic readout loss

The central question is:

> **How does output-layer initialization control whether optimization preserves, damages, or constructs the hidden representation?**

## Experiment 071 — Output Initialization

### Question

How does the output-readout state present immediately after feature injection control subsequent hidden-representation learning?

Experiments 069 and 070 showed that the output state can influence whether hidden representation movement occurs.

Experiment 071 varied the output state continuously between:

* the actual output state learned before injection
* the analytically optimal output readout for the post-injection representation

The goal was to determine whether output initialization produces a smooth change in learning behavior.

### Setup

The experiment used the same three feature structures:

* `0010|0100` — complementary
* `0010|0111` — partial overlap
* `0010|0010` — redundant

Feature strengths:

`0.001`, `0.01`, `0.1`, `1.0`, `4.0`

Output interpolation values:

`α = 0`, `0.25`, `0.50`, `0.75`, `1.00`

The interpolation was:

`output(α) = original_output + α × (analytic_output - original_output)`

Therefore:

* `α=0` preserved the actual output state at injection
* `α=1` replaced it with the analytic best readout
* intermediate values interpolated between those two complete output states

Each condition contained:

* 10 seeds
* 8 initialization offsets
* 80 runs

Total:

`15 structure/strength conditions × 5 α values × 80 runs = 6000 runs`

Success threshold:

`loss < 1e-6`

### Important Endpoint Verification

The α=0 endpoint reproduced ordinary post-injection training.

The α=1 endpoint reproduced the analytic-readout initialization condition used in the controlled Experiment 070 run.

This connects 071 directly to Experiments 069 and 070.

---

### Results — Complementary `0010|0100`

The most striking result occurred at weak feature strength.

For strength `0.001`:

| α | Final success |
| ---: | ---: |
| 0.00 | 78/80 |
| 0.25 | 0/80 |
| 0.50 | 0/80 |
| 0.75 | 0/80 |
| 1.00 | 78/80 |

The intermediate states were dramatically worse than either endpoint.

At α values `0.25–0.75`, mean final loss was approximately `0.47–0.50` and mean feature cosine fell to approximately `0.125–0.194`.

The representation therefore moved much farther from the useful injection state.

At strength `1.0`, however, every α condition succeeded:

`80/80`

At strength `4.0`, α=0, 0.25, and 1.0 all reached `80/80`, while α=0.5 and 0.75 reached `72/80` and `75/80`.

### Observation

The output initialization effect is not monotonic.

Intermediate output states can be substantially worse than either endpoint.

---

### Results — Partial-Overlap `0010|0111`

At strength `0.001`:

| α | Final success |
| ---: | ---: |
| 0.00 | 55/80 |
| 0.25 | 0/80 |
| 0.50 | 0/80 |
| 0.75 | 0/80 |
| 1.00 | 27/80 |

The intermediate conditions again produced severe representation degradation.

Mean feature cosine was approximately:

`0.226 → 0.225 → 0.200`

for α values `0.25`, `0.50`, and `0.75`.

Mean pattern Hamming distance reached approximately `7.3–7.4`.

At strength `0.01`:

| α | Final success |
| ---: | ---: |
| 0.00 | 48/80 |
| 0.25 | 7/80 |
| 0.50 | 6/80 |
| 0.75 | 5/80 |
| 1.00 | 26/80 |

At strength `0.1`:

| α | Final success |
| ---: | ---: |
| 0.00 | 55/80 |
| 0.25 | 42/80 |
| 0.50 | 26/80 |
| 0.75 | 17/80 |
| 1.00 | 28/80 |

At strength `1.0` the effect became smaller:

| α | Final success |
| ---: | ---: |
| 0.00 | 56/80 |
| 0.25 | 56/80 |
| 0.50 | 54/80 |
| 0.75 | 54/80 |
| 1.00 | 51/80 |

At strength `4.0`:

| α | Final success |
| ---: | ---: |
| 0.00 | 63/80 |
| 0.25 | 55/80 |
| 0.50 | 55/80 |
| 0.75 | 55/80 |
| 1.00 | 57/80 |

### Interpretation

Weak injected features are especially sensitive to output initialization.

Stronger features reduce the severity of the effect.

However, there is no simple rule that moving closer to the analytic readout improves success.

---

### Results — Redundant `0010|0010`

At strength `0.001`:

| α | Final success |
| ---: | ---: |
| 0.00 | 48/80 |
| 0.25 | 0/80 |
| 0.50 | 0/80 |
| 0.75 | 0/80 |
| 1.00 | 35/80 |

At strength `0.01`:

| α | Final success |
| ---: | ---: |
| 0.00 | 48/80 |
| 0.25 | 1/80 |
| 0.50 | 0/80 |
| 0.75 | 0/80 |
| 1.00 | 28/80 |

At strength `0.1`:

| α | Final success |
| ---: | ---: |
| 0.00 | 55/80 |
| 0.25 | 45/80 |
| 0.50 | 40/80 |
| 0.75 | 40/80 |
| 1.00 | 35/80 |

At strength `1.0`:

| α | Final success |
| ---: | ---: |
| 0.00 | 56/80 |
| 0.25 | 48/80 |
| 0.50 | 46/80 |
| 0.75 | 40/80 |
| 1.00 | 40/80 |

At strength `4.0`:

| α | Final success |
| ---: | ---: |
| 0.00 | 56/80 |
| 0.25 | 48/80 |
| 0.50 | 48/80 |
| 0.75 | 40/80 |
| 1.00 | 40/80 |

### Major Findings

Experiment 071 establishes that output initialization has a strong effect on hidden-representation trajectories.

**1. The effect is not monotonic.**

Intermediate output states can be worse than both endpoints.

**2. Weak features are most sensitive.**

At low feature strengths, partial and redundant structures can collapse almost completely under intermediate output initialization.

**3. Output initialization changes representation movement.**

The catastrophic intermediate cases often showed much lower feature cosine and much larger activation-pattern changes.

**4. The output layer is part of the representation-learning mechanism.**

The output state determines the error signal that is propagated into the hidden layer.

Therefore the output layer cannot be treated as an independent downstream component.

### Important Methodological Limitation

The α interpolation changed the complete output state:

* old hidden-neuron output weights
* newly injected-neuron output weights
* output bias

Therefore the experiment does not yet isolate the effect of coupling the new features to the output layer.

Changing the old output readout may itself disrupt the representation learned before injection.

This is especially important because the original output state already contains information learned during the first 100 epochs.

### Important Understanding

Experiments 069–071 now show:

**Experiment 069**

Preserving a useful representation can prevent some optimization failures.

**Experiment 070**

Penalizing all representation movement is not sufficient because useful movement can also be necessary.

**Experiment 071**

The output state that drives hidden gradients can radically alter the representation trajectory.

The emerging picture is:

> Representation learning depends on the interaction between the hidden state and the output state that generates its gradient signal.

### Conclusion

Experiment 071 shows that output initialization is a major control variable for hidden-representation learning.

However, the complete-output interpolation used here mixes two different effects:

`changing the readout of the old representation`

and

`changing the readout coupling of the newly injected features`

Those effects must be separated before drawing a stronger conclusion.

### Status

**Experiment 071 complete.**

The result is considered a valid exploratory experiment, with the complete-output interpolation limitation documented above.

### Next Direction

Experiment 072 should isolate the newly injected features.

At injection:

* keep the original output weights unchanged
* keep the original output bias unchanged
* vary only the two output weights connected to the newly injected neurons

Set the new output weights to:

`β × analytic_new_feature_weights`

for:

`β = 0`, `0.1`, `0.25`, `0.50`, `0.75`, `1.0`

This tests whether the amount of direct output coupling given to a newly injected feature controls whether that feature participates in representation learning.

The central question becomes:

> **When a new feature is injected, how much output coupling does it need before it can meaningfully influence hidden-layer optimization?**

## Experiment 072 — New-Feature Output Coupling

### Question

How much output coupling should a newly injected hidden feature receive before it meaningfully influences hidden-layer optimization?

Experiment 071 showed that changing the complete output state at injection can radically alter the hidden-representation trajectory.

However, that experiment changed both:

* the learned pre-injection output pathway
* the output coupling of the newly injected features

Experiment 072 isolated the second effect.

### Setup

The three feature structures remained:

* `0010|0100` — complementary
* `0010|0111` — partial overlap
* `0010|0010` — redundant

Feature strengths:

`0.001`, `0.01`, `0.1`, `1.0`, `4.0`

New-feature output coupling:

`β = 0`, `0.1`, `0.25`, `0.50`, `0.75`, `1.0`

At injection:

* the original output bias was preserved
* the original output weights were preserved
* only the two output weights connected to the newly injected neurons were changed

The new-neuron output weights were initialized as:

`β × analytic_new_feature_weights`

Therefore:

* `β=0` — new features enter with zero output coupling
* `β=1` — new features receive their analytic best-readout output-weight components
* intermediate β values provide partial coupling

Each condition contained:

* 10 seeds
* 8 initialization offsets
* 80 runs

Total:

`15 structure/strength conditions × 6 β values × 80 runs = 7200 runs`

Success threshold:

`loss < 1e-6`

### Control Verification

The `β=0` condition preserves the normal post-injection output state.

The β=0 results reproduced the Experiment 069 baseline:

#### Complementary `0010|0100`

`80/80` at every strength.

#### Partial-overlap `0010|0111`

`55/80`, `48/80`, `55/80`, `56/80`, `63/80`

for strengths:

`0.001`, `0.01`, `0.1`, `1.0`, `4.0`

#### Redundant `0010|0010`

`48/80`, `48/80`, `55/80`, `56/80`, `56/80`

for the same strengths.

This confirms that β=0 is a valid ordinary-training control.

---

### Results — Complementary `0010|0100`

| Strength | β=0 | β=.10 | β=.25 | β=.50 | β=.75 | β=1.00 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.001 | 80 | 0 | 0 | 0 | 40 | 40 |
| 0.010 | 80 | 10 | 0 | 3 | 40 | 40 |
| 0.100 | 80 | 77 | 53 | 41 | 40 | 40 |
| 1.000 | 80 | 80 | 80 | 80 | 80 | 80 |
| 4.000 | 80 | 80 | 80 | 80 | 80 | 80 |

The strongest effect occurred for weak injected features.

At strength `0.001`, introducing even modest output coupling caused complete failure at β=`0.10` and `0.25`.

At β=`0.75` and `1.0`, success recovered partially to `40/80`.

At strengths `1.0` and `4.0`, the complementary representation was robust to the new-feature output coupling.

### Observation

The newly injected features do not need direct output coupling to enable successful learning when the complementary representation is strong.

In fact, for weak features, adding output coupling can be strongly destructive.

---

### Results — Partial-Overlap `0010|0111`

| Strength | β=0 | β=.10 | β=.25 | β=.50 | β=.75 | β=1.00 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.001 | 55 | 16 | 16 | 16 | 43 | 43 |
| 0.010 | 48 | 20 | 16 | 16 | 40 | 40 |
| 0.100 | 55 | 49 | 45 | 41 | 39 | 39 |
| 1.000 | 56 | 56 | 56 | 56 | 56 | 56 |
| 4.000 | 63 | 62 | 63 | 61 | 60 | 58 |

At weak strengths, small amounts of coupling sharply reduced success.

At strength `0.001`:

`55/80 → 16/80`

when β increased from `0` to `0.10`.

At strength `0.010`:

`48/80 → 20/80`

under the same change.

At strengths `1.0` and `4.0`, the effect became much smaller.

### Observation

Output coupling of the new features is most dangerous when the injected features are weak.

Strong features can tolerate or dominate the effect.

---

### Results — Redundant `0010|0010`

| Strength | β=0 | β=.10 | β=.25 | β=.50 | β=.75 | β=1.00 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.001 | 48 | 8 | 8 | 16 | 40 | 40 |
| 0.010 | 48 | 33 | 21 | 20 | 40 | 40 |
| 0.100 | 55 | 53 | 45 | 45 | 45 | 45 |
| 1.000 | 56 | 56 | 48 | 48 | 48 | 48 |
| 4.000 | 56 | 56 | 48 | 48 | 48 | 48 |

The redundant representation was also highly sensitive at weak strengths.

At strength `0.001`, success fell from:

`48/80 → 8/80`

at β=`0.10`.

At higher strengths, the effect was smaller but remained visible.

---

### Representation Movement

The output-coupling intervention changed the hidden representation substantially.

For example, complementary `0010|0100` at strength `0.001`:

* β=0 mean feature cosine: `0.948371`
* β=0.10: `0.125000`
* β=0.25: `0.125000`
* β=0.50: `0.175000`
* β=0.75: `0.557037`
* β=1.00: `0.557036`

The corresponding activation-pattern Hamming distance increased sharply for the low-β conditions.

This establishes that changing only the output weights of the newly injected features is sufficient to redirect hidden-representation learning.

### Major Findings

**1. New-feature output coupling is itself a major optimization variable.**

The old learned output weights and bias can remain unchanged while changing the hidden trajectory dramatically.

**2. Zero output coupling can still produce successful learning.**

β=0 reproduces ordinary post-injection training and, in many conditions, produces the highest success.

**3. Small nonzero coupling can be harmful.**

Weak feature strengths were particularly sensitive to β=`0.10–0.50`.

**4. Stronger features reduce sensitivity.**

At feature strengths `1.0` and `4.0`, several conditions became largely insensitive to β.

**5. The effect is not monotonic.**

Increasing β did not simply make performance continuously better or worse.

This mirrors the non-monotonic behavior observed in Experiment 071.

### Important Understanding

Experiments 071 and 072 separate an important part of the mechanism.

Experiment 071 showed that changing the complete output state can dramatically alter learning.

Experiment 072 shows that changing only the new-feature output pathway is enough to reproduce that effect.

Therefore:

> **The initial output coupling of a newly injected feature determines how strongly its representation participates in the subsequent gradient dynamics.**

A newly injected feature is not passive.

Its output connection immediately determines how much task-error signal is associated with that feature and therefore how strongly its hidden weights are driven.

### Interpretation

The results suggest that feature injection has two distinct components:

`feature representation`

and

`feature output coupling`

The representation determines what information the feature provides.

The output coupling determines how strongly that feature participates in the task-loss gradient.

This creates a feedback loop:

`feature → output contribution → output error → hidden gradient → feature change`

Experiment 072 shows that controlling the initial strength of this loop can dramatically change the final representation.

### Conclusion

Experiment 072 provides strong evidence that the newly injected feature's output coupling is an important control variable for representation learning.

The best-performing condition is not universally the strongest or weakest coupling.

Instead, the effect depends on:

* feature structure
* feature strength
* current hidden representation
* output coupling

The main lesson is:

> **A feature's usefulness is determined not only by the representation it creates, but also by how strongly that representation is connected to the task output.**

### Status

**Experiment 072 complete.**

### Next Direction

Experiment 073 should separate the two components of the new-feature gradient:

1. the gradient magnitude created by output coupling
2. the direction of that gradient determined by the feature structure

A controlled experiment should compare positive, zero, and reversed output coupling for the same injected feature.

This will test whether the destructive behavior comes mainly from:

`too much gradient`

or from:

`gradient in the wrong direction`

The central question becomes:

> **Is harmful representation drift caused by the magnitude of the new-feature gradient, or by its directional alignment with the task?**

## Experiment 073 — Gradient Direction

### Question

Is harmful representation drift caused by gradient magnitude or gradient direction?

### Setup

Experiment 073 used the same post-injection network state across all gamma conditions.

The two newly injected hidden features received output coupling:

- gamma = -1.0, -0.5, 0.0, +0.5, +1.0
- gamma > 0 follows the analytic new-feature output direction
- gamma < 0 reverses that direction
- gamma = 0 gives the new features zero output coupling

The original learned output weights and bias were left unchanged.

Each condition used 10 seeds x 8 initialization offsets = 80 runs.

### Results

The sign of the new-feature output coupling had a large effect even when the absolute coupling magnitude was identical.

Examples:

- Complementary `0010|0100`, strength 0.001:
  - gamma = -0.50: 40/80 successful
  - gamma = +0.50: 0/80 successful
  - gamma = 0.00: 80/80 successful

- Complementary `0010|0100`, strength 0.010:
  - gamma = -0.50: 40/80
  - gamma = +0.50: 3/80
  - gamma = 0.00: 80/80

- Partial `0010|0111`, strength 0.001:
  - gamma = -0.50: 43/80
  - gamma = +0.50: 16/80
  - gamma = 0.00: 55/80

- Redundant `0010|0010`, strength 0.001:
  - gamma = -0.50: 40/80
  - gamma = +0.50: 16/80
  - gamma = 0.00: 48/80

At strong complementary feature strength (4.0), the direction effect largely disappeared because the representation was already highly stable. The complementary structure reached 77/80 or better for every gamma condition.

### Interpretation

The results show that harmful representation drift is substantially influenced by gradient direction, not only gradient magnitude.

Equal-magnitude positive and negative coupling produced very different representation movement and final success rates. The effect was strongest when the injected features were weak.

This also reinforces the earlier result that output-layer coupling is an important control variable for hidden representation learning.

The evidence does not yet directly measure the hidden gradient vectors themselves. Experiment 073 changes the sign of the output coupling and observes the resulting behavior, so the next step is to measure the actual hidden-gradient direction at injection.

### Next Direction

Measure the hidden-gradient vectors produced by positive and negative gamma at the injection step and compare their cosine similarity and magnitude.

The goal is to connect:

`gamma sign -> hidden gradient direction -> representation movement -> final success`

## Experiment 074 — Hidden Gradient Measurement

### Results

Experiment 074 directly measured the batch-averaged gradient vectors for the two newly injected hidden neurons immediately after injection.

For the complementary `0010|0100` structure, equal-magnitude gamma conditions produced substantially different gradient magnitudes:

- strength 0.001:
  - gamma = -1.00: 1373.07
  - gamma = -0.50: 460.74
  - gamma = +0.50: 48.56
  - gamma = +1.00: 456.03
- gamma = 0 produced exactly zero hidden gradient for the injected neurons.

The direction was not a simple sign reversal.

For gamma = -0.50 versus +0.50:

- mean hidden-gradient cosine = -0.1383

For gamma = -1.00 versus +1.00:

- mean hidden-gradient cosine = 0.9703

This means increasing the magnitude of the signed output coupling can cause the positive and negative conditions to become aligned again, even though their output couplings have opposite signs.

The partial structure `0010|0111` showed the same general effect but with weaker alignment:

- gamma = -0.50 vs +0.50: cosine = -0.0737
- gamma = -1.00 vs +1.00: cosine = 0.7614

The redundant `0010|0010` structure was especially unstable and produced extremely large gradients because both injected features represented the same structure.

### Interpretation

The result shows that changing output coupling does more than reverse the hidden-gradient direction.

For a new hidden feature with output coupling `gamma`, the output prediction can be written conceptually as:

`prediction = old_prediction + gamma * new_feature_contribution`

Therefore the hidden gradient contains the output error produced by both the old network and the newly coupled feature.

The gradient therefore has the form:

`g(gamma) = gamma * A + gamma² * B`

where:

- `A` is the gradient driven by the pre-existing output error
- `B` is the gradient created by the injected feature's own contribution to the output

The odd component changes sign with gamma.

The even component does not.

This explains why small positive and negative gamma values can produce opposing hidden gradients, while larger values can become aligned.

### Conclusion

Experiment 074 directly connects output coupling to the hidden gradient itself.

The mechanism is more specific than:

`gamma sign -> gradient sign`

The evidence instead suggests:

`gamma -> output contribution -> output error -> hidden gradient`

with both a sign-sensitive component and a self-induced component.

### Next Direction

Separate the hidden gradient into its odd and even components using matched `+gamma` and `-gamma` measurements.

The goal is to test whether:

`[g(+gamma) - g(-gamma)] / 2`

isolates the external-error-driven component, while:

`[g(+gamma) + g(-gamma)] / 2`

isolates the self-induced component.

## Experiment 075 — Gradient Decomposition

### Question

Can the hidden gradient be decomposed into an external-error component and a self-induced component?

### Hypothesis

For matched positive and negative output coupling:

`g(gamma) = gamma*A + gamma²*B`

Therefore:

`odd component = [g(+gamma) - g(-gamma)] / 2`

and:

`even component = [g(+gamma) + g(-gamma)] / 2`

The experiment should compare these components across gamma magnitude and feature structure.

