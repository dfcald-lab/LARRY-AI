import math

from src.network import Network
from src.neuron import Neuron


training_data = [
    ([0.0, 0.0], 0.0),
    ([0.0, 1.0], 1.0),
    ([1.0, 0.0], 1.0),
    ([1.0, 1.0], 0.0),
]


learning_rate = 0.10
epochs = 4000
injection_epoch = 100

seeds = range(10)
initialization_offsets = [0, 1, 2, 3, 4, 5, 10, 100]

extra_width = 2
success_threshold = 1e-6


def apply_he_initialization(network, seed):
    import random

    rng = random.Random(seed)

    for layer in network.layers:
        if layer.neurons[0].activation != "relu":
            continue

        fan_in = len(layer.neurons[0].weights)
        std = math.sqrt(2.0 / fan_in)

        for neuron in layer.neurons:
            neuron.weights = [
                rng.gauss(0.0, std)
                for _ in neuron.weights
            ]
            neuron.bias = 0.0


def make_width2_network(seed):
    network = Network(
        number_of_inputs=2,
        layer_sizes=[2, 1],
        activations=["relu", "linear"],
        seed=seed,
    )

    apply_he_initialization(network, seed)

    return network


def batch_gradients(network):
    accumulated = []

    for layer in network.layers:
        accumulated.append(
            [
                {
                    "weights": [0.0 for _ in neuron.weights],
                    "bias": 0.0,
                }
                for neuron in layer.neurons
            ]
        )

    for inputs, target in training_data:
        prediction = network.forward(inputs)[0]

        network.backward([prediction - target])

        for layer_index, layer in enumerate(network.layers):
            for neuron_index, neuron in enumerate(layer.neurons):
                for weight_index, gradient in enumerate(
                    neuron.weight_gradients
                ):
                    accumulated[layer_index][neuron_index]["weights"][
                        weight_index
                    ] += gradient

                accumulated[layer_index][neuron_index]["bias"] += (
                    neuron.bias_gradient
                )

    return accumulated


def batch_epoch(network):
    gradients = batch_gradients(network)
    batch_size = len(training_data)

    for layer_index, layer in enumerate(network.layers):
        for neuron_index, neuron in enumerate(layer.neurons):
            neuron.weight_gradients = [
                gradient / batch_size
                for gradient in gradients[layer_index][neuron_index]["weights"]
            ]

            neuron.bias_gradient = (
                gradients[layer_index][neuron_index]["bias"]
                / batch_size
            )

    network.update(learning_rate)


def loss(network):
    total = 0.0

    for inputs, target in training_data:
        prediction = network.forward(inputs)[0]
        error = prediction - target
        total += 0.5 * error ** 2

    return total


def standard_initialization_norm(
    seed,
    initialization_offset,
):
    import random

    rng = random.Random(
        seed + 10000 + initialization_offset
    )

    std = math.sqrt(2.0 / 2.0)

    first = [
        rng.gauss(0.0, std)
        for _ in range(2)
    ]

    second = [
        rng.gauss(0.0, std)
        for _ in range(2)
    ]

    first_norm = math.sqrt(
        sum(value * value for value in first)
    )

    second_norm = math.sqrt(
        sum(value * value for value in second)
    )

    return (first_norm + second_norm) / 2.0


def scaled_feature(
    weights,
    bias,
    target_norm,
):
    norm = math.sqrt(
        sum(value * value for value in weights)
    )

    if norm == 0.0:
        raise ValueError(
            "Feature weight vector cannot have zero norm."
        )

    scale = target_norm / norm

    return (
        [
            value * scale
            for value in weights
        ],
        bias * scale,
    )


def construct_feature_pair(
    condition,
    target_norm,
):
    # The first injected feature is fixed at 0010.
    # The second injected feature cycles through every non-XOR
    # activation pattern realizable by a single affine + ReLU unit
    # on the four XOR inputs.
    feature_definitions = {
        # second feature: 0000
        "0000": (([-1.0, -1.0], 0.0),),

        # second feature: 0001
        "0001": (([1.0, 1.0], -1.0),),

        # second feature: 0010
        "0010": (([1.0, -2.0], 0.0),),

        # second feature: 0011
        "0011": (([1.0, 0.0], 0.0),),

        # second feature: 0100
        "0100": (([-2.0, 1.0], 0.0),),

        # second feature: 0101
        "0101": (([-1.0, 2.0], 0.0),),

        # second feature: 0111
        "0111": (([1.0, 1.0], 0.0),),

        # second feature: 1000
        "1000": (([-2.0, -2.0], 1.0),),

        # second feature: 1010
        "1010": (([-1.0, -2.0], 2.0),),

        # second feature: 1011
        "1011": (([1.0, -2.0], 2.0),),

        # second feature: 1100
        "1100": (([-2.0, -1.0], 2.0),),

        # second feature: 1101
        "1101": (([-2.0, 1.0], 2.0),),

        # second feature: 1110
        "1110": (([-1.0, -1.0], 2.0),),

        # second feature: 1111
        "1111": (([-1.0, 0.0], 2.0),),
    }

    if condition not in feature_definitions:
        raise ValueError(
            f"Unknown condition: {condition}"
        )

    first = scaled_feature(
        [1.0, -2.0],
        0.0,
        target_norm,
    )

    second_definition = feature_definitions[condition][0]

    second = scaled_feature(
        second_definition[0],
        second_definition[1],
        target_norm,
    )

    return [first, second]


def append_new_neurons(
    network,
    features,
):
    hidden_layer = network.layers[0]
    output_neuron = network.layers[1].neurons[0]

    for weights, bias in features:
        hidden_layer.neurons.append(
            Neuron(
                weights=list(weights),
                bias=bias,
                activation="relu",
            )
        )

    output_neuron.weights.extend(
        [0.0] * extra_width
    )

    output_neuron.weight_gradients.extend(
        [0.0] * extra_width
    )


def activation_values(
    network,
    neuron,
):
    values = []

    for inputs, _ in training_data:
        network.forward(inputs)

        values.append(
            neuron.activate(neuron.raw_output)
        )

    return values


def binary_pattern(values):
    return "".join(
        "1" if value > 1e-12 else "0"
        for value in values
    )


def pattern_metrics(
    first_values,
    second_values,
):
    first_pattern = binary_pattern(first_values)
    second_pattern = binary_pattern(second_values)

    first_active = {
        index
        for index, value in enumerate(first_values)
        if value > 1e-12
    }

    second_active = {
        index
        for index, value in enumerate(second_values)
        if value > 1e-12
    }

    overlap = len(
        first_active & second_active
    )

    union = len(
        first_active | second_active
    )

    jaccard = (
        overlap / union
        if union
        else 0.0
    )

    total_active = (
        len(first_active)
        + len(second_active)
    )

    positive_examples = {1, 2}
    negative_examples = {0, 3}

    union_active = first_active | second_active

    return {
        "patterns": (
            f"{first_pattern}|{second_pattern}"
        ),
        "first_active": len(first_active),
        "second_active": len(second_active),
        "overlap": overlap,
        "union": union,
        "jaccard": jaccard,
        "total_active": total_active,
        "positive_coverage": len(union_active & positive_examples),
        "negative_coverage": len(union_active & negative_examples),
    }


def run(
    condition,
    seed,
    initialization_offset,
):
    network = make_width2_network(seed)

    for _ in range(injection_epoch):
        batch_epoch(network)

    target_norm = standard_initialization_norm(
        seed,
        initialization_offset,
    )

    features = construct_feature_pair(
        condition,
        target_norm,
    )

    append_new_neurons(
        network,
        features,
    )

    hidden_neurons = network.layers[0].neurons[-extra_width:]

    first_values = activation_values(
        network,
        hidden_neurons[0],
    )

    second_values = activation_values(
        network,
        hidden_neurons[1],
    )

    injection_metrics = pattern_metrics(
        first_values,
        second_values,
    )

    for _ in range(
        epochs - injection_epoch
    ):
        batch_epoch(network)

    final_loss = loss(network)
    success = final_loss < success_threshold

    return (
        injection_metrics,
        final_loss,
        success,
    )


conditions = [
    "0000",
    "0001",
    "0010",
    "0011",
    "0100",
    "0101",
    "0111",
    "1000",
    "1010",
    "1011",
    "1100",
    "1101",
    "1110",
    "1111",
]


print("\n" + "=" * 75)
print("EXPERIMENT 062 — TASK COVERAGE MATRIX")
print("Question: which task examples must new features cover")
print("for rescue to become reliable?")
print()
print("Width 2 -> add 2 hidden neurons at epoch 100")
print("Learning rate = 0.10")
print("4000 total epochs")
print("10 seeds x 8 initialization offsets = 80 runs/condition")
print("New output weights start at zero.")
print("First feature fixed at 0010; second feature sweeps all realizable patterns.")
print("=" * 75)


all_results = {}

for condition in conditions:
    successful_count = 0
    losses = []
    first_metrics = None

    print("\n" + "-" * 75)
    print(condition)

    for offset in initialization_offsets:
        offset_successes = 0

        for seed in seeds:
            (
                injection_metrics,
                final_loss,
                success,
            ) = run(
                condition,
                seed,
                offset,
            )

            if first_metrics is None:
                first_metrics = injection_metrics

            losses.append(final_loss)

            if success:
                successful_count += 1
                offset_successes += 1

        print(
            f"offset={offset:3d} "
            f"successful={offset_successes}/10"
        )

    mean_loss = (
        sum(losses) / len(losses)
    )

    all_results[condition] = {
        "successes": successful_count,
        "mean_loss": mean_loss,
        "metrics": first_metrics,
    }

    metrics = first_metrics

    print(
        f"\npattern={metrics['patterns']} "
        f"first_active={metrics['first_active']} "
        f"second_active={metrics['second_active']} "
        f"overlap={metrics['overlap']} "
        f"union={metrics['union']} "
        f"jaccard={metrics['jaccard']:.3f} "
        f"positive={metrics['positive_coverage']} "
        f"negative={metrics['negative_coverage']}"
    )

    print(
        f"summary {condition}: "
        f"successful={successful_count}/80 "
        f"mean_loss={mean_loss:.12f}"
    )


print("\n" + "=" * 75)
print("EXPERIMENT 062 SUMMARY")
print("=" * 75)

for condition in conditions:
    result = all_results[condition]
    metrics = result["metrics"]

    print(
        f"{condition:24s} "
        f"pattern={metrics['patterns']:9s} "
        f"overlap={metrics['overlap']} "
        f"union={metrics['union']} "
        f"jaccard={metrics['jaccard']:.3f} "
        f"active={metrics['total_active']} "
        f"positive={metrics['positive_coverage']} "
        f"negative={metrics['negative_coverage']} "
        f"successful={result['successes']:2d}/80 "
        f"mean_loss={result['mean_loss']:.12f}"
    )
