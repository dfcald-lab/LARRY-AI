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

feature_strengths = [
    0.00000001,
    0.0000001,
    0.000001,
    0.00001,
    0.0001,
    0.001,
    0.01,
    0.10,
    1.00,
    4.00,
]

training_budgets = [
    100,
    250,
    500,
    1000,
    2000,
    3900,
]

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
    target_norm,
    strength,
):
    effective_norm = target_norm * strength

    first = scaled_feature(
        [1.0, -2.0],
        0.0,
        effective_norm,
    )

    second = scaled_feature(
        [-2.0, 1.0],
        0.0,
        effective_norm,
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

    positive_activations = [
        first_values[2],
        second_values[1],
    ]

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
        "positive_mean_activation": (
            sum(positive_activations)
            / len(positive_activations)
        ),
        "positive_min_activation": min(
            positive_activations
        ),
        "positive_max_activation": max(
            positive_activations
        ),
    }


def run(
    strength,
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
        target_norm,
        strength,
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

    injection_loss = loss(network)

    checkpoint_losses = {}

    for post_injection_step in range(
        1,
        max(training_budgets) + 1,
    ):
        batch_epoch(network)

        if post_injection_step in training_budgets:
            checkpoint_losses[post_injection_step] = loss(network)

    return (
        injection_metrics,
        injection_loss,
        checkpoint_losses,
    )


all_results = {}

print("\n" + "=" * 75)
print("EXPERIMENT 065 — STRENGTH × TRAINING BUDGET")
print("Question: can additional training time compensate for weak features?")
print()
print("Width 2 -> add 2 hidden neurons at epoch 100")
print("Learning rate = 0.10")
print("10 seeds x 8 initialization offsets = 80 runs/strength")
print("New output weights start at zero.")
print("Feature pattern is fixed at 0010|0100.")
print("Only feature strength and training budget vary.")
print()
print("Success = loss < 1e-6 at the selected budget.")
print("=" * 75)


for strength in feature_strengths:
    successes = {
        budget: 0
        for budget in training_budgets
    }

    losses = {
        budget: []
        for budget in training_budgets
    }

    first_metrics = None

    for offset in initialization_offsets:
        offset_successes = {
            budget: 0
            for budget in training_budgets
        }

        for seed in seeds:
            (
                injection_metrics,
                injection_loss,
                checkpoint_losses,
            ) = run(
                strength,
                seed,
                offset,
            )

            if first_metrics is None:
                first_metrics = injection_metrics

            for budget in training_budgets:
                checkpoint_loss = checkpoint_losses[budget]

                losses[budget].append(
                    checkpoint_loss
                )

                if checkpoint_loss < success_threshold:
                    successes[budget] += 1
                    offset_successes[budget] += 1

        print(
            f"strength={strength:.8f} "
            f"offset={offset:3d} "
            + " ".join(
                f"b{budget}={offset_successes[budget]}/10"
                for budget in training_budgets
            )
        )

    all_results[strength] = {
        "metrics": first_metrics,
        "successes": successes,
        "mean_losses": {
            budget: (
                sum(losses[budget])
                / len(losses[budget])
            )
            for budget in training_budgets
        },
    }

    print(
        f"summary strength={strength:.8f} "
        f"pattern={first_metrics['patterns']} "
        f"positive_mean="
        f"{first_metrics['positive_mean_activation']:.12f}"
    )

    for budget in training_budgets:
        print(
            f"  budget={budget:4d} "
            f"successful={successes[budget]}/80 "
            f"mean_loss="
            f"{all_results[strength]['mean_losses'][budget]:.12f}"
        )


print("\n" + "=" * 75)
print("EXPERIMENT 065 SUMMARY")
print("=" * 75)

print("Success counts by feature strength and training budget:")

print(
    "strength".ljust(14)
    + "".join(
        f"{budget:>12d}"
        for budget in training_budgets
    )
)

for strength in feature_strengths:
    row = f"{strength:<14.8f}"

    for budget in training_budgets:
        row += (
            f"{all_results[strength]['successes'][budget]:>9d}/80"
            + "   "
        )

    print(row)

print("\nMean loss by feature strength and training budget:")

print(
    "strength".ljust(14)
    + "".join(
        f"{budget:>12d}"
        for budget in training_budgets
    )
)

for strength in feature_strengths:
    row = f"{strength:<14.8f}"

    for budget in training_budgets:
        row += (
            f"{all_results[strength]['mean_losses'][budget]:>12.9f}"
        )

    print(row)
