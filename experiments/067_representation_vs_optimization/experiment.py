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

feature_patterns = {
    "0010|0100": (
        ([1.0, -2.0], 0.0),
        ([-2.0, 1.0], 0.0),
    ),
    "0010|0111": (
        ([1.0, -2.0], 0.0),
        ([1.0, 1.0], -0.5),
    ),
    "0010|0010": (
        ([1.0, -2.0], 0.0),
        ([1.0, -2.0], 0.0),
    ),
}

feature_strengths = [
    0.001,
    0.01,
    0.10,
    1.00,
    4.00,
]

training_budgets = [
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


def solve_linear_system(matrix, vector):
    size = len(vector)
    augmented = [
        list(matrix[row]) + [vector[row]]
        for row in range(size)
    ]

    for column in range(size):
        pivot_row = max(
            range(column, size),
            key=lambda row: abs(augmented[row][column]),
        )

        pivot = augmented[pivot_row][column]

        if abs(pivot) < 1e-12:
            return None

        augmented[column], augmented[pivot_row] = (
            augmented[pivot_row],
            augmented[column],
        )

        pivot = augmented[column][column]

        for index in range(column, size + 1):
            augmented[column][index] /= pivot

        for row in range(size):
            if row == column:
                continue

            factor = augmented[row][column]

            if factor == 0.0:
                continue

            for index in range(column, size + 1):
                augmented[row][index] -= (
                    factor * augmented[column][index]
                )

    return [
        augmented[row][size]
        for row in range(size)
    ]


def hidden_representation(network):
    values = [
        []
        for _ in network.layers[0].neurons
    ]

    for inputs, _ in training_data:
        for neuron_index, neuron in enumerate(
            network.layers[0].neurons
        ):
            _, activated = neuron.forward(inputs)
            values[neuron_index].append(activated)

    return values


def best_linear_readout_loss(
    hidden_values,
):
    import itertools

    target_values = [
        target
        for _, target in training_data
    ]

    columns = [
        [1.0 for _ in training_data]
    ] + hidden_values

    best_loss = float("inf")

    for size in range(
        1,
        min(len(columns), len(training_data)) + 1,
    ):
        for selected in itertools.combinations(
            range(len(columns)),
            size,
        ):
            selected_columns = [
                columns[index]
                for index in selected
            ]

            normalized_columns = []

            valid = True

            for column in selected_columns:
                norm = math.sqrt(
                    sum(
                        value * value
                        for value in column
                    )
                )

                if norm < 1e-14:
                    valid = False
                    break

                normalized_columns.append(
                    [
                        value / norm
                        for value in column
                    ]
                )

            if not valid:
                continue

            matrix = [
                [
                    sum(
                        normalized_columns[left][row]
                        * normalized_columns[right][row]
                        for row in range(
                            len(training_data)
                        )
                    )
                    for right in range(size)
                ]
                for left in range(size)
            ]

            vector = [
                sum(
                    normalized_columns[column][row]
                    * target_values[row]
                    for row in range(
                        len(training_data)
                    )
                )
                for column in range(size)
            ]

            weights = solve_linear_system(
                matrix,
                vector,
            )

            if weights is None:
                continue

            total = 0.0

            for row in range(len(training_data)):
                prediction = sum(
                    weights[column]
                    * normalized_columns[column][row]
                    for column in range(size)
                )

                error = prediction - target_values[row]
                total += 0.5 * error ** 2

            best_loss = min(
                best_loss,
                total,
            )

    return best_loss


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
    pattern_name,
    target_norm,
    strength,
):
    effective_norm = target_norm * strength

    raw_first, raw_second = feature_patterns[pattern_name]

    first = scaled_feature(
        raw_first[0],
        raw_first[1],
        effective_norm,
    )

    second = scaled_feature(
        raw_second[0],
        raw_second[1],
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
        max(first_values[1], second_values[1]),
        max(first_values[2], second_values[2]),
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
    pattern_name,
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
        pattern_name,
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

    representation = hidden_representation(
        network,
    )

    injection_readout_loss = best_linear_readout_loss(
        representation,
    )

    for post_injection_step in range(
        1,
        max(training_budgets) + 1,
    ):
        batch_epoch(network)

    final_loss = loss(network)

    return (
        injection_metrics,
        injection_readout_loss,
        final_loss,
    )


all_results = {}

print()
print("=" * 75)
print("EXPERIMENT 067 — REPRESENTATION VS. OPTIMIZATION")
print("Question: is failure caused by representation or optimization?")
print()
print("At injection, the four hidden features are held fixed analytically")
print("while finding the best possible linear output readout.")
print("This gives an instantaneous representation-capacity measurement.")
print()
print("The same run is then trained for 3900 post-injection updates.")
print("This lets me compare initial linear sufficiency with actual training.")
print()
print("10 seeds x 8 initialization offsets = 80 runs/condition")
print("Success = loss < 1e-6.")
print("=" * 75)

for pattern_name in feature_patterns:
    all_results[pattern_name] = {}

    for strength in feature_strengths:
        injection_readout_losses = []
        final_losses = []

        injection_readout_successes = 0
        final_successes = 0

        initially_sufficient_but_failed = 0
        initially_insufficient_but_final_success = 0

        for offset in initialization_offsets:
            for seed in seeds:
                (
                    injection_metrics,
                    injection_readout_loss,
                    final_loss,
                ) = run(
                    pattern_name,
                    strength,
                    seed,
                    offset,
                )

                injection_readout_losses.append(
                    injection_readout_loss
                )

                final_losses.append(
                    final_loss
                )

                initially_sufficient = (
                    injection_readout_loss
                    < success_threshold
                )

                final_success = (
                    final_loss
                    < success_threshold
                )

                if initially_sufficient:
                    injection_readout_successes += 1

                if final_success:
                    final_successes += 1

                if (
                    initially_sufficient
                    and not final_success
                ):
                    initially_sufficient_but_failed += 1

                if (
                    not initially_sufficient
                    and final_success
                ):
                    initially_insufficient_but_final_success += 1

        all_results[pattern_name][strength] = {
            "mean_injection_readout_loss": (
                sum(injection_readout_losses)
                / len(injection_readout_losses)
            ),
            "mean_final_loss": (
                sum(final_losses)
                / len(final_losses)
            ),
            "injection_readout_successes": (
                injection_readout_successes
            ),
            "final_successes": final_successes,
            "initially_sufficient_but_failed": (
                initially_sufficient_but_failed
            ),
            "initially_insufficient_but_final_success": (
                initially_insufficient_but_final_success
            ),
        }

        result = all_results[pattern_name][strength]

        print(
            f"pattern={pattern_name} "
            f"strength={strength:.3f} "
            f"readout={result['injection_readout_successes']}/80 "
            f"final={result['final_successes']}/80 "
            f"readout_mean_loss="
            f"{result['mean_injection_readout_loss']:.9f} "
            f"final_mean_loss="
            f"{result['mean_final_loss']:.9f}"
        )


print()
print("=" * 75)
print("EXPERIMENT 067 SUMMARY")
print("=" * 75)

print(
    "Representation capacity versus trained optimization:"
)

print()
print(
    "pattern          strength   "
    "readout   final   "
    "readout->fail   new_solution   "
    "readout_mean_loss   final_mean_loss"
)

for pattern_name in feature_patterns:
    for strength in feature_strengths:
        result = all_results[pattern_name][strength]

        print(
            f"{pattern_name:<16}"
            f"{strength:>8.3f}   "
            f"{result['injection_readout_successes']:>3d}/80   "
            f"{result['final_successes']:>3d}/80   "
            f"{result['initially_sufficient_but_failed']:>7d}        "
            f"{result['initially_insufficient_but_final_success']:>7d}        "
            f"{result['mean_injection_readout_loss']:>16.9f}   "
            f"{result['mean_final_loss']:>15.9f}"
        )

print()
print(
    "Definitions:"
)
print(
    "readout = runs where the fixed injection representation "
    "can already fit XOR with a linear output."
)
print(
    "readout->fail = initially sufficient representation, "
    "but training did not reach success."
)
print(
    "new_solution = representation was initially insufficient, "
    "but training changed the hidden representation enough to succeed."
)
