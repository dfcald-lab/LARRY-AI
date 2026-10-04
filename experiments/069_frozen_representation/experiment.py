import copy
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



def cosine_similarity(first, second):
    first_norm = math.sqrt(
        sum(value * value for value in first)
    )

    second_norm = math.sqrt(
        sum(value * value for value in second)
    )

    if first_norm < 1e-14 and second_norm < 1e-14:
        return 1.0

    if first_norm < 1e-14 or second_norm < 1e-14:
        return 0.0

    dot_product = sum(
        first[index] * second[index]
        for index in range(len(first))
    )

    return dot_product / (
        first_norm * second_norm
    )


def representation_change(
    reference,
    current,
):
    cosines = []
    pattern_changes = 0
    pattern_hamming = 0

    for reference_values, current_values in zip(
        reference,
        current,
    ):
        cosines.append(
            cosine_similarity(
                reference_values,
                current_values,
            )
        )

        reference_pattern = binary_pattern(
            reference_values
        )

        current_pattern = binary_pattern(
            current_values
        )

        if reference_pattern != current_pattern:
            pattern_changes += 1

        pattern_hamming += sum(
            left != right
            for left, right in zip(
                reference_pattern,
                current_pattern,
            )
        )

    return {
        "mean_feature_cosine": (
            sum(cosines) / len(cosines)
        ),
        "min_feature_cosine": min(cosines),
        "pattern_changes": pattern_changes,
        "pattern_hamming": pattern_hamming,
    }


def best_linear_readout_parameters(
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
    best_bias = 0.0
    best_weights = [
        0.0
        for _ in hidden_values
    ]

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
            norms = []

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

                norms.append(norm)

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

            solved = solve_linear_system(
                matrix,
                vector,
            )

            if solved is None:
                continue

            total = 0.0

            for row in range(len(training_data)):
                prediction = sum(
                    solved[column]
                    * normalized_columns[column][row]
                    for column in range(size)
                )

                error = prediction - target_values[row]
                total += 0.5 * error ** 2

            if total >= best_loss:
                continue

            best_loss = total

            candidate_bias = 0.0
            candidate_weights = [
                0.0
                for _ in hidden_values
            ]

            for local_index, column_index in enumerate(
                selected
            ):
                coefficient = (
                    solved[local_index]
                    / norms[local_index]
                )

                if column_index == 0:
                    candidate_bias = coefficient
                else:
                    candidate_weights[
                        column_index - 1
                    ] = coefficient

            best_bias = candidate_bias
            best_weights = candidate_weights

    return (
        best_loss,
        best_bias,
        best_weights,
    )


def set_output_readout(
    network,
    bias,
    weights,
):
    output_neuron = network.layers[-1].neurons[0]

    output_neuron.bias = bias
    output_neuron.weights = list(weights)

    output_neuron.weight_gradients = [
        0.0
        for _ in weights
    ]

    output_neuron.bias_gradient = 0.0


def output_only_epoch(network):
    gradients = batch_gradients(network)
    batch_size = len(training_data)

    output_layer = network.layers[-1]

    for neuron_index, neuron in enumerate(output_layer.neurons):
        neuron.weight_gradients = [
            gradient / batch_size
            for gradient in gradients[
                -1
            ][neuron_index]["weights"]
        ]

        neuron.bias_gradient = (
            gradients[-1][neuron_index]["bias"]
            / batch_size
        )

    for neuron in output_layer.neurons:
        neuron.update(learning_rate)


training_budgets = [3900]


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

    injection_representation = hidden_representation(
        network,
    )

    (
        injection_readout_loss,
        injection_bias,
        injection_weights,
    ) = best_linear_readout_parameters(
        injection_representation,
    )

    normal_network = copy.deepcopy(network)
    frozen_network = copy.deepcopy(network)

    set_output_readout(
        frozen_network,
        injection_bias,
        injection_weights,
    )

    for _ in range(max(training_budgets)):
        batch_epoch(normal_network)

    normal_final_loss = loss(normal_network)
    frozen_final_loss = loss(frozen_network)

    normal_representation = hidden_representation(
        normal_network,
    )

    frozen_representation = hidden_representation(
        frozen_network,
    )

    normal_representation_change = representation_change(
        injection_representation,
        normal_representation,
    )

    frozen_representation_change = representation_change(
        injection_representation,
        frozen_representation,
    )

    return {
        "injection_readout_loss": (
            injection_readout_loss
        ),
        "normal_final_loss": (
            normal_final_loss
        ),
        "frozen_final_loss": (
            frozen_final_loss
        ),
        "normal_final_readout_loss": (
            best_linear_readout_loss(
                normal_representation,
            )
        ),
        "frozen_final_readout_loss": (
            best_linear_readout_loss(
                frozen_representation,
            )
        ),
        "normal_representation_change": (
            normal_representation_change
        ),
        "frozen_representation_change": (
            frozen_representation_change
        ),
    }


all_results = {}

print()
print("=" * 90)
print("EXPERIMENT 069 — FROZEN REPRESENTATION")
print(
    "Question: does representation drift cause failure when "
    "a sufficient representation already exists?"
)
print()
print(
    "Each run is split at the exact same post-injection state."
)
print()
print(
    "NORMAL:"
)
print(
    "  hidden layer and output layer continue ordinary training."
)
print()
print(
    "FROZEN COUNTERFACTUAL:"
)
print(
    "  hidden representation is held exactly at injection."
)
print(
    "  output weights are set immediately to the analytic "
    "best linear readout."
)
print()
print(
    "The frozen branch therefore removes output-layer optimization "
    "as a confounding factor."
)
print()
print(
    "Initially sufficient representations should remain perfectly "
    "solvable under the frozen counterfactual."
)
print(
    "Initially insufficient representations should remain insufficient "
    "because the frozen branch cannot change the hidden representation."
)
print()
print("10 seeds x 8 initialization offsets = 80 runs/condition")
print("15 structure/strength conditions = 1200 run pairs")
print("Success = loss < 1e-6.")
print("=" * 90)


for pattern_name in feature_patterns:
    all_results[pattern_name] = {}

    for strength in feature_strengths:

        injection_sufficient_count = 0
        normal_success_count = 0
        frozen_success_count = 0

        sufficient_normal_failed = 0
        sufficient_frozen_failed = 0
        normal_failed_frozen_success = 0

        insufficient_normal_success = 0
        insufficient_frozen_success = 0

        normal_final_losses = []
        frozen_final_losses = []

        normal_readout_losses = []
        frozen_readout_losses = []

        normal_cosines = []
        normal_pattern_changes = []
        normal_hamming = []

        for offset in initialization_offsets:
            for seed in seeds:

                result = run(
                    pattern_name,
                    strength,
                    seed,
                    offset,
                )

                initially_sufficient = (
                    result["injection_readout_loss"]
                    < success_threshold
                )

                normal_success = (
                    result["normal_final_loss"]
                    < success_threshold
                )

                frozen_success = (
                    result["frozen_final_loss"]
                    < success_threshold
                )

                if initially_sufficient:
                    injection_sufficient_count += 1

                if normal_success:
                    normal_success_count += 1

                if frozen_success:
                    frozen_success_count += 1

                if initially_sufficient:
                    if not normal_success:
                        sufficient_normal_failed += 1

                    if not frozen_success:
                        sufficient_frozen_failed += 1

                    if (
                        not normal_success
                        and frozen_success
                    ):
                        normal_failed_frozen_success += 1

                else:
                    if normal_success:
                        insufficient_normal_success += 1

                    if frozen_success:
                        insufficient_frozen_success += 1

                normal_final_losses.append(
                    result["normal_final_loss"]
                )

                frozen_final_losses.append(
                    result["frozen_final_loss"]
                )

                normal_readout_losses.append(
                    result["normal_final_readout_loss"]
                )

                frozen_readout_losses.append(
                    result["frozen_final_readout_loss"]
                )

                normal_cosines.append(
                    result[
                        "normal_representation_change"
                    ]["mean_feature_cosine"]
                )

                normal_pattern_changes.append(
                    result[
                        "normal_representation_change"
                    ]["pattern_changes"]
                )

                normal_hamming.append(
                    result[
                        "normal_representation_change"
                    ]["pattern_hamming"]
                )

        all_results[pattern_name][strength] = {
            "injection_sufficient": (
                injection_sufficient_count
            ),
            "normal_success": normal_success_count,
            "frozen_success": frozen_success_count,
            "sufficient_normal_failed": (
                sufficient_normal_failed
            ),
            "sufficient_frozen_failed": (
                sufficient_frozen_failed
            ),
            "normal_failed_frozen_success": (
                normal_failed_frozen_success
            ),
            "insufficient_normal_success": (
                insufficient_normal_success
            ),
            "insufficient_frozen_success": (
                insufficient_frozen_success
            ),
            "mean_normal_final_loss": (
                sum(normal_final_losses)
                / len(normal_final_losses)
            ),
            "mean_frozen_final_loss": (
                sum(frozen_final_losses)
                / len(frozen_final_losses)
            ),
            "mean_normal_readout_loss": (
                sum(normal_readout_losses)
                / len(normal_readout_losses)
            ),
            "mean_frozen_readout_loss": (
                sum(frozen_readout_losses)
                / len(frozen_readout_losses)
            ),
            "mean_normal_cosine": (
                sum(normal_cosines)
                / len(normal_cosines)
            ),
            "mean_normal_pattern_changes": (
                sum(normal_pattern_changes)
                / len(normal_pattern_changes)
            ),
            "mean_normal_hamming": (
                sum(normal_hamming)
                / len(normal_hamming)
            ),
        }

        result = all_results[pattern_name][strength]

        print()
        print("-" * 90)
        print(
            f"pattern={pattern_name} "
            f"strength={strength:.3f}"
        )
        print("-" * 90)

        print(
            f"injection_sufficient="
            f"{result['injection_sufficient']}/80"
        )

        print(
            f"normal_success="
            f"{result['normal_success']}/80   "
            f"frozen_success="
            f"{result['frozen_success']}/80"
        )

        print(
            "initially_sufficient: "
            f"normal_failed="
            f"{result['sufficient_normal_failed']}   "
            f"frozen_failed="
            f"{result['sufficient_frozen_failed']}   "
            f"normal_failed->frozen_success="
            f"{result['normal_failed_frozen_success']}"
        )

        print(
            "initially_insufficient: "
            f"normal_success="
            f"{result['insufficient_normal_success']}   "
            f"frozen_success="
            f"{result['insufficient_frozen_success']}"
        )

        print(
            f"mean_normal_final_loss="
            f"{result['mean_normal_final_loss']:.9f}"
        )

        print(
            f"mean_frozen_counterfactual_loss="
            f"{result['mean_frozen_final_loss']:.9f}"
        )

        print(
            f"mean_normal_final_readout_loss="
            f"{result['mean_normal_readout_loss']:.9f}"
        )

        print(
            f"mean_frozen_readout_loss="
            f"{result['mean_frozen_readout_loss']:.9f}"
        )

        print(
            f"normal_mean_feature_cosine="
            f"{result['mean_normal_cosine']:.6f}"
        )

        print(
            f"normal_mean_pattern_changes="
            f"{result['mean_normal_pattern_changes']:.3f}"
        )

        print(
            f"normal_mean_pattern_hamming="
            f"{result['mean_normal_hamming']:.3f}"
        )


print()
print("=" * 90)
print("EXPERIMENT 069 SUMMARY")
print("=" * 90)

print()
print(
    "Causal comparison:"
)

print(
    "The frozen counterfactual preserves the exact hidden "
    "representation present at injection and supplies its "
    "best possible linear output readout."
)

print()
print(
    "normal_failed->frozen_success = initially sufficient "
    "representation, normal training failed, but the same "
    "representation remained solvable when hidden drift was prevented."
)

print()
print(
    "This is the key measure of whether representation drift "
    "is implicated in normal-training failure."
)

print()
print(
    "insufficient->normal_success with "
    "insufficient->frozen_success = 0 shows the opposite case:"
)

print(
    "some successful normal trajectories require hidden "
    "representation change because the frozen injection "
    "representation cannot solve the task."
)

print()
print(
    "The frozen branch is deliberately not trained."
)

print(
    "It represents a counterfactual question:"
)

print(
    "What would happen if the hidden representation at injection "
    "were preserved perfectly and the output layer were given "
    "the best possible solution?"
)

print()
print("Status: Experiment 069 complete.")
