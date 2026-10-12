import copy
import math
import statistics

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

# Signed output coupling for the newly injected neurons.
# gamma=0 preserves their normal zero coupling at injection.
# positive gamma follows the analytic output direction.
# negative gamma reverses the analytic output direction.
new_feature_gammas = [
    -1.0,
    -0.5,
    0.0,
    0.5,
    1.0,
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


def anchored_batch_epoch(
    network,
    reference_representation,
    stability_lambda,
):
    gradients = batch_gradients(network)
    batch_size = len(training_data)

    if stability_lambda != 0.0:
        hidden_layer = network.layers[0]

        for example_index, (inputs, _) in enumerate(
            training_data
        ):
            network.forward(inputs)

            for neuron_index, neuron in enumerate(
                hidden_layer.neurons
            ):
                current_activation = neuron.activate(
                    neuron.raw_output
                )

                reference_activation = (
                    reference_representation[
                        neuron_index
                    ][example_index]
                )

                activation_difference = (
                    current_activation
                    - reference_activation
                )

                dz = (
                    stability_lambda
                    * activation_difference
                    * neuron.activation_derivative(
                        neuron.raw_output
                    )
                )

                for weight_index, input_value in enumerate(
                    inputs
                ):
                    gradients[0][neuron_index]["weights"][
                        weight_index
                    ] += dz * input_value

                gradients[0][neuron_index]["bias"] += dz

    for layer_index, layer in enumerate(network.layers):
        for neuron_index, neuron in enumerate(layer.neurons):
            neuron.weight_gradients = [
                gradient / batch_size
                for gradient in gradients[
                    layer_index
                ][neuron_index]["weights"]
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




def set_new_feature_output_coupling(
    network,
    original_bias,
    original_weights,
    analytic_weights,
    gamma,
):
    output_neuron = network.layers[-1].neurons[0]

    # Preserve the learned pre-injection output state.
    output_neuron.bias = original_bias
    output_neuron.weights = list(original_weights)

    # Change only the two output weights connected to
    # the newly injected hidden neurons.
    for index in range(extra_width):
        output_neuron.weights[
            -extra_width + index
        ] = (
            gamma
            * analytic_weights[
                -extra_width + index
            ]
        )

    output_neuron.weight_gradients = [
        0.0
        for _ in output_neuron.weights
    ]

    output_neuron.bias_gradient = 0.0



def averaged_hidden_gradients(network):
    gradients = batch_gradients(network)
    batch_size = len(training_data)

    return [
        {
            "weights": [
                value / batch_size
                for value in gradient["weights"]
            ],
            "bias": gradient["bias"] / batch_size,
        }
        for gradient in gradients[0]
    ]


def flatten_new_feature_gradients(gradients):
    values = []

    for gradient in gradients[-extra_width:]:
        values.extend(gradient["weights"])
        values.append(gradient["bias"])

    return values


def vector_norm(values):
    return math.sqrt(
        sum(value * value for value in values)
    )



def vector_add(first, second):
    return [
        left + right
        for left, right in zip(first, second)
    ]


def vector_subtract(first, second):
    return [
        left - right
        for left, right in zip(first, second)
    ]


def vector_scale(values, factor):
    return [
        value * factor
        for value in values
    ]


def vector_mean_absolute_error(actual, predicted):
    if not actual:
        return 0.0

    return sum(
        abs(left - right)
        for left, right in zip(actual, predicted)
    ) / len(actual)



trajectory_horizons = [1, 2, 5, 10, 25]
trajectory_gammas = [-1.0, -0.5, 0.5, 1.0]
minimum_signal_norm = 1e-10


def new_feature_weight_state(network):
    values = []

    for neuron in network.layers[0].neurons[-extra_width:]:
        values.extend(neuron.weights)
        values.append(neuron.bias)

    return values


def new_feature_representation(network):
    return hidden_representation(network)[-extra_width:]


def flatten_representation(values):
    return [
        value
        for feature in values
        for value in feature
    ]


def all_network_parameters_finite(network):
    return all(
        math.isfinite(neuron.bias)
        and all(math.isfinite(weight) for weight in neuron.weights)
        for layer in network.layers
        for neuron in layer.neurons
    )


def apply_new_feature_update(network, gradients):
    for index, gradient in enumerate(gradients[-extra_width:]):
        neuron = network.layers[0].neurons[
            -extra_width + index
        ]

        neuron.weights = [
            weight - learning_rate * value
            for weight, value in zip(
                neuron.weights,
                gradient["weights"],
            )
        ]

        neuron.bias -= learning_rate * gradient["bias"]


def run(
    pattern_name,
    strength,
    gamma,
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

    append_new_neurons(network, features)

    # IMPORTANT: fit the analytic readout using every hidden feature.
    full_injection_representation = hidden_representation(network)

    # Measure movement only for the two newly injected features.
    initial_representation = (
        full_injection_representation[-extra_width:]
    )

    output_neuron = network.layers[-1].neurons[0]
    original_bias = output_neuron.bias
    original_weights = list(output_neuron.weights)

    _, _, analytic_weights = best_linear_readout_parameters(
        full_injection_representation
    )

    measured_network = copy.deepcopy(network)

    set_new_feature_output_coupling(
        measured_network,
        original_bias,
        original_weights,
        analytic_weights,
        gamma,
    )

    initial_weights = new_feature_weight_state(measured_network)

    gradient_records = averaged_hidden_gradients(measured_network)

    predicted_update = vector_scale(
        flatten_new_feature_gradients(gradient_records),
        -learning_rate,
    )

    # Predict the first representation change using only the
    # initial gradients of the injected hidden neurons.
    predicted_network = copy.deepcopy(measured_network)

    apply_new_feature_update(
        predicted_network,
        gradient_records,
    )

    predicted_representation = new_feature_representation(
        predicted_network
    )

    predicted_representation_delta = vector_subtract(
        flatten_representation(predicted_representation),
        flatten_representation(initial_representation),
    )

    predicted_weight_step_norm = vector_norm(predicted_update)
    predicted_representation_step_norm = vector_norm(
        predicted_representation_delta
    )

    # Let the real training loop update every layer normally.
    actual_network = copy.deepcopy(measured_network)

    results = {}

    for epoch in range(1, max(trajectory_horizons) + 1):
        batch_epoch(actual_network)

        if not all_network_parameters_finite(actual_network):
            for horizon in trajectory_horizons:
                if horizon >= epoch:
                    results[horizon] = {"unstable": True}
            break

        if epoch not in trajectory_horizons:
            continue

        current_weights = new_feature_weight_state(actual_network)
        current_representation = new_feature_representation(
            actual_network
        )

        actual_representation_delta = vector_subtract(
            flatten_representation(current_representation),
            flatten_representation(initial_representation),
        )

        actual_weight_displacement = vector_subtract(
            current_weights,
            initial_weights,
        )

        initial_patterns = [
            binary_pattern(values)
            for values in initial_representation
        ]

        current_patterns = [
            binary_pattern(values)
            for values in current_representation
        ]

        pattern_hamming = sum(
            left != right
            for initial, current in zip(
                initial_patterns,
                current_patterns,
            )
            for left, right in zip(initial, current)
        )

        weight_signal_valid = (
            predicted_weight_step_norm > minimum_signal_norm
        )
        representation_signal_valid = (
            predicted_representation_step_norm > minimum_signal_norm
        )

        actual_weight_norm = vector_norm(actual_weight_displacement)
        actual_representation_norm = vector_norm(
            actual_representation_delta
        )

        results[epoch] = {
            "unstable": False,
            "initial_weight_step_norm": predicted_weight_step_norm,
            "initial_representation_step_norm": (
                predicted_representation_step_norm
            ),
            "actual_weight_displacement_norm": actual_weight_norm,
            "actual_representation_displacement_norm": (
                actual_representation_norm
            ),
            "weight_direction_valid": weight_signal_valid,
            "representation_direction_valid": (
                representation_signal_valid
            ),
            "weight_cosine": (
                cosine_similarity(
                    predicted_update,
                    actual_weight_displacement,
                )
                if weight_signal_valid else None
            ),
            "weight_ratio": (
                actual_weight_norm / predicted_weight_step_norm
                if weight_signal_valid else None
            ),
            "representation_cosine": (
                cosine_similarity(
                    predicted_representation_delta,
                    actual_representation_delta,
                )
                if representation_signal_valid else None
            ),
            "representation_ratio": (
                actual_representation_norm
                / predicted_representation_step_norm
                if representation_signal_valid else None
            ),
            "pattern_hamming": pattern_hamming,
        }

    for horizon in trajectory_horizons:
        results.setdefault(horizon, {"unstable": True})

    return results


if __name__ == "__main__":
    print()
    print("=" * 100)
    print("EXPERIMENT 076 — GRADIENT TO TRAJECTORY")
    print("Question: how long does the initial gradient predict hidden-feature movement?")
    print()
    print("The analytic readout uses the full hidden representation.")
    print("Movement measurements track only the newly injected features.")
    print("All layers train normally in the actual trajectory.")
    print()
    print("10 seeds x 8 initialization offsets = 80 runs per condition")
    print(f"Gammas: {trajectory_gammas}")
    print(f"Update horizons: {trajectory_horizons}")
    print("=" * 100)


    for pattern_name in feature_patterns:
        for strength in feature_strengths:
            for gamma in trajectory_gammas:
                all_runs = []

                for offset in initialization_offsets:
                    for seed in seeds:
                        all_runs.append(
                            run(
                                pattern_name,
                                strength,
                                gamma,
                                seed,
                                offset,
                            )
                        )

                print()
                print("-" * 100)
                print(
                    f"pattern={pattern_name} "
                    f"strength={strength:.3f} "
                    f"gamma={gamma:+.2f}"
                )
                print("-" * 100)

                for horizon in trajectory_horizons:
                    subset = [
                        result[horizon]
                        for result in all_runs
                        if not result[horizon]["unstable"]
                    ]

                    nonfinite_count = len(all_runs) - len(subset)

                    if not subset:
                        print(
                            f"epoch={horizon:>2} finite_runs=0 "
                            f"nonfinite_runs={nonfinite_count}"
                        )
                        continue

                    weight_valid = [
                        result for result in subset
                        if result["weight_direction_valid"]
                    ]
                    representation_valid = [
                        result for result in subset
                        if result["representation_direction_valid"]
                    ]

                    weight_ratios = [
                        result["weight_ratio"]
                        for result in weight_valid
                    ]
                    representation_ratios = [
                        result["representation_ratio"]
                        for result in representation_valid
                    ]

                    weight_initial_step_norm = statistics.median(
                        result["initial_weight_step_norm"]
                        for result in weight_valid
                    ) if weight_valid else float("nan")

                    weight_actual_displacement_norm = statistics.median(
                        result["actual_weight_displacement_norm"]
                        for result in weight_valid
                    ) if weight_valid else float("nan")

                    representation_initial_step_norm = statistics.median(
                        result["initial_representation_step_norm"]
                        for result in representation_valid
                    ) if representation_valid else float("nan")

                    representation_actual_displacement_norm = statistics.median(
                        result["actual_representation_displacement_norm"]
                        for result in representation_valid
                    ) if representation_valid else float("nan")

                    weight_cosine = (
                        statistics.mean(
                            result["weight_cosine"]
                            for result in weight_valid
                        )
                        if weight_valid else float("nan")
                    )

                    representation_cosine = (
                        statistics.mean(
                            result["representation_cosine"]
                            for result in representation_valid
                        )
                        if representation_valid else float("nan")
                    )

                    weight_median_ratio = (
                        statistics.median(weight_ratios)
                        if weight_ratios else float("nan")
                    )

                    representation_median_ratio = (
                        statistics.median(representation_ratios)
                        if representation_ratios else float("nan")
                    )

                    print(
                        f"epoch={horizon:>2} "
                        f"finite_runs={len(subset):>2} "
                        f"nonfinite_runs={nonfinite_count:>2} "
                        f"weight_signal_runs={len(weight_valid):>2} "
                        f"representation_signal_runs="
                        f"{len(representation_valid):>2} "
                        f"weight_step_norm={weight_initial_step_norm:.3e} "
                        f"weight_displacement_norm="
                        f"{weight_actual_displacement_norm:.3e} "
                        f"representation_step_norm="
                        f"{representation_initial_step_norm:.3e} "
                        f"representation_displacement_norm="
                        f"{representation_actual_displacement_norm:.3e} "
                        f"weight_cosine={weight_cosine:.6f} "
                        f"representation_cosine="
                        f"{representation_cosine:.6f} "
                        f"weight_median_ratio={weight_median_ratio:.6f} "
                        f"representation_median_ratio="
                        f"{representation_median_ratio:.6f} "
                        f"pattern_hamming="
                        f"{statistics.mean(r['pattern_hamming'] for r in subset):.3f}"
                    )


    print()
    print("=" * 100)
    print("EXPERIMENT 076 SUMMARY")
    print("=" * 100)
    print("Epoch 1 should closely match the predicted initial update.")
    print("Later cosine values show whether the trajectory changes direction.")
    print("Pattern Hamming tracks changes to the injected features' ReLU patterns.")
    print("Non-finite runs are reported separately.")
    print("Near-zero initial directions are excluded from cosine and ratio summaries.")
    print("Ratios are medians of cumulative movement divided by initial predicted movement.")
