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



trajectory_steps = [
    0,
    1,
    5,
    20,
    100,
    500,
    1000,
    2000,
    3900,
]


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

    injection_readout_loss = (
        best_linear_readout_loss(
            injection_representation,
        )
    )

    trajectory = []

    current_step = 0

    for step in trajectory_steps:
        while current_step < step:
            batch_epoch(network)
            current_step += 1

        current_representation = hidden_representation(
            network,
        )

        change = representation_change(
            injection_representation,
            current_representation,
        )

        trajectory.append(
            {
                "step": step,
                "readout_loss": (
                    best_linear_readout_loss(
                        current_representation,
                    )
                ),
                "mean_feature_cosine": (
                    change["mean_feature_cosine"]
                ),
                "min_feature_cosine": (
                    change["min_feature_cosine"]
                ),
                "pattern_changes": (
                    change["pattern_changes"]
                ),
                "pattern_hamming": (
                    change["pattern_hamming"]
                ),
            }
        )

    final_loss = loss(network)

    return (
        injection_readout_loss,
        final_loss,
        trajectory,
    )


all_results = {}

print()
print("=" * 90)
print("EXPERIMENT 068 — REPRESENTATION TRAJECTORY")
print(
    "Question: does successful training preserve, refine, "
    "or construct the hidden representation?"
)
print()
print(
    "The hidden representation is measured at multiple points "
    "after injection."
)
print(
    "At every checkpoint, the representation is given the best "
    "possible analytic linear readout."
)
print()
print(
    "Representation change is measured relative to the "
    "injection representation using:"
)
print("- mean activation-vector cosine similarity")
print("- number of hidden features whose binary pattern changed")
print("- total binary activation-pattern Hamming distance")
print()
print(
    "Runs are grouped by:"
)
print("S→S = initially sufficient, final training success")
print("S→F = initially sufficient, final training failure")
print("I→S = initially insufficient, final training success")
print("I→F = initially insufficient, final training failure")
print()
print("10 seeds x 8 initialization offsets = 80 runs/condition")
print("15 structure/strength conditions = 1200 total runs")
print("Success = loss < 1e-6.")
print("=" * 90)


for pattern_name in feature_patterns:
    all_results[pattern_name] = {}

    for strength in feature_strengths:

        groups = {
            "S→S": {
                "count": 0,
                "points": {
                    step: {
                        "readout": [],
                        "cosine": [],
                        "changed": [],
                        "hamming": [],
                    }
                    for step in trajectory_steps
                },
            },
            "S→F": {
                "count": 0,
                "points": {
                    step: {
                        "readout": [],
                        "cosine": [],
                        "changed": [],
                        "hamming": [],
                    }
                    for step in trajectory_steps
                },
            },
            "I→S": {
                "count": 0,
                "points": {
                    step: {
                        "readout": [],
                        "cosine": [],
                        "changed": [],
                        "hamming": [],
                    }
                    for step in trajectory_steps
                },
                "first_sufficient_steps": [],
            },
            "I→F": {
                "count": 0,
                "points": {
                    step: {
                        "readout": [],
                        "cosine": [],
                        "changed": [],
                        "hamming": [],
                    }
                    for step in trajectory_steps
                },
            },
        }

        for offset in initialization_offsets:
            for seed in seeds:

                (
                    injection_readout_loss,
                    final_loss,
                    trajectory,
                ) = run(
                    pattern_name,
                    strength,
                    seed,
                    offset,
                )

                initially_sufficient = (
                    injection_readout_loss
                    < success_threshold
                )

                final_success = (
                    final_loss
                    < success_threshold
                )

                if initially_sufficient and final_success:
                    group_name = "S→S"

                elif initially_sufficient and not final_success:
                    group_name = "S→F"

                elif not initially_sufficient and final_success:
                    group_name = "I→S"

                else:
                    group_name = "I→F"

                group = groups[group_name]
                group["count"] += 1

                for point in trajectory:
                    step = point["step"]

                    group["points"][step][
                        "readout"
                    ].append(
                        point["readout_loss"]
                    )

                    group["points"][step][
                        "cosine"
                    ].append(
                        point["mean_feature_cosine"]
                    )

                    group["points"][step][
                        "changed"
                    ].append(
                        point["pattern_changes"]
                    )

                    group["points"][step][
                        "hamming"
                    ].append(
                        point["pattern_hamming"]
                    )

                if group_name == "I→S":
                    first_sufficient_step = None

                    for point in trajectory:
                        if (
                            point["readout_loss"]
                            < success_threshold
                        ):
                            first_sufficient_step = (
                                point["step"]
                            )
                            break

                    group[
                        "first_sufficient_steps"
                    ].append(
                        first_sufficient_step
                    )

        all_results[pattern_name][strength] = groups

        print()
        print("-" * 90)
        print(
            f"pattern={pattern_name} "
            f"strength={strength:.3f}"
        )
        print("-" * 90)

        for group_name in [
            "S→S",
            "S→F",
            "I→S",
            "I→F",
        ]:
            group = groups[group_name]

            print()
            print(
                f"group={group_name} "
                f"count={group['count']}/80"
            )

            if group["count"] == 0:
                continue

            print(
                "step   "
                "readout_loss       "
                "mean_cosine       "
                "pattern_changes   "
                "pattern_hamming"
            )

            for step in trajectory_steps:
                point = group["points"][step]

                mean_readout = (
                    sum(point["readout"])
                    / len(point["readout"])
                )

                mean_cosine = (
                    sum(point["cosine"])
                    / len(point["cosine"])
                )

                mean_changed = (
                    sum(point["changed"])
                    / len(point["changed"])
                )

                mean_hamming = (
                    sum(point["hamming"])
                    / len(point["hamming"])
                )

                print(
                    f"{step:>4}   "
                    f"{mean_readout:>16.9f}   "
                    f"{mean_cosine:>15.6f}   "
                    f"{mean_changed:>16.3f}   "
                    f"{mean_hamming:>15.3f}"
                )

            if (
                group_name == "I→S"
                and group["first_sufficient_steps"]
            ):
                counts = {}

                for step in group[
                    "first_sufficient_steps"
                ]:
                    counts[step] = counts.get(step, 0) + 1

                print()
                print(
                    "first checkpoint where "
                    "representation became linearly sufficient:"
                )

                for step in trajectory_steps:
                    count = counts.get(step, 0)

                    if count:
                        print(
                            f"  step={step:>4}: "
                            f"{count} runs"
                        )


print()
print("=" * 90)
print("EXPERIMENT 068 SUMMARY")
print("=" * 90)

print()
print(
    "Interpretation guide:"
)
print(
    "S→S with high cosine and low pattern change "
    "suggests preservation."
)
print(
    "S→S with declining cosine or changing patterns "
    "suggests refinement despite an already-sufficient representation."
)
print(
    "I→S with readout loss falling below the threshold "
    "shows that training constructed a linearly sufficient representation."
)
print(
    "Large representation change together with I→S "
    "would indicate substantial representational construction."
)
print(
    "S→F isolates optimization failure from representation sufficiency."
)
print(
    "I→F shows cases where training did not construct a "
    "sufficient representation."
)
