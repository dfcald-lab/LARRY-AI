import copy
import importlib.util
import math
import statistics
from pathlib import Path

EXPERIMENT_PATH = Path(
    "experiments/076_gradient_to_trajectory/experiment.py"
)

spec = importlib.util.spec_from_file_location(
    "experiment_076",
    EXPERIMENT_PATH,
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

# Edit PATTERN, FEATURE_STRENGTH, and GAMMA to select a test condition.
PATTERN = "0010|0100"
FEATURE_STRENGTH = 0.001
GAMMA = 0.5
BASE_RATE = 0.10
TARGET_STEP_RATIO = 0.01
CHECKPOINTS = {1, 2, 5, 10, 25}
FINAL_EPOCH = 25
ORIGINAL_HIDDEN_COUNT = 2
EXTRA_WIDTH = m.extra_width


def averaged_gradients(network):
    gradients = m.batch_gradients(network)
    batch_size = len(m.training_data)

    return [
        [
            {
                "weights": [
                    value / batch_size
                    for value in neuron_gradient["weights"]
                ],
                "bias": neuron_gradient["bias"] / batch_size,
            }
            for neuron_gradient in layer_gradients
        ]
        for layer_gradients in gradients
    ]


def apply_epoch(network, feature_rate=None):
    gradients = averaged_gradients(network)

    for layer_index, layer in enumerate(network.layers):
        for neuron_index, neuron in enumerate(layer.neurons):
            neuron.weight_gradients = list(
                gradients[layer_index][neuron_index]["weights"]
            )
            neuron.bias_gradient = (
                gradients[layer_index][neuron_index]["bias"]
            )

    for layer_index, layer in enumerate(network.layers):
        for neuron_index, neuron in enumerate(layer.neurons):
            rate = BASE_RATE

            if (
                feature_rate is not None
                and layer_index == 0
                and neuron_index >= ORIGINAL_HIDDEN_COUNT
            ):
                rate = feature_rate

            neuron.update(rate)


def new_feature_parameter_norm(network):
    values = []

    for neuron in network.layers[0].neurons[-EXTRA_WIDTH:]:
        values.extend(neuron.weights)
        values.append(neuron.bias)

    return math.sqrt(sum(value * value for value in values))


def new_feature_gradient_norm(network):
    gradients = averaged_gradients(network)
    values = []

    for gradient in gradients[0][-EXTRA_WIDTH:]:
        values.extend(gradient["weights"])
        values.append(gradient["bias"])

    return math.sqrt(sum(value * value for value in values))


def prepare_network(seed, offset):
    network = m.make_width2_network(seed)

    # Train the original width-2 network to the injection point.
    for _ in range(m.injection_epoch):
        apply_epoch(network)

    target_norm = m.standard_initialization_norm(seed, offset)
    features = m.construct_feature_pair(
        PATTERN,
        target_norm,
        FEATURE_STRENGTH,
    )

    m.append_new_neurons(network, features)

    # Save the existing output state, then couple the two new features
    # to the analytic readout direction using the specified gamma.
    output = network.layers[-1].neurons[0]
    original_bias = output.bias
    original_weights = list(output.weights)

    hidden_values = m.hidden_representation(network)
    _, _, analytic_weights = m.best_linear_readout_parameters(
        hidden_values
    )

    m.set_new_feature_output_coupling(
        network,
        original_bias,
        original_weights,
        analytic_weights,
        GAMMA,
    )

    return network


def both_new_features_active(network):
    hidden_values = m.hidden_representation(network)
    new_features = hidden_values[-EXTRA_WIDTH:]

    return all(
        any(value > 1e-12 for value in activations)
        for activations in new_features
    )


def network_is_finite(network):
    for layer in network.layers:
        for neuron in layer.neurons:
            values = list(neuron.weights) + [neuron.bias]

            if not all(math.isfinite(value) for value in values):
                return False

    return True


def run_condition(seed, offset):
    initial_network = prepare_network(seed, offset)
    injection_loss = m.loss(initial_network)

    parameter_norm = new_feature_parameter_norm(initial_network)
    gradient_norm = new_feature_gradient_norm(initial_network)

    if gradient_norm <= 1e-14:
        feature_rate = 0.0
    else:
        feature_rate = (
            TARGET_STEP_RATIO * parameter_norm / gradient_norm
        )

    outcomes = {}

    for mode in ("normal", "hidden-only"):
        network = copy.deepcopy(initial_network)
        losses = {}
        active = {}
        non_finite = False

        for epoch in range(1, FINAL_EPOCH + 1):
            rate = feature_rate if mode == "hidden-only" else None
            apply_epoch(network, feature_rate=rate)

            current_loss = m.loss(network)

            if (
                not math.isfinite(current_loss)
                or not network_is_finite(network)
            ):
                non_finite = True

            if epoch in CHECKPOINTS:
                losses[epoch] = current_loss
                active[epoch] = both_new_features_active(network)

        outcomes[mode] = {
            "initial_loss": injection_loss,
            "losses": losses,
            "active": active,
            "non_finite": non_finite,
        }

    return outcomes


def median_loss(rows, epoch):
    values = [
        row["losses"][epoch]
        for row in rows
        if math.isfinite(row["losses"][epoch])
    ]

    return statistics.median(values) if values else float("nan")


results = {
    "normal": [],
    "hidden-only": [],
}

for offset in m.initialization_offsets:
    for seed in m.seeds:
        condition = run_condition(seed, offset)

        for mode in results:
            results[mode].append(condition[mode])

print()
print("=" * 100)
print(
    f"80-run validation: {PATTERN}, "
    f"strength {FEATURE_STRENGTH}, gamma {GAMMA:+.3f}"
)
print("=" * 100)
print(
    "Mode        | Runs | Median loss @1 | @2 | @25 "
    "| Improved at @25 | Both active @2 | Both active @25 | Non-finite"
)

for mode, rows in results.items():
    improved = sum(
        math.isfinite(row["losses"][25])
        and row["losses"][25] < row["initial_loss"]
        for row in rows
    )
    active_at_2 = sum(row["active"][2] for row in rows)
    active_at_25 = sum(row["active"][25] for row in rows)
    non_finite = sum(row["non_finite"] for row in rows)

    print(
        f"{mode:<12} | {len(rows):4d} "
        f"| {median_loss(rows, 1):13.6g} "
        f"| {median_loss(rows, 2):.6g} "
        f"| {median_loss(rows, 25):.6g} "
        f"| {improved:3d}/{len(rows)} "
        f"| {active_at_2:3d}/{len(rows)} "
        f"| {active_at_25:3d}/{len(rows)} "
        f"| {non_finite}"
    )
