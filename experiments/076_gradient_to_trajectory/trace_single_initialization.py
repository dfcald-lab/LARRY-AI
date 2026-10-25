import copy
import importlib.util
import math
from pathlib import Path

source = Path(
    "experiments/076_gradient_to_trajectory/experiment.py"
)
spec = importlib.util.spec_from_file_location("exp076", source)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

SEED = 0
OFFSET = 0
STRENGTH = 0.001
PATTERN = "0010|0100"
BASE_RATE = 0.10
STEP_RATIO = 0.01
GAMMAS = [0.0, 0.125, 0.5, 1.0]
CHECKPOINTS = {0, 1, 2, 3, 5, 10, 25}
OLD_HIDDEN_COUNT = 2
NEW_WIDTH = m.extra_width


def norm(values):
    return math.sqrt(sum(value * value for value in values))


def new_parameters(network):
    values = []
    for neuron in network.layers[0].neurons[-NEW_WIDTH:]:
        values.extend(neuron.weights)
        values.append(neuron.bias)
    return values


def predictions(network):
    return [
        network.forward(inputs)[0]
        for inputs, _ in m.training_data
    ]


def activation_patterns(network):
    features = m.hidden_representation(network)[-NEW_WIDTH:]
    return "|".join(m.binary_pattern(values) for values in features)


def averaged_gradients(network):
    gradients = m.batch_gradients(network)
    batch_size = len(m.training_data)

    for layer_index, layer in enumerate(network.layers):
        for neuron_index, neuron in enumerate(layer.neurons):
            item = gradients[layer_index][neuron_index]
            neuron.weight_gradients = [
                value / batch_size for value in item["weights"]
            ]
            neuron.bias_gradient = item["bias"] / batch_size


def train_one_epoch(network, hidden_rate=None):
    averaged_gradients(network)

    for layer_index, layer in enumerate(network.layers):
        for neuron_index, neuron in enumerate(layer.neurons):
            rate = BASE_RATE
            if (
                hidden_rate is not None
                and layer_index == 0
                and neuron_index >= OLD_HIDDEN_COUNT
            ):
                rate = hidden_rate
            neuron.update(rate)


def make_starting_network(gamma):
    network = m.make_width2_network(SEED)

    for _ in range(m.injection_epoch):
        m.batch_epoch(network)

    target_norm = m.standard_initialization_norm(SEED, OFFSET)
    features = m.construct_feature_pair(
        PATTERN, target_norm, STRENGTH
    )
    m.append_new_neurons(network, features)

    representation = m.hidden_representation(network)
    output = network.layers[-1].neurons[0]
    original_bias = output.bias
    original_weights = list(output.weights)

    _, _, analytic_weights = m.best_linear_readout_parameters(
        representation
    )

    m.set_new_feature_output_coupling(
        network,
        original_bias,
        original_weights,
        analytic_weights,
        gamma,
    )
    return network


def measure(network, initial_parameters):
    current_parameters = new_parameters(network)
    displacement = norm([
        current - initial
        for current, initial in zip(
            current_parameters, initial_parameters
        )
    ])
    start_norm = norm(initial_parameters)
    movement_percent = (
        100.0 * displacement / start_norm
        if start_norm > 1e-14 else 0.0
    )

    output_weights = network.layers[-1].neurons[0].weights[-NEW_WIDTH:]
    output_text = ",".join(f"{value:.3g}" for value in output_weights)
    prediction_text = ",".join(
        f"{value:.3g}" for value in predictions(network)
    )

    print(
        f"epoch={epoch:2d} "
        f"loss={m.loss(network):.6g} "
        f"pred=[{prediction_text}] "
        f"patterns={activation_patterns(network)} "
        f"new_output_weights=[{output_text}] "
        f"hidden_movement={movement_percent:.4g}%"
    )


print("Single-initialization trajectory trace")
print(f"seed={SEED}, offset={OFFSET}, pattern={PATTERN}")
print(f"strength={STRENGTH}, base_rate={BASE_RATE}")
print("Couplings are previously tested; no broad sweep is running.")

for gamma in GAMMAS:
    starting_network = make_starting_network(gamma)
    initial_parameters = new_parameters(starting_network)

    gradient_norm = norm(
        m.flatten_new_feature_gradients(
            m.averaged_hidden_gradients(starting_network)
        )
    )
    parameter_norm = norm(initial_parameters)
    hidden_rate = (
        STEP_RATIO * parameter_norm / gradient_norm
        if gradient_norm > 1e-14 else 0.0
    )

    for mode in ("normal", "hidden-only"):
        network = copy.deepcopy(starting_network)
        rate = hidden_rate if mode == "hidden-only" else None

        print()
        print(
            f"gamma={gamma:+.3f} mode={mode} "
            f"hidden_rate={rate if rate is not None else BASE_RATE:.6g}"
        )

        for epoch in range(0, max(CHECKPOINTS) + 1):
            if epoch > 0:
                train_one_epoch(network, hidden_rate=rate)

            if epoch in CHECKPOINTS:
                measure(network, initial_parameters)
