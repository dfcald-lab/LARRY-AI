import argparse
import importlib.util
import math
import statistics
from pathlib import Path


SOURCE = Path(
    "experiments/076_gradient_to_trajectory/experiment.py"
)

if not SOURCE.is_file():
    raise SystemExit(
        "Run this script from the LARRY-AI repository root."
    )

spec = importlib.util.spec_from_file_location("exp076", SOURCE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

PATTERN = "0010|0100"
STRENGTH = 0.001
GAMMAS = [0.0, 0.125, 0.5, 1.0]
SEEDS = list(range(10))
OFFSETS = [0, 1, 2, 3, 4, 5, 10, 100]
MAX_UPDATES = 25

LOSS_MULTIPLIER = 10.0
LOSS_ADDITIVE_MARGIN = 1.0

SIGNALS = [
    "predicted_step_pct",
    "update1_loss_ratio",
    "update1_movement_pct",
    "update1_pattern_hamming",
    "update1_dead_features",
]


def norm(values):
    return math.sqrt(sum(value * value for value in values))


def feature_patterns(network):
    representation = m.hidden_representation(network)[-m.extra_width:]
    return [m.binary_pattern(values) for values in representation]


def feature_active_counts(patterns):
    return [pattern.count("1") for pattern in patterns]


def pattern_hamming(first, second):
    return sum(
        left != right
        for a, b in zip(first, second)
        for left, right in zip(a, b)
    )


def new_parameter_movement(network, initial_parameters):
    current = m.new_feature_weight_state(network)
    start_norm = norm(initial_parameters)

    if start_norm < 1e-14:
        return 0.0

    return 100.0 * norm([
        current_value - initial_value
        for current_value, initial_value
        in zip(current, initial_parameters)
    ]) / start_norm


def make_starting_network(seed, offset, gamma):
    network = m.make_width2_network(seed)

    for _ in range(m.injection_epoch):
        m.batch_epoch(network)

    target_norm = m.standard_initialization_norm(seed, offset)
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


def run_one(seed, offset, gamma):
    network = make_starting_network(seed, offset, gamma)

    initial_loss = m.loss(network)
    initial_parameters = m.new_feature_weight_state(network)
    initial_parameter_norm = norm(initial_parameters)
    initial_patterns = feature_patterns(network)

    initial_gradients = m.flatten_new_feature_gradients(
        m.averaged_hidden_gradients(network)
    )
    initial_gradient_norm = norm(initial_gradients)

    predicted_step_pct = (
        100.0 * m.learning_rate
        * initial_gradient_norm / initial_parameter_norm
        if initial_parameter_norm > 1e-14 else 0.0
    )

    update1_loss_ratio = float("inf")
    update1_movement_pct = float("inf")
    update1_pattern_hamming = float("inf")
    update1_dead_features = float("inf")

    current_patterns = initial_patterns
    final_loss = initial_loss
    future_losses = []
    nonfinite = False

    for epoch in range(1, MAX_UPDATES + 1):
        m.batch_epoch(network)

        if not m.all_network_parameters_finite(network):
            nonfinite = True
            final_loss = float("inf")
            break

        current_loss = m.loss(network)

        if not math.isfinite(current_loss):
            nonfinite = True
            final_loss = float("inf")
            break

        current_patterns = feature_patterns(network)
        final_loss = current_loss

        if epoch == 1:
            update1_loss_ratio = (
                current_loss / max(initial_loss, 1e-15)
            )
            update1_movement_pct = new_parameter_movement(
                network, initial_parameters
            )
            update1_pattern_hamming = pattern_hamming(
                initial_patterns, current_patterns
            )
            update1_dead_features = sum(
                count == 0
                for count in feature_active_counts(current_patterns)
            )

        if epoch >= 2:
            future_losses.append(current_loss)

    future_max_loss = (
        max(future_losses)
        if future_losses else float("inf")
    )

    escalation_threshold = max(
        LOSS_MULTIPLIER * initial_loss,
        initial_loss + LOSS_ADDITIVE_MARGIN,
    )

    future_escalation = (
        nonfinite or future_max_loss > escalation_threshold
    )

    dead_final = sum(
        count == 0
        for count in feature_active_counts(current_patterns)
    )

    return {
        "seed": seed,
        "offset": offset,
        "gamma": gamma,
        "initial_loss": initial_loss,
        "update1_loss_ratio": update1_loss_ratio,
        "update1_movement_pct": update1_movement_pct,
        "update1_pattern_hamming": update1_pattern_hamming,
        "update1_dead_features": update1_dead_features,
        "predicted_step_pct": predicted_step_pct,
        "future_max_loss": future_max_loss,
        "final_loss": final_loss,
        "future_escalation": future_escalation,
        "dead_final": dead_final,
        "nonfinite": nonfinite,
    }


def average_ranks(values):
    values = [
        float("inf") if math.isnan(value) else value
        for value in values
    ]

    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    start = 0

    while start < len(order):
        end = start + 1

        while (
            end < len(order)
            and values[order[end]] == values[order[start]]
        ):
            end += 1

        average_rank = ((start + 1) + end) / 2.0

        for position in range(start, end):
            ranks[order[position]] = average_rank

        start = end

    return ranks


def spearman(first, second):
    if len(first) != len(second) or len(first) < 2:
        return float("nan")

    x = average_ranks(first)
    y = average_ranks(second)

    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)

    covariance = sum(
        (a - mean_x) * (b - mean_y)
        for a, b in zip(x, y)
    )
    variance_x = sum((a - mean_x) ** 2 for a in x)
    variance_y = sum((b - mean_y) ** 2 for b in y)

    if variance_x == 0.0 or variance_y == 0.0:
        return float("nan")

    return covariance / math.sqrt(variance_x * variance_y)


def fmt(value):
    if math.isnan(value):
        return "n/a"
    if math.isinf(value):
        return "inf"
    return f"{value:.4g}"


def summarize(records):
    print()
    print("=" * 88)
    print("EXPERIMENT 077 — RESULTS BY OUTPUT COUPLING")
    print("=" * 88)
    print(
        "Future escalation means loss exceeds max(10x injection loss, "
        "injection loss + 1.0) during updates 2–25."
    )
    print(
        "Feature death means a new ReLU feature is inactive on all "
        "four XOR examples."
    )

    for gamma in GAMMAS:
        group = [r for r in records if r["gamma"] == gamma]
        count = len(group)

        print()
        print(
            f"gamma={gamma:+.3f} runs={count} "
            f"future_escalations="
            f"{sum(r['future_escalation'] for r in group)}/{count} "
            f"nonfinite={sum(r['nonfinite'] for r in group)}/{count}"
        )
        print(
            f"  median loss: injection={fmt(statistics.median(r['initial_loss'] for r in group))} "
            f"update1 ratio={fmt(statistics.median(r['update1_loss_ratio'] for r in group))} "
            f"future max={fmt(statistics.median(r['future_max_loss'] for r in group))} "
            f"final={fmt(statistics.median(r['final_loss'] for r in group))}"
        )
        print(
            f"  median predicted step={fmt(statistics.median(r['predicted_step_pct'] for r in group))}% "
            f"update1 movement={fmt(statistics.median(r['update1_movement_pct'] for r in group))}% "
            f"update1 pattern flips={fmt(statistics.median(r['update1_pattern_hamming'] for r in group))}"
        )
        print(
            f"  at least one feature dead at update 25: "
            f"{sum(r['dead_final'] >= 1 for r in group)}/{count}; "
            f"both dead: {sum(r['dead_final'] == 2 for r in group)}/{count}"
        )

        print("  Within-coupling Spearman correlations:")
        for signal in SIGNALS:
            rho_loss = spearman(
                [r[signal] for r in group],
                [r["future_max_loss"] for r in group],
            )
            rho_death = spearman(
                [r[signal] for r in group],
                [r["dead_final"] for r in group],
            )
            print(
                f"    {signal}: "
                f"future_max_loss rho={fmt(rho_loss)}, "
                f"final_dead_features rho={fmt(rho_death)}"
            )

    print()
    print("Interpretation note:")
    print(
        "Correlations describe association within each tested coupling. "
        "They do not prove causation or guarantee prediction on new data."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one seed and one offset at each of the four couplings.",
    )
    args = parser.parse_args()

    seeds = [0] if args.smoke_test else SEEDS
    offsets = [0] if args.smoke_test else OFFSETS
    expected = len(seeds) * len(offsets)

    print("=" * 88)
    print("EXPERIMENT 077 — EARLY WARNING SIGNALS")
    print("=" * 88)
    print(f"Feature pattern: {PATTERN}")
    print(f"Feature strength: {STRENGTH}")
    print(f"Couplings: {GAMMAS}")
    print(f"Updates per run: {MAX_UPDATES}")
    print(f"Runs per coupling: {expected}")
    print(f"Total runs: {expected * len(GAMMAS)}")
    print("Training mode: normal training at the original learning rate.")
    print("Existing conditions only; no new coupling values.")

    records = []

    for gamma in GAMMAS:
        for offset in offsets:
            for seed in seeds:
                print(
                    f"Running gamma={gamma:+.3f}, "
                    f"seed={seed}, offset={offset}",
                    flush=True,
                )
                records.append(run_one(seed, offset, gamma))

    summarize(records)


if __name__ == "__main__":
    main()
