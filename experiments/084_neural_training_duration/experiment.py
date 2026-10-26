import importlib.util
import math
import random
from collections import Counter
from pathlib import Path

from src.layer import Layer
from src.network import Network
from src.neuron import Neuron


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "experiments" / "082_public_domain_text"
TOKENIZER_PATH = (
    REPO_ROOT
    / "experiments"
    / "078_character_tokenization"
    / "experiment.py"
)

TRAIN_PATH = DATA_DIR / "train_clean.txt"
HELD_OUT_PATH = DATA_DIR / "held_out_clean.txt"

HIDDEN_SIZE = 24
EPOCHS = 800
CHECKPOINTS = [200, 400, 600, 800]
LEARNING_RATE = 0.5
RANDOM_SEED = 8301
EPSILON = 1e-5


if not TRAIN_PATH.is_file() or not HELD_OUT_PATH.is_file():
    raise SystemExit(
        "Experiment 082 text files are missing. Download the book "
        "and run experiments/082_public_domain_text/prepare_data.py "
        "before running this experiment."
    )

spec = importlib.util.spec_from_file_location(
    "exp078_tokenizer", TOKENIZER_PATH
)

if spec is None or spec.loader is None:
    raise SystemExit("Could not load the Experiment 078 tokenizer.")

tokenizer_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tokenizer_module)


def check(name, condition):
    if not condition:
        raise AssertionError(f"FAILED: {name}")
    print(f"[PASS] {name}")


def read_paragraphs(path):
    text = path.read_text(encoding="utf-8")
    return [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]


def softmax(logits):
    """Convert raw scores into a stable probability distribution."""

    maximum = max(logits)
    exponentials = [
        math.exp(value - maximum)
        for value in logits
    ]
    denominator = sum(exponentials)

    return [
        value / denominator
        for value in exponentials
    ]


def one_hot(token_id, vocabulary_size):
    values = [0.0] * vocabulary_size
    values[token_id] = 1.0
    return values


def build_network(vocabulary_size):
    """Build a small ReLU network with linear output logits."""

    rng = random.Random(RANDOM_SEED)

    # He-style scale for the ReLU hidden layer.
    hidden_limit = math.sqrt(6.0 / vocabulary_size)

    hidden_neurons = []

    for _ in range(HIDDEN_SIZE):
        weights = [
            rng.uniform(-hidden_limit, hidden_limit)
            for _ in range(vocabulary_size)
        ]
        hidden_neurons.append(
            Neuron(weights, 0.0, activation="relu")
        )

    hidden_layer = Layer(neurons=hidden_neurons)

    # Use a smaller initial scale for the output scores.
    output_limit = math.sqrt(
        6.0 / (HIDDEN_SIZE + vocabulary_size)
    )

    output_neurons = []

    for _ in range(vocabulary_size):
        weights = [
            rng.uniform(-output_limit, output_limit)
            for _ in range(HIDDEN_SIZE)
        ]
        output_neurons.append(
            Neuron(weights, 0.0, activation="linear")
        )

    output_layer = Layer(neurons=output_neurons)

    return Network(layers=[hidden_layer, output_layer])


def transition_counts(sequences):
    """Count target characters by preceding character."""

    by_context = {}
    global_counts = Counter()
    total_targets = 0

    for sequence in sequences:
        for context_id, target_id in zip(
            sequence[:-1], sequence[1:]
        ):
            if context_id not in by_context:
                by_context[context_id] = Counter()

            by_context[context_id][target_id] += 1
            global_counts[target_id] += 1
            total_targets += 1

    return by_context, global_counts, total_targets


def mean_loss_from_probabilities(sequences, probabilities_by_context):
    total_loss = 0.0
    total_targets = 0
    correct = 0

    for sequence in sequences:
        for context_id, target_id in zip(
            sequence[:-1], sequence[1:]
        ):
            probabilities = probabilities_by_context[context_id]
            probability = max(probabilities[target_id], 1e-15)

            total_loss -= math.log(probability)
            prediction = max(
                range(len(probabilities)),
                key=lambda token: (probabilities[token], -token),
            )
            correct += prediction == target_id
            total_targets += 1

    if total_targets == 0:
        raise ValueError("No next-character targets were found.")

    return (
        total_loss / total_targets,
        correct,
        total_targets,
    )


def probabilities_for_all_contexts(network, vocabulary_size):
    """Evaluate the network once per possible preceding character."""

    distributions = []

    for context_id in range(vocabulary_size):
        logits = network.forward(
            one_hot(context_id, vocabulary_size)
        )
        distributions.append(softmax(logits))

    return distributions


def full_batch_gradient(network, counts_by_context, total_targets):
    """Accumulate the exact full-data gradient.

    All examples with the same preceding character have identical
    inputs. Their target counts can therefore be aggregated exactly.
    """

    vocabulary_size = len(network.layers[-1].neurons)

    weight_sums = [
        [
            [0.0] * len(neuron.weights)
            for neuron in layer.neurons
        ]
        for layer in network.layers
    ]

    bias_sums = [
        [0.0] * len(layer.neurons)
        for layer in network.layers
    ]

    total_loss = 0.0

    for context_id, target_counts in sorted(
        counts_by_context.items()
    ):
        context_total = sum(target_counts.values())

        logits = network.forward(
            one_hot(context_id, vocabulary_size)
        )
        probabilities = softmax(logits)

        for target_id, count in target_counts.items():
            total_loss -= (
                count / total_targets
            ) * math.log(max(probabilities[target_id], 1e-15))

        # Derivative of mean softmax cross-entropy with respect
        # to each output logit.
        output_gradients = [
            (
                probabilities[target_id] * context_total
                - target_counts.get(target_id, 0)
            ) / total_targets
            for target_id in range(vocabulary_size)
        ]

        network.backward(output_gradients)

        for layer_index, layer in enumerate(network.layers):
            for neuron_index, neuron in enumerate(layer.neurons):
                for weight_index, gradient in enumerate(
                    neuron.weight_gradients
                ):
                    weight_sums[layer_index][neuron_index][
                        weight_index
                    ] += gradient

                bias_sums[layer_index][neuron_index] += (
                    neuron.bias_gradient
                )

    # The existing Neuron.update method applies gradient descent.
    for layer_index, layer in enumerate(network.layers):
        for neuron_index, neuron in enumerate(layer.neurons):
            neuron.weight_gradients = weight_sums[
                layer_index
            ][neuron_index]

            neuron.bias_gradient = bias_sums[
                layer_index
            ][neuron_index]

    return total_loss


def finite_difference_gradient_check(network, context_id, target_id,
                                     vocabulary_size):
    """Compare one analytic gradient with a numerical estimate."""

    inputs = one_hot(context_id, vocabulary_size)
    logits = network.forward(inputs)
    probabilities = softmax(logits)

    output_gradients = list(probabilities)
    output_gradients[target_id] -= 1.0
    network.backward(output_gradients)

    hidden_activations = [
        neuron.activate(neuron.raw_output)
        for neuron in network.layers[0].neurons
    ]
    hidden_index = max(
        range(len(hidden_activations)),
        key=lambda index: hidden_activations[index],
    )

    if hidden_activations[hidden_index] <= 0:
        raise AssertionError(
            "Gradient check could not find an active hidden neuron."
        )

    output_neuron = network.layers[-1].neurons[target_id]
    analytic = output_neuron.weight_gradients[hidden_index]
    original_weight = output_neuron.weights[hidden_index]

    def sample_loss():
        current_logits = network.forward(inputs)
        current_probabilities = softmax(current_logits)
        return -math.log(
            max(current_probabilities[target_id], 1e-15)
        )

    output_neuron.weights[hidden_index] = (
        original_weight + EPSILON
    )
    loss_plus = sample_loss()

    output_neuron.weights[hidden_index] = (
        original_weight - EPSILON
    )
    loss_minus = sample_loss()

    output_neuron.weights[hidden_index] = original_weight

    numerical = (loss_plus - loss_minus) / (2.0 * EPSILON)

    return math.isclose(
        analytic,
        numerical,
        rel_tol=1e-4,
        abs_tol=1e-5,
    ), analytic, numerical


def evaluate_count_model(sequences, counts_by_context, global_counts):
    """Evaluate the one-character count-based reference."""

    global_prediction = max(
        global_counts,
        key=lambda token: (global_counts[token], -token),
    )

    correct = 0
    total = 0

    for sequence in sequences:
        for context_id, target_id in zip(
            sequence[:-1], sequence[1:]
        ):
            context_counts = counts_by_context.get(context_id)

            if context_counts:
                prediction = max(
                    context_counts,
                    key=lambda token: (
                        context_counts[token],
                        -token,
                    ),
                )
            else:
                prediction = global_prediction

            correct += prediction == target_id
            total += 1

    return correct, total


def parameters_are_finite(network):
    for layer in network.layers:
        for neuron in layer.neurons:
            if not math.isfinite(neuron.bias):
                return False
            if not all(
                math.isfinite(weight)
                for weight in neuron.weights
            ):
                return False
    return True


def main():
    train_paragraphs = read_paragraphs(TRAIN_PATH)
    held_out_paragraphs = read_paragraphs(HELD_OUT_PATH)

    train_text = TRAIN_PATH.read_text(encoding="utf-8")

    # Vocabulary is built from training text only.
    tokenizer = tokenizer_module.CharacterTokenizer("".join(train_paragraphs))

    train_sequences = [
        tokenizer.encode(paragraph)
        for paragraph in train_paragraphs
    ]
    held_out_sequences = [
        tokenizer.encode(paragraph)
        for paragraph in held_out_paragraphs
    ]

    if not train_sequences or not held_out_sequences:
        raise SystemExit("Training and held-out data cannot be empty.")

    vocabulary_size = tokenizer.vocab_size

    counts_by_context, global_counts, training_targets = (
        transition_counts(train_sequences)
    )

    held_out_targets = sum(
        len(sequence) - 1
        for sequence in held_out_sequences
    )

    print("=" * 76)
    print("EXPERIMENT 084 — NEURAL TRAINING DURATION")
    print("=" * 76)

    check(
        "Training and held-out data are present",
        bool(train_sequences) and bool(held_out_sequences),
    )

    check(
        "All held-out characters exist in the training vocabulary",
        all(
            token != tokenizer_module.UNK_ID
            for sequence in held_out_sequences
            for token in sequence
        ),
    )

    # Check softmax before training.
    sample_distribution = softmax([1.0, 2.0, -1.0])

    check(
        "Softmax probabilities sum to one",
        math.isclose(
            sum(sample_distribution),
            1.0,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        and all(
            0.0 <= probability <= 1.0
            for probability in sample_distribution
        ),
    )

    network = build_network(vocabulary_size)

    # Validate the backpropagation gradient numerically.
    first_pair = next(
        (sequence[0], sequence[1])
        for sequence in train_sequences
        if len(sequence) >= 2
    )

    gradient_passed, analytic, numerical = (
        finite_difference_gradient_check(
            network,
            first_pair[0],
            first_pair[1],
            vocabulary_size,
        )
    )

    check(
        "Backpropagation matches the numerical gradient",
        gradient_passed,
    )

    print(
        f"  Gradient check: analytic={analytic:.6g}, "
        f"numerical={numerical:.6g}"
    )

    initial_probabilities = probabilities_for_all_contexts(
        network, vocabulary_size
    )
    initial_train_loss, _, _ = mean_loss_from_probabilities(
        train_sequences, initial_probabilities
    )

    print()
    print(
        f"Network: {vocabulary_size} inputs -> "
        f"{HIDDEN_SIZE} ReLU neurons -> "
        f"{vocabulary_size} output scores"
    )
    print(f"Training targets: {training_targets}")
    print(f"Held-out targets: {held_out_targets}")
    print(f"Full-batch gradient epochs: {EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Initial training cross-entropy: {initial_train_loss:.4f}")

    # Every epoch uses the exact full-data gradient. Identical
    # one-character inputs are aggregated by their target counts.
    for epoch in range(1, EPOCHS + 1):
        loss_before_update = full_batch_gradient(
            network,
            counts_by_context,
            training_targets,
        )
        network.update(LEARNING_RATE)

        if epoch == 1 or epoch in CHECKPOINTS:
            print(
                f"epoch={epoch:03d} "
                f"training_loss_before_update={loss_before_update:.4f}",
                flush=True,
            )

        if epoch in CHECKPOINTS:
            checkpoint_probabilities = probabilities_for_all_contexts(
                network, vocabulary_size
            )
            train_loss, train_correct, train_total = (
                mean_loss_from_probabilities(
                    train_sequences,
                    checkpoint_probabilities,
                )
            )
            held_loss, held_correct, held_total = (
                mean_loss_from_probabilities(
                    held_out_sequences,
                    checkpoint_probabilities,
                )
            )

            print(
                f"  CHECKPOINT {epoch}: "
                f"train_loss={train_loss:.4f}, "
                f"train_accuracy={train_correct / train_total:.2%}, "
                f"held_out_loss={held_loss:.4f}, "
                f"held_out_accuracy={held_correct / held_total:.2%}",
                flush=True,
            )

    check(
        "All trained parameters remain finite",
        parameters_are_finite(network),
    )

    trained_probabilities = probabilities_for_all_contexts(
        network, vocabulary_size
    )

    final_train_loss, train_correct, train_total = (
        mean_loss_from_probabilities(
            train_sequences,
            trained_probabilities,
        )
    )

    held_out_loss, neural_correct, neural_total = (
        mean_loss_from_probabilities(
            held_out_sequences,
            trained_probabilities,
        )
    )

    check(
        "Training cross-entropy decreases",
        final_train_loss < initial_train_loss,
    )

    check(
        "All held-out targets are evaluated",
        neural_total == held_out_targets,
    )

    check(
        "Neural predictions use valid vocabulary IDs",
        all(
            max(
                range(vocabulary_size),
                key=lambda token: (
                    distribution[token],
                    -token,
                ),
            ) in tokenizer.id_to_char
            for distribution in trained_probabilities
        ),
    )

    baseline_correct, baseline_total = evaluate_count_model(
        held_out_sequences,
        {},
        global_counts,
    )

    count_correct, count_total = evaluate_count_model(
        held_out_sequences,
        counts_by_context,
        global_counts,
    )

    print()
    print("-" * 76)
    print("RESULTS")
    print("-" * 76)
    print(
        f"Training cross-entropy: {initial_train_loss:.4f} -> "
        f"{final_train_loss:.4f}"
    )
    print(
        f"Training accuracy: {train_correct}/{train_total} "
        f"({train_correct / train_total:.2%})"
    )
    print(
        f"Held-out cross-entropy: {held_out_loss:.4f}"
    )
    print(
        f"Most-common-character baseline: "
        f"{baseline_correct}/{baseline_total} "
        f"({baseline_correct / baseline_total:.2%})"
    )
    print(
        f"Count-based context-1 model: "
        f"{count_correct}/{count_total} "
        f"({count_correct / count_total:.2%})"
    )
    print(
        f"Neural model: {neural_correct}/{neural_total} "
        f"({neural_correct / neural_total:.2%})"
    )

    # Demonstrate greedy generation from a short prefix.
    prefix = tokenizer.encode("Alice ")
    generated = list(prefix)

    for _ in range(100):
        probabilities = trained_probabilities[generated[-1]]
        next_id = max(
            range(vocabulary_size),
            key=lambda token: (probabilities[token], -token),
        )
        generated.append(next_id)

    print()
    print("Generated sample (greedy, starting with 'Alice '):")
    print(tokenizer.decode(generated))

    print()
    print("Limitations:")
    print("- The model sees only the preceding character.")
    print("- It uses a small feed-forward network, not a transformer.")
    print("- Results are measured on later chapters of one book.")
    print("- Next-character accuracy does not demonstrate comprehension.")

    print()
    print("Conclusion:")
    print(
        "Larry now has a trainable neural model for next-character "
        "prediction, using softmax, cross-entropy, backpropagation, "
        "and gradient descent."
    )


if __name__ == "__main__":
    main()
