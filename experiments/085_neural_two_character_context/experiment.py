import importlib.util
import math
import random
from collections import Counter
from pathlib import Path

from src.layer import Layer
from src.network import Network
from src.neuron import Neuron


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "experiments" / "082_public_domain_text"
TOKENIZER_PATH = (
    ROOT / "experiments" / "078_character_tokenization" / "experiment.py"
)

TRAIN_PATH = DATA_DIR / "train_clean.txt"
HELD_OUT_PATH = DATA_DIR / "held_out_clean.txt"

HIDDEN_SIZE = 24
EPOCHS = 200
LEARNING_RATE = 0.5
SEED_CONTEXT_1 = 8301
SEED_CONTEXT_2 = 8502
MIN_HISTORY = 2


if not TRAIN_PATH.is_file() or not HELD_OUT_PATH.is_file():
    raise SystemExit(
        "Experiment 082 data is missing. Prepare the local book text first."
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
        part.strip()
        for part in text.split("\n\n")
        if part.strip()
    ]


def softmax(logits):
    maximum = max(logits)
    exponential_values = [
        math.exp(value - maximum) for value in logits
    ]
    denominator = sum(exponential_values)
    return [value / denominator for value in exponential_values]


def one_hot(token_id, vocabulary_size):
    values = [0.0] * vocabulary_size
    values[token_id] = 1.0
    return values


def encode_context(context, context_length, vocabulary_size):
    """Encode one character or an ordered pair as concatenated one-hot vectors."""

    token_ids = (
        (context,)
        if context_length == 1
        else tuple(context)
    )

    return [
        value
        for token_id in token_ids
        for value in one_hot(token_id, vocabulary_size)
    ]


def context_at(sequence, target_index, context_length):
    if context_length == 1:
        return sequence[target_index - 1]

    return tuple(
        sequence[target_index - context_length:target_index]
    )


def build_network(input_size, vocabulary_size, seed):
    rng = random.Random(seed)

    hidden_limit = math.sqrt(6.0 / input_size)
    hidden_neurons = []

    for _ in range(HIDDEN_SIZE):
        weights = [
            rng.uniform(-hidden_limit, hidden_limit)
            for _ in range(input_size)
        ]
        hidden_neurons.append(
            Neuron(weights, 0.0, activation="relu")
        )

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

    return Network(
        layers=[
            Layer(neurons=hidden_neurons),
            Layer(neurons=output_neurons),
        ]
    )


def make_transition_counts(sequences, context_length):
    """Count targets after at least two preceding characters."""

    counts_by_context = {}
    global_counts = Counter()
    total_targets = 0

    for sequence in sequences:
        for index in range(MIN_HISTORY, len(sequence)):
            context = context_at(
                sequence, index, context_length
            )
            target = sequence[index]

            if context not in counts_by_context:
                counts_by_context[context] = Counter()

            counts_by_context[context][target] += 1
            global_counts[target] += 1
            total_targets += 1

    return counts_by_context, global_counts, total_targets


def all_evaluation_contexts(sequences, context_length):
    contexts = set()

    for sequence in sequences:
        for index in range(MIN_HISTORY, len(sequence)):
            contexts.add(
                context_at(sequence, index, context_length)
            )

    return contexts


def probabilities_for_contexts(
    network, contexts, context_length, vocabulary_size
):
    result = {}

    for context in sorted(contexts):
        inputs = encode_context(
            context, context_length, vocabulary_size
        )
        result[context] = softmax(network.forward(inputs))

    return result


def evaluate_probabilities(
    sequences, context_length, probabilities_by_context
):
    total_loss = 0.0
    correct = 0
    total = 0

    for sequence in sequences:
        for index in range(MIN_HISTORY, len(sequence)):
            context = context_at(
                sequence, index, context_length
            )
            target = sequence[index]
            probabilities = probabilities_by_context[context]

            total_loss -= math.log(
                max(probabilities[target], 1e-15)
            )

            prediction = max(
                range(len(probabilities)),
                key=lambda token: (
                    probabilities[token], -token
                ),
            )
            correct += prediction == target
            total += 1

    if total == 0:
        raise ValueError("No targets available for evaluation.")

    return total_loss / total, correct, total


def full_batch_gradient(
    network,
    counts_by_context,
    total_targets,
    context_length,
    vocabulary_size,
):
    """Accumulate gradients exactly, weighting each context by its counts."""

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

    for context, target_counts in sorted(
        counts_by_context.items()
    ):
        context_total = sum(target_counts.values())

        inputs = encode_context(
            context, context_length, vocabulary_size
        )
        probabilities = softmax(network.forward(inputs))

        for target, count in target_counts.items():
            total_loss -= (
                count / total_targets
            ) * math.log(max(probabilities[target], 1e-15))

        output_gradients = [
            (
                probabilities[target]
                * context_total
                - target_counts.get(target, 0)
            ) / total_targets
            for target in range(vocabulary_size)
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

    for layer_index, layer in enumerate(network.layers):
        for neuron_index, neuron in enumerate(layer.neurons):
            neuron.weight_gradients = weight_sums[
                layer_index
            ][neuron_index]
            neuron.bias_gradient = bias_sums[
                layer_index
            ][neuron_index]

    return total_loss


def gradient_check(network, context, target, context_length,
                   vocabulary_size):
    inputs = encode_context(
        context, context_length, vocabulary_size
    )
    probabilities = softmax(network.forward(inputs))

    output_gradients = list(probabilities)
    output_gradients[target] -= 1.0
    network.backward(output_gradients)

    hidden_outputs = [
        neuron.activate(neuron.raw_output)
        for neuron in network.layers[0].neurons
    ]
    active_indices = [
        i for i, value in enumerate(hidden_outputs)
        if value > 0.0
    ]

    if not active_indices:
        return False, float("nan"), float("nan")

    hidden_index = active_indices[0]
    output_neuron = network.layers[-1].neurons[target]
    analytic = output_neuron.weight_gradients[hidden_index]
    original_weight = output_neuron.weights[hidden_index]

    def loss_for_current_weight():
        current_probabilities = softmax(network.forward(inputs))
        return -math.log(
            max(current_probabilities[target], 1e-15)
        )

    epsilon = 1e-5
    output_neuron.weights[hidden_index] = original_weight + epsilon
    loss_plus = loss_for_current_weight()

    output_neuron.weights[hidden_index] = original_weight - epsilon
    loss_minus = loss_for_current_weight()

    output_neuron.weights[hidden_index] = original_weight
    numerical = (loss_plus - loss_minus) / (2.0 * epsilon)

    return (
        math.isclose(
            analytic, numerical, rel_tol=1e-4, abs_tol=1e-5
        ),
        analytic,
        numerical,
    )


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


def train_model(
    model_name,
    network,
    counts_by_context,
    train_sequences,
    held_out_sequences,
    context_length,
    vocabulary_size,
    total_targets,
):
    train_contexts = all_evaluation_contexts(
        train_sequences, context_length
    )
    initial_probabilities = probabilities_for_contexts(
        network, train_contexts, context_length, vocabulary_size
    )
    initial_loss, _, _ = evaluate_probabilities(
        train_sequences, context_length, initial_probabilities
    )

    print(f"\nTraining neural context-{context_length} model")
    print(f"Distinct training contexts: {len(counts_by_context)}")
    print(f"Initial training cross-entropy: {initial_loss:.4f}")

    for epoch in range(1, EPOCHS + 1):
        full_batch_gradient(
            network,
            counts_by_context,
            total_targets,
            context_length,
            vocabulary_size,
        )
        network.update(LEARNING_RATE)

        if epoch == 1 or epoch % 50 == 0 or epoch == EPOCHS:
            print(f"  {model_name} epoch={epoch:03d}", flush=True)

    trained_train_probabilities = probabilities_for_contexts(
        network, train_contexts, context_length, vocabulary_size
    )
    final_train_loss, train_correct, train_total = (
        evaluate_probabilities(
            train_sequences,
            context_length,
            trained_train_probabilities,
        )
    )

    held_contexts = all_evaluation_contexts(
        held_out_sequences, context_length
    )
    held_probabilities = probabilities_for_contexts(
        network, held_contexts, context_length, vocabulary_size
    )
    held_loss, held_correct, held_total = evaluate_probabilities(
        held_out_sequences, context_length, held_probabilities
    )

    return (
        initial_loss,
        final_train_loss,
        train_correct,
        train_total,
        held_loss,
        held_correct,
        held_total,
        held_probabilities,
    )


def count_model_predictions(
    sequences, context_length, primary_counts,
    context1_counts, global_counts
):
    global_prediction = max(
        global_counts,
        key=lambda token: (global_counts[token], -token),
    )

    correct = 0
    total = 0

    for sequence in sequences:
        for index in range(MIN_HISTORY, len(sequence)):
            target = sequence[index]

            if context_length == 1:
                context = sequence[index - 1]
                counts = primary_counts.get(context)
            else:
                context = tuple(sequence[index - 2:index])
                counts = primary_counts.get(context)

                if not counts:
                    counts = context1_counts.get(sequence[index - 1])

            if counts:
                prediction = max(
                    counts,
                    key=lambda token: (counts[token], -token),
                )
            else:
                prediction = global_prediction

            correct += prediction == target
            total += 1

    return correct, total


def main():
    train_paragraphs = read_paragraphs(TRAIN_PATH)
    held_out_paragraphs = read_paragraphs(HELD_OUT_PATH)

    train_text = "".join(train_paragraphs)
    tokenizer = tokenizer_module.CharacterTokenizer(train_text)
    vocabulary_size = tokenizer.vocab_size

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

    train_counts_1, global_counts, train_targets = (
        make_transition_counts(train_sequences, 1)
    )
    train_counts_2, global_counts_2, train_targets_2 = (
        make_transition_counts(train_sequences, 2)
    )

    held_out_targets = sum(
        max(0, len(sequence) - MIN_HISTORY)
        for sequence in held_out_sequences
    )

    print("=" * 76)
    print("EXPERIMENT 085 — NEURAL TWO-CHARACTER CONTEXT")
    print("=" * 76)
    print(f"Vocabulary size: {vocabulary_size}")
    print(f"Training targets: {train_targets}")
    print(f"Held-out targets: {held_out_targets}")

    check(
        "Training and held-out data are present",
        bool(train_sequences) and bool(held_out_sequences),
    )
    check(
        "All held-out characters exist in training vocabulary",
        all(
            token != tokenizer_module.UNK_ID
            for sequence in held_out_sequences
            for token in sequence
        ),
    )
    check(
        "Both count models use the same training targets",
        train_targets == train_targets_2
        and global_counts == global_counts_2,
    )
    check(
        "Target totals match paragraph-local counting",
        train_targets
        == sum(
            max(0, len(sequence) - MIN_HISTORY)
            for sequence in train_sequences
        ),
    )

    # Softmax test.
    example_probs = softmax([0.5, -1.0, 2.0])
    check(
        "Softmax probabilities sum to one",
        math.isclose(sum(example_probs), 1.0, rel_tol=1e-12)
        and all(0.0 <= p <= 1.0 for p in example_probs),
    )

    # Neural context-1 model: same target positions as context-2.
    network_1 = build_network(
        vocabulary_size, vocabulary_size, SEED_CONTEXT_1
    )
    (
        initial_1, final_1,
        train_correct_1, train_total_1,
        held_loss_1, correct_1, total_1, probs_1,
    ) = train_model(
        "context-1",
        network_1,
        train_counts_1,
        train_sequences,
        held_out_sequences,
        1,
        vocabulary_size,
        train_targets,
    )

    # Neural context-2 model: concatenate two one-hot input vectors.
    network_2 = build_network(
        vocabulary_size * 2, vocabulary_size, SEED_CONTEXT_2
    )

    gradient_context = next(
        (
            tuple(sequence[0:2]), sequence[2]
        )
        for sequence in train_sequences
        if len(sequence) >= 3
    )
    gradient_ok, analytic, numerical = gradient_check(
        network_2,
        gradient_context[0],
        gradient_context[1],
        2,
        vocabulary_size,
    )

    check(
        "Two-character backpropagation matches numerical gradient",
        gradient_ok,
    )
    print(
        f"  Gradient check: analytic={analytic:.6g}, "
        f"numerical={numerical:.6g}"
    )

    (
        initial_2, final_2,
        train_correct_2, train_total_2,
        held_loss_2, correct_2, total_2, probs_2,
    ) = train_model(
        "context-2",
        network_2,
        train_counts_2,
        train_sequences,
        held_out_sequences,
        2,
        vocabulary_size,
        train_targets_2,
    )

    check(
        "All trained parameters remain finite",
        parameters_are_finite(network_1)
        and parameters_are_finite(network_2),
    )
    check(
        "Both models reduce training cross-entropy",
        final_1 < initial_1 and final_2 < initial_2,
    )
    check(
        "Both models evaluate every held-out target",
        total_1 == held_out_targets and total_2 == held_out_targets,
    )
    check(
        "Neural predictions use valid vocabulary IDs",
        all(
            0 <= max(
                range(vocabulary_size),
                key=lambda token: (distribution[token], -token),
            ) < vocabulary_size
            for distribution in list(probs_1.values())
            + list(probs_2.values())
        ),
    )

    held_count_1, count_total_1 = count_model_predictions(
        held_out_sequences,
        1,
        train_counts_1,
        train_counts_1,
        global_counts,
    )
    held_count_2, count_total_2 = count_model_predictions(
        held_out_sequences,
        2,
        train_counts_2,
        train_counts_1,
        global_counts,
    )

    baseline_prediction = max(
        global_counts,
        key=lambda token: (global_counts[token], -token),
    )
    baseline_correct = sum(
        target == baseline_prediction
        for sequence in held_out_sequences
        for target in sequence[MIN_HISTORY:]
    )

    print()
    print("-" * 76)
    print("HELD-OUT RESULTS — SAME TARGET POSITIONS")
    print("-" * 76)
    print(
        f"Most-common-character baseline: "
        f"{baseline_correct}/{held_out_targets} "
        f"({baseline_correct / held_out_targets:.2%})"
    )
    print(
        f"Count-based context-1: {held_count_1}/{count_total_1} "
        f"({held_count_1 / count_total_1:.2%})"
    )
    print(
        f"Count-based context-2: {held_count_2}/{count_total_2} "
        f"({held_count_2 / count_total_2:.2%})"
    )
    print(
        f"Neural context-1: held-out={correct_1}/{total_1} "
        f"({correct_1 / total_1:.2%}), "
        f"train_accuracy={train_correct_1 / train_total_1:.2%}, "
        f"train_loss={final_1:.4f}, held_out_loss={held_loss_1:.4f}"
    )
    print(
        f"Neural context-2: held-out={correct_2}/{total_2} "
        f"({correct_2 / total_2:.2%}), "
        f"train_accuracy={train_correct_2 / train_total_2:.2%}, "
        f"train_loss={final_2:.4f}, held_out_loss={held_loss_2:.4f}"
    )

    print()
    print("Limitations:")
    print("- Both neural models use one small hidden layer.")
    print("- The comparison uses one random initialization per model.")
    print("- The dataset is later chapters of one book.")
    print("- Next-character accuracy does not demonstrate comprehension.")


if __name__ == "__main__":
    main()
