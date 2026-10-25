import importlib.util
import math
from collections import Counter
from pathlib import Path


SOURCE = Path(
    "experiments/078_character_tokenization/experiment.py"
)

if not SOURCE.is_file():
    raise SystemExit(
        "Run this script from the LARRY-AI repository root."
    )

spec = importlib.util.spec_from_file_location(
    "exp078_tokenizer", SOURCE
)

if spec is None or spec.loader is None:
    raise SystemExit("Could not load Experiment 078.")

tokenizer_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tokenizer_module)


TRAIN_LINES = [
    "larry learns from text.",
    "larry reads text.",
    "larry learns patterns.",
    "the model learns letters.",
    "the model reads text.",
    "letters form words.",
    "words form sentences.",
    "larry builds a model.",
    "the model predicts letters.",
    "patterns help models learn.",
]

HELD_OUT_LINES = [
    "larry reads letters.",
    "the model learns from text.",
    "words form patterns.",
    "larry builds words.",
    "the model predicts words.",
]

CONTEXT_LENGTHS = [1, 2, 3]


class ContextModel:
    """Predict characters using up to N preceding characters."""

    def __init__(self, max_context):
        if max_context < 1:
            raise ValueError("Context length must be at least one.")

        self.max_context = max_context
        self.context_counts = {
            length: {}
            for length in range(1, max_context + 1)
        }
        self.next_counts = Counter()

    def fit(self, sequences):
        """Learn transition counts from training sequences."""

        self.context_counts = {
            length: {}
            for length in range(1, self.max_context + 1)
        }
        self.next_counts = Counter()

        for sequence in sequences:
            for index in range(1, len(sequence)):
                target = sequence[index]
                self.next_counts[target] += 1

                largest_context = min(
                    self.max_context, index
                )

                for length in range(1, largest_context + 1):
                    context = tuple(
                        sequence[index - length:index]
                    )

                    table = self.context_counts[length]

                    if context not in table:
                        table[context] = Counter()

                    table[context][target] += 1

    def probabilities(self, previous_ids):
        """Return probabilities and the context length used.

        If a longer context is unseen, progressively try shorter
        contexts. With no matching context, use global frequencies.
        """

        largest_context = min(
            self.max_context, len(previous_ids)
        )

        for length in range(largest_context, 0, -1):
            context = tuple(previous_ids[-length:])
            counts = self.context_counts[length].get(context)

            if counts:
                total = sum(counts.values())

                return (
                    {
                        token_id: count / total
                        for token_id, count in counts.items()
                    },
                    length,
                )

        total = sum(self.next_counts.values())

        if total == 0:
            raise ValueError("Train the model before predicting.")

        return (
            {
                token_id: count / total
                for token_id, count in self.next_counts.items()
            },
            0,
        )

    def predict_next(self, previous_ids):
        """Choose the most frequent supported next character."""

        probabilities, context_used = self.probabilities(
            previous_ids
        )

        prediction = max(
            probabilities,
            key=lambda token_id: (
                probabilities[token_id],
                -token_id,
            ),
        )

        return prediction, context_used


def check(name, condition):
    if not condition:
        raise AssertionError(f"FAILED: {name}")

    print(f"[PASS] {name}")


def evaluate(model, sequences):
    correct = 0
    total = 0
    context_usage = Counter()

    for sequence in sequences:
        # The first character has no preceding character in its line.
        for index in range(1, len(sequence)):
            prediction, context_used = model.predict_next(
                sequence[:index]
            )

            correct += prediction == sequence[index]
            total += 1
            context_usage[context_used] += 1

    if total == 0:
        raise ValueError("There are no targets to evaluate.")

    return correct, total, context_usage


def main():
    # Build the vocabulary from training text only.
    tokenizer = tokenizer_module.CharacterTokenizer(
        "\n".join(TRAIN_LINES)
    )

    train_sequences = [
        tokenizer.encode(line)
        for line in TRAIN_LINES
    ]

    held_out_sequences = [
        tokenizer.encode(line)
        for line in HELD_OUT_LINES
    ]

    print("=" * 76)
    print("EXPERIMENT 080 — CONTEXT LENGTH COMPARISON")
    print("=" * 76)

    # Test 1: Held-out characters are representable by the training
    # vocabulary, so every target can be evaluated as a character.
    check(
        "All held-out characters exist in the training vocabulary",
        all(
            token_id != tokenizer_module.UNK_ID
            for sequence in held_out_sequences
            for token_id in sequence
        ),
    )

    # Test 2: No complete held-out sentence is in the training set.
    check(
        "Training and held-out sentences are separate",
        not (set(TRAIN_LINES) & set(HELD_OUT_LINES)),
    )

    models = {}

    for length in CONTEXT_LENGTHS:
        model = ContextModel(length)
        model.fit(train_sequences)
        models[length] = model

    # Test 3: All adjacent transitions are counted at context length 1.
    expected_transitions = sum(
        len(sequence) - 1
        for sequence in train_sequences
    )

    actual_transitions = sum(
        sum(counts.values())
        for counts in models[1].context_counts[1].values()
    )

    check(
        "Training transition counts are complete",
        actual_transitions == expected_transitions,
    )

    # Test 4: A prediction distribution is normalized.
    example_probabilities, _ = models[3].probabilities(
        held_out_sequences[0][:5]
    )

    check(
        "Next-character probabilities sum to one",
        math.isclose(
            sum(example_probabilities.values()),
            1.0,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ),
    )

    # Test 5: An empty context uses the global fallback.
    fallback_probabilities, fallback_length = (
        models[3].probabilities([])
    )

    check(
        "Global fallback works without a preceding context",
        fallback_length == 0
        and math.isclose(
            sum(fallback_probabilities.values()),
            1.0,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ),
    )

    # Test 6: Predictions stay inside the tokenizer vocabulary.
    prediction, _ = models[3].predict_next(
        held_out_sequences[0][:5]
    )

    check(
        "Predictions use valid vocabulary IDs",
        prediction in tokenizer.id_to_char,
    )

    print()
    print("-" * 76)
    print("DATASET")
    print("-" * 76)
    print(f"Training sentences: {len(TRAIN_LINES)}")
    print(f"Held-out sentences: {len(HELD_OUT_LINES)}")
    print(f"Training characters: {sum(map(len, TRAIN_LINES))}")
    print(
        "Held-out prediction targets: "
        f"{sum(len(sequence) - 1 for sequence in held_out_sequences)}"
    )
    print(f"Vocabulary size including UNK: {tokenizer.vocab_size}")

    # Baseline predicts the most common next character everywhere.
    global_model = models[1]
    baseline_id = max(
        global_model.next_counts,
        key=lambda token_id: (
            global_model.next_counts[token_id],
            -token_id,
        ),
    )

    baseline_correct = sum(
        token_id == baseline_id
        for sequence in held_out_sequences
        for token_id in sequence[1:]
    )

    total_targets = sum(
        len(sequence) - 1
        for sequence in held_out_sequences
    )

    print()
    print("-" * 76)
    print("HELD-OUT RESULTS")
    print("-" * 76)

    results = {}

    print(
        f"Most-common-character baseline: "
        f"{baseline_correct}/{total_targets} "
        f"({baseline_correct / total_targets:.1%})"
    )

    for length in CONTEXT_LENGTHS:
        correct, total, usage = evaluate(
            models[length], held_out_sequences
        )

        accuracy = correct / total
        results[length] = accuracy

        print()
        print(
            f"Maximum context {length}: "
            f"{correct}/{total} ({accuracy:.1%})"
        )

        print(
            "  Predictions using context length: "
            + ", ".join(
                f"{used} = {count}"
                for used, count in sorted(usage.items())
            )
        )

    best_length = max(
        results,
        key=lambda length: (results[length], -length),
    )

    print()
    print("-" * 76)
    print("INTERPRETATION")
    print("-" * 76)

    print(
        f"Best measured context length: {best_length}"
    )
    print(
        "These results describe this small held-out set only. "
        "They do not establish which context length is best "
        "for general language modeling."
    )


if __name__ == "__main__":
    main()
