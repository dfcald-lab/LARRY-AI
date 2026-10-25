import importlib.util
from collections import Counter
from pathlib import Path
import math


ROOT = Path.cwd()
EXPERIMENT_DIR = Path("experiments/081_larger_text_corpus")
TRAIN_PATH = EXPERIMENT_DIR / "train.txt"
HELD_OUT_PATH = EXPERIMENT_DIR / "held_out.txt"
TOKENIZER_PATH = Path(
    "experiments/078_character_tokenization/experiment.py"
)

if not TRAIN_PATH.is_file() or not HELD_OUT_PATH.is_file():
    raise SystemExit(
        "Corpus files are missing. Run this from the repository root."
    )

if not TOKENIZER_PATH.is_file():
    raise SystemExit("Experiment 078 tokenizer was not found.")

spec = importlib.util.spec_from_file_location(
    "exp078_tokenizer", TOKENIZER_PATH
)

if spec is None or spec.loader is None:
    raise SystemExit("Could not load the tokenizer.")

tokenizer_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tokenizer_module)

CONTEXT_LENGTHS = [1, 2, 3]


class ContextModel:
    """Count next-character patterns using up to N prior characters."""

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
        """Learn only from the sequences supplied for training."""

        self.context_counts = {
            length: {}
            for length in range(1, self.max_context + 1)
        }
        self.next_counts = Counter()

        for sequence in sequences:
            for index in range(1, len(sequence)):
                target = sequence[index]
                self.next_counts[target] += 1

                largest_context = min(self.max_context, index)

                for length in range(1, largest_context + 1):
                    context = tuple(sequence[index - length:index])
                    table = self.context_counts[length]

                    if context not in table:
                        table[context] = Counter()

                    table[context][target] += 1

    def probabilities(self, previous_ids):
        """Use the longest observed context, backing off if necessary."""

        largest_context = min(
            self.max_context,
            len(previous_ids),
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
            raise ValueError("The model has not been trained.")

        return (
            {
                token_id: count / total
                for token_id, count in self.next_counts.items()
            },
            0,
        )

    def predict_next(self, previous_ids):
        """Return the most frequent supported next character."""

        probabilities, context_used = self.probabilities(previous_ids)

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
        # Treat each sentence as a separate sequence.
        for index in range(1, len(sequence)):
            prediction, context_used = model.predict_next(
                sequence[:index]
            )

            correct += prediction == sequence[index]
            total += 1
            context_usage[context_used] += 1

    if total == 0:
        raise ValueError("No next-character targets to evaluate.")

    return correct, total, context_usage


def main():
    train_lines = TRAIN_PATH.read_text(encoding="utf-8").splitlines()
    held_out_lines = HELD_OUT_PATH.read_text(
        encoding="utf-8"
    ).splitlines()

    if not train_lines or not held_out_lines:
        raise SystemExit("Training and held-out files must not be empty.")

    if any(not line for line in train_lines + held_out_lines):
        raise SystemExit("Empty lines are not permitted in this corpus.")

    if set(train_lines) & set(held_out_lines):
        raise SystemExit(
            "A complete sentence appears in both datasets."
        )

    # Build vocabulary only from training text.
    training_text = "".join(train_lines)
    tokenizer = tokenizer_module.CharacterTokenizer(training_text)

    train_sequences = [
        tokenizer.encode(line) for line in train_lines
    ]
    held_out_sequences = [
        tokenizer.encode(line) for line in held_out_lines
    ]

    print("=" * 76)
    print("EXPERIMENT 081 — LARGER TEXT CORPUS")
    print("=" * 76)

    check(
        "Training and held-out sentences do not overlap",
        not (set(train_lines) & set(held_out_lines)),
    )

    check(
        "All held-out characters exist in the training vocabulary",
        all(
            token_id != tokenizer_module.UNK_ID
            for sequence in held_out_sequences
            for token_id in sequence
        ),
    )

    expected_transitions = sum(
        len(sequence) - 1 for sequence in train_sequences
    )

    models = {}

    for length in CONTEXT_LENGTHS:
        model = ContextModel(length)
        model.fit(train_sequences)
        models[length] = model

    actual_transitions = sum(
        sum(counts.values())
        for counts in models[1].context_counts[1].values()
    )

    check(
        "Training transitions are counted without crossing sentences",
        actual_transitions == expected_transitions,
    )

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

    fallback_probabilities, fallback_length = (
        models[3].probabilities([])
    )

    check(
        "Global fallback works with no preceding context",
        fallback_length == 0
        and math.isclose(
            sum(fallback_probabilities.values()),
            1.0,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ),
    )

    prediction, _ = models[3].predict_next(
        held_out_sequences[0][:5]
    )

    check(
        "Predictions use valid vocabulary IDs",
        prediction in tokenizer.id_to_char,
    )

    target_count = sum(
        len(sequence) - 1 for sequence in held_out_sequences
    )

    baseline_id = max(
        models[1].next_counts,
        key=lambda token_id: (
            models[1].next_counts[token_id],
            -token_id,
        ),
    )

    baseline_correct = sum(
        token_id == baseline_id
        for sequence in held_out_sequences
        for token_id in sequence[1:]
    )

    results = {}

    print()
    print("-" * 76)
    print("DATASET")
    print("-" * 76)
    print(f"Training sentences: {len(train_lines)}")
    print(f"Held-out sentences: {len(held_out_lines)}")
    print(f"Training characters: {sum(map(len, train_lines))}")
    print(f"Held-out characters: {sum(map(len, held_out_lines))}")
    print(f"Held-out prediction targets: {target_count}")
    print(f"Vocabulary size including UNK: {tokenizer.vocab_size}")

    print()
    print("-" * 76)
    print("HELD-OUT RESULTS")
    print("-" * 76)
    print(
        f"Most-common-character baseline: {baseline_correct}/"
        f"{target_count} ({baseline_correct / target_count:.1%})"
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
            "  Actual context used: "
            + ", ".join(
                f"{used} = {count}"
                for used, count in sorted(usage.items())
            )
        )

        check(
            f"Context {length} evaluation covered every target",
            total == target_count and sum(usage.values()) == total,
        )

    best_length = max(
        results,
        key=lambda length: (results[length], -length),
    )

    print()
    print("-" * 76)
    print("INTERPRETATION")
    print("-" * 76)
    print(f"Best measured context length: {best_length}")
    print(
        "This comparison measures accuracy on the included held-out "
        "sentences. It does not establish general performance on "
        "unrelated or naturally occurring text."
    )


if __name__ == "__main__":
    main()
