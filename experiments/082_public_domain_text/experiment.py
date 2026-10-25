import importlib.util
import math
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TOKENIZER_PATH = (
    ROOT.parent
    / "078_character_tokenization"
    / "experiment.py"
)

TRAIN_PATH = ROOT / "train_clean.txt"
HELD_OUT_PATH = ROOT / "held_out_clean.txt"
CONTEXT_LENGTHS = [1, 2, 3]


if not TRAIN_PATH.is_file() or not HELD_OUT_PATH.is_file():
    raise SystemExit(
        "Clean data files are missing. Run prepare_data.py first."
    )

spec = importlib.util.spec_from_file_location(
    "exp078_tokenizer", TOKENIZER_PATH
)

if spec is None or spec.loader is None:
    raise SystemExit("Could not load the Experiment 078 tokenizer.")

tokenizer_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tokenizer_module)


def read_paragraphs(path):
    text = path.read_text(encoding="utf-8")
    return [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]


def check(name, condition):
    if not condition:
        raise AssertionError(f"FAILED: {name}")
    print(f"[PASS] {name}")


class ContextModel:
    """A count-based next-character model with context backoff."""

    def __init__(self, max_context):
        if max_context < 1:
            raise ValueError("Context length must be at least one.")

        self.max_context = max_context
        self.context_counts = {}
        self.next_counts = Counter()

    def fit(self, sequences):
        self.context_counts = {
            length: {}
            for length in range(1, self.max_context + 1)
        }
        self.next_counts = Counter()

        for sequence in sequences:
            # Each sequence is one paragraph.
            for index in range(1, len(sequence)):
                target = sequence[index]
                self.next_counts[target] += 1

                largest_context = min(
                    self.max_context,
                    index,
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
        """Use the longest observed context, backing off as needed."""

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


def evaluate(model, sequences):
    correct = 0
    total = 0
    context_usage = Counter()

    for sequence in sequences:
        # Do not carry context from one paragraph into another.
        for index in range(1, len(sequence)):
            prediction, context_used = model.predict_next(
                sequence[:index]
            )

            correct += prediction == sequence[index]
            total += 1
            context_usage[context_used] += 1

    if total == 0:
        raise ValueError("No held-out targets are available.")

    return correct, total, context_usage


def main():
    train_paragraphs = read_paragraphs(TRAIN_PATH)
    held_out_paragraphs = read_paragraphs(HELD_OUT_PATH)

    if not train_paragraphs or not held_out_paragraphs:
        raise SystemExit("Training and held-out data cannot be empty.")

    train_text = TRAIN_PATH.read_text(encoding="utf-8")
    held_out_text = HELD_OUT_PATH.read_text(encoding="utf-8")

    # Build the vocabulary from training text only.
    tokenizer = tokenizer_module.CharacterTokenizer(train_text)

    train_sequences = [
        tokenizer.encode(paragraph)
        for paragraph in train_paragraphs
    ]
    held_out_sequences = [
        tokenizer.encode(paragraph)
        for paragraph in held_out_paragraphs
    ]

    print("=" * 76)
    print("EXPERIMENT 082 — PUBLIC-DOMAIN TEXT")
    print("Alice's Adventures in Wonderland")
    print("=" * 76)

    check(
        "Training and held-out paragraphs are present",
        bool(train_sequences) and bool(held_out_sequences),
    )

    check(
        "All held-out characters exist in training vocabulary",
        all(
            token_id != tokenizer_module.UNK_ID
            for sequence in held_out_sequences
            for token_id in sequence
        ),
    )

    training_targets = sum(
        len(sequence) - 1
        for sequence in train_sequences
    )
    held_out_targets = sum(
        len(sequence) - 1
        for sequence in held_out_sequences
    )

    models = {}

    for length in CONTEXT_LENGTHS:
        model = ContextModel(length)
        model.fit(train_sequences)
        models[length] = model

    recorded_transitions = sum(
        sum(counts.values())
        for counts in models[1].context_counts[1].values()
    )

    check(
        "Training counts do not cross paragraph boundaries",
        recorded_transitions == training_targets,
    )

    example_probabilities, _ = models[3].probabilities(
        held_out_sequences[0][:10]
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
        "Global fallback works without preceding context",
        fallback_length == 0
        and math.isclose(
            sum(fallback_probabilities.values()),
            1.0,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ),
    )

    prediction, _ = models[3].predict_next(
        held_out_sequences[0][:10]
    )

    check(
        "Predictions use valid vocabulary IDs",
        prediction in tokenizer.id_to_char,
    )

    print()
    print("-" * 76)
    print("DATASET")
    print("-" * 76)
    print("Training chapters: I–VIII")
    print("Held-out chapters: IX–XII")
    print(f"Training paragraphs: {len(train_paragraphs)}")
    print(f"Held-out paragraphs: {len(held_out_paragraphs)}")
    print(f"Training text characters: {len(train_text)}")
    print(f"Held-out text characters: {len(held_out_text)}")
    print(f"Training prediction targets: {training_targets}")
    print(f"Held-out prediction targets: {held_out_targets}")
    print(f"Vocabulary size including UNK: {tokenizer.vocab_size}")

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

    print()
    print("-" * 76)
    print("HELD-OUT RESULTS")
    print("-" * 76)
    print(
        f"Most-common-character baseline: "
        f"{baseline_correct}/{held_out_targets} "
        f"({baseline_correct / held_out_targets:.2%})"
    )

    results = {}

    for length in CONTEXT_LENGTHS:
        correct, total, usage = evaluate(
            models[length],
            held_out_sequences,
        )

        check(
            f"Context {length} evaluates all held-out targets",
            total == held_out_targets
            and sum(usage.values()) == held_out_targets,
        )

        accuracy = correct / total
        results[length] = accuracy

        print()
        print(
            f"Maximum context {length}: "
            f"{correct}/{total} ({accuracy:.2%})"
        )
        print(
            "  Actual context used: "
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
    print(f"Best measured context length: {best_length}")
    print(
        "This is a count-based character model evaluated on later "
        "chapters of one book. The chapter split is a useful holdout, "
        "but the result does not establish general language ability."
    )


if __name__ == "__main__":
    main()
