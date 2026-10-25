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

CORPUS = tokenizer_module.CORPUS


class CharacterBigramModel:
    """Learn next-character counts from adjacent character pairs."""

    def __init__(self):
        self.transitions = {}
        self.global_next_counts = Counter()
        self.training_token_count = 0

    def fit(self, token_ids):
        """Count which token follows each preceding token."""

        if len(token_ids) < 2:
            raise ValueError(
                "Training data needs at least two tokens."
            )

        self.transitions = {}
        self.global_next_counts = Counter()
        self.training_token_count = len(token_ids)

        for current_id, next_id in zip(
            token_ids[:-1], token_ids[1:]
        ):
            if current_id not in self.transitions:
                self.transitions[current_id] = Counter()

            self.transitions[current_id][next_id] += 1
            self.global_next_counts[next_id] += 1

    def probabilities(self, current_id):
        """Return next-token probabilities for one context."""

        counts = self.transitions.get(current_id)

        # If the context is unseen, fall back to overall
        # next-character frequencies.
        if not counts:
            counts = self.global_next_counts

        total = sum(counts.values())

        if total == 0:
            raise ValueError("The model has not been trained.")

        return {
            token_id: count / total
            for token_id, count in counts.items()
        }

    def predict_next(self, current_id):
        """Choose the most frequently observed next token."""

        probabilities = self.probabilities(current_id)

        # Break ties consistently by choosing the smallest ID.
        return max(
            probabilities,
            key=lambda token_id: (
                probabilities[token_id],
                -token_id,
            ),
        )

    def generate(self, prefix_ids, new_token_count):
        """Extend a prefix using greedy next-character prediction."""

        if not prefix_ids:
            raise ValueError("Generation requires a nonempty prefix.")

        if new_token_count < 0:
            raise ValueError("Token count cannot be negative.")

        result = list(prefix_ids)

        for _ in range(new_token_count):
            result.append(self.predict_next(result[-1]))

        return result


def check(name, condition):
    """Print a test result and stop if it fails."""

    if not condition:
        raise AssertionError(f"FAILED: {name}")

    print(f"[PASS] {name}")


def main():
    tokenizer = tokenizer_module.CharacterTokenizer(CORPUS)
    token_ids = tokenizer.encode(CORPUS)

    model = CharacterBigramModel()
    model.fit(token_ids)

    # Test 1: Every adjacent pair contributes one observation.
    transition_total = sum(
        sum(counts.values())
        for counts in model.transitions.values()
    )

    check(
        "All adjacent character pairs are counted",
        transition_total == len(token_ids) - 1,
    )

    # Test 2: Probabilities for a context sum to one.
    example_context = token_ids[0]
    example_probabilities = model.probabilities(example_context)

    check(
        "Next-character probabilities sum to one",
        math.isclose(
            sum(example_probabilities.values()),
            1.0,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ),
    )

    # Test 3: Predictions always refer to known token IDs.
    prediction = model.predict_next(example_context)

    check(
        "Prediction returns a valid vocabulary ID",
        prediction in tokenizer.id_to_char,
    )

    # Test 4: An unseen context uses global frequencies.
    unknown_prediction = model.predict_next(
        tokenizer_module.UNK_ID
    )

    check(
        "Unseen contexts use the fallback distribution",
        unknown_prediction in model.global_next_counts,
    )

    # Test 5: Generation preserves the prefix and requested length.
    prefix = tokenizer.encode("larry")
    generated = model.generate(prefix, 30)

    check(
        "Generation preserves the prefix and adds 30 tokens",
        generated[:len(prefix)] == prefix
        and len(generated) == len(prefix) + 30,
    )

    # Evaluate predictions against the same sequence used for
    # training. This measures fit, not generalization.
    correct = sum(
        model.predict_next(token_ids[index - 1])
        == token_ids[index]
        for index in range(1, len(token_ids))
    )

    target_count = len(token_ids) - 1
    model_accuracy = correct / target_count

    # Baseline: predict the most common next character every time.
    most_common_next_id = max(
        model.global_next_counts,
        key=lambda token_id: (
            model.global_next_counts[token_id],
            -token_id,
        ),
    )

    baseline_correct = sum(
        token_id == most_common_next_id
        for token_id in token_ids[1:]
    )
    baseline_accuracy = baseline_correct / target_count

    print()
    print("=" * 72)
    print("EXPERIMENT 079 — CHARACTER BIGRAM MODEL")
    print("=" * 72)
    print(f"Training corpus characters: {len(CORPUS)}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Adjacent-pair observations: {transition_total}")
    print(
        "Distinct preceding characters with observed transitions: "
        f"{len(model.transitions)}"
    )

    for character in ("l", "r", " ", "t"):
        context_id = tokenizer.char_to_id[character]
        predicted_id = model.predict_next(context_id)
        predicted_character = tokenizer.decode([predicted_id])

        print(
            f"After {character!r}, predicted next character: "
            f"{predicted_character!r}"
        )

    print()
    print("Training-sequence prediction accuracy:")
    print(
        f"  Bigram model: {correct}/{target_count} "
        f"({model_accuracy:.1%})"
    )
    print(
        f"  Most-common-character baseline: "
        f"{baseline_correct}/{target_count} "
        f"({baseline_accuracy:.1%})"
    )

    print()
    print("Generated sample:")
    print(tokenizer.decode(generated))

    print()
    print("Limitations:")
    print("- The model considers only the immediately preceding character.")
    print("- Greedy generation always chooses the most frequent next character.")
    print("- Accuracy is measured on training text, not unseen text.")
    print("- This is a statistical model, not a neural network.")

    print()
    print("Conclusion:")
    print(
        "Larry now learns next-character patterns by counting "
        "observed character pairs."
    )


if __name__ == "__main__":
    main()
