from pathlib import Path


if not Path("experiments").is_dir():
    raise SystemExit(
        "Run this script from the LARRY-AI repository root."
    )


EXPERIMENT_NUMBER = "078"
EXPERIMENT_NAME = "Character Tokenization"

UNK_TOKEN = "<UNK>"
UNK_ID = 0

CORPUS = (
    "larry learns from text.\n"
    "text becomes tokens.\n"
    "tokens become numbers.\n"
    "larry learns one character at a time.\n"
)


class CharacterTokenizer:
    """Convert individual characters into stable integer IDs."""

    def __init__(self, training_text):
        if not isinstance(training_text, str):
            raise TypeError("Training text must be a string.")

        if not training_text:
            raise ValueError("Training text cannot be empty.")

        characters = sorted(set(training_text))

        # ID 0 is reserved for characters not seen during training.
        self.char_to_id = {
            character: index + 1
            for index, character in enumerate(characters)
        }

        self.id_to_char = {
            token_id: character
            for character, token_id in self.char_to_id.items()
        }

        self.id_to_char[UNK_ID] = UNK_TOKEN
        self.vocab_size = len(self.id_to_char)

    def encode(self, text):
        """Convert text into a list of integer token IDs."""

        if not isinstance(text, str):
            raise TypeError("Text to encode must be a string.")

        return [
            self.char_to_id.get(character, UNK_ID)
            for character in text
        ]

    def decode(self, token_ids):
        """Convert token IDs back into text."""

        characters = []

        for token_id in token_ids:
            if not isinstance(token_id, int):
                raise TypeError("Every token ID must be an integer.")

            if token_id not in self.id_to_char:
                raise ValueError(
                    f"Unknown token ID: {token_id}"
                )

            characters.append(self.id_to_char[token_id])

        return "".join(characters)


def check(name, condition):
    """Report a test result and stop if it fails."""

    if not condition:
        raise AssertionError(f"FAILED: {name}")

    print(f"[PASS] {name}")


def main():
    print("=" * 72)
    print("EXPERIMENT 078 — CHARACTER TOKENIZATION")
    print("=" * 72)

    tokenizer = CharacterTokenizer(CORPUS)

    encoded_corpus = tokenizer.encode(CORPUS)
    decoded_corpus = tokenizer.decode(encoded_corpus)

    # Test 1: Known text survives encoding and decoding.
    check(
        "Known text reconstructs exactly",
        decoded_corpus == CORPUS,
    )

    # Test 2: Vocabulary IDs do not depend on text order.
    reversed_tokenizer = CharacterTokenizer(CORPUS[::-1])

    check(
        "Vocabulary IDs are deterministic",
        tokenizer.char_to_id == reversed_tokenizer.char_to_id,
    )

    # Test 3: Unseen characters map to the unknown-token ID.
    new_text = "Larry 😊"
    new_ids = tokenizer.encode(new_text)

    check(
        "Unseen characters map to UNK",
        new_ids.count(UNK_ID) == 2,
    )

    check(
        "Unknown characters remain visible when decoded",
        tokenizer.decode(new_ids) == "<UNK>arry <UNK>",
    )

    # Test 4: Invalid token IDs are rejected.
    invalid_id_rejected = False

    try:
        tokenizer.decode([tokenizer.vocab_size])
    except ValueError:
        invalid_id_rejected = True

    check(
        "Invalid token IDs are rejected",
        invalid_id_rejected,
    )

    print()
    print("-" * 72)
    print("RESULTS")
    print("-" * 72)
    print(f"Corpus characters: {len(CORPUS)}")
    print(
        "Unique characters in training text: "
        f"{len(tokenizer.char_to_id)}"
    )
    print(f"Vocabulary size including UNK: {tokenizer.vocab_size}")
    print(f"Encoded sequence length: {len(encoded_corpus)}")
    print(f"Sample text: {CORPUS.splitlines()[0]!r}")
    print(
        "Sample token IDs: "
        f"{tokenizer.encode(CORPUS.splitlines()[0])}"
    )
    print(f"Unknown-character example: {new_text!r}")
    print(f"Encoded unknown example: {new_ids}")
    print(f"Decoded unknown example: {tokenizer.decode(new_ids)!r}")

    print()
    print("Conclusion:")
    print(
        "Text can now be represented as integers and reconstructed. "
        "Characters absent from the training vocabulary use UNK."
    )
    print(
        "This experiment implements tokenization, not language learning."
    )


if __name__ == "__main__":
    main()
