
from tokenizers import Tokenizer, models, pre_tokenizers, processors, trainers


def build_tokenizer(config):
    algorithm = config["algorithm"].lower()
    pre_type = config["pre_tokenizer"].lower()

    model_classes = {
        "bpe": models.BPE,
        "wordlevel": models.WordLevel
    }
    trainer_classes = {
        "bpe": trainers.BpeTrainer,
        "wordlevel": trainers.WordLevelTrainer
    }
    pre_classes = {
        "whitespace": pre_tokenizers.Whitespace,
        "bytelevel": pre_tokenizers.ByteLevel
    }

    if algorithm not in model_classes:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    if pre_type not in pre_classes:
        raise ValueError(f"Unknown pre-tokenizer: {pre_type}")

    model_kwargs = {"unk_token": "[UNK]", **config.get("model_kwargs", {})}
    pre_kwargs = dict(config.get("pre_tokenizer_kwargs", {}))
    trainer_kwargs = {
        "vocab_size": 32000,
        "special_tokens": ["[PAD]", "[UNK]"],
        **config.get("trainer_kwargs", {})
    }

    tokenizer = Tokenizer(model_classes[algorithm](**model_kwargs))
    tokenizer.pre_tokenizer = pre_classes[pre_type](**pre_kwargs)

    if pre_type == "bytelevel" and algorithm == "bpe":
        trainer_kwargs.setdefault("initial_alphabet", pre_tokenizers.ByteLevel.alphabet())
        tokenizer.post_processor = processors.ByteLevel(
            trim_offsets=True,
            add_prefix_space=pre_kwargs.get("add_prefix_space", True)
        )

    trainer = trainer_classes[algorithm](**trainer_kwargs)
    return tokenizer, trainer


# usage example
# config = {
#     "algorithm": "bpe",
#     "pre_tokenizer": "bytelevel",
#     "pre_tokenizer_kwargs": {"add_prefix_space": False},
#     "trainer_kwargs": {
#         "vocab_size": 32000,
#         "special_tokens": ["[PAD]", "[UNK]"]
#     }
# }

# tokenizer, trainer = build_tokenizer(config)