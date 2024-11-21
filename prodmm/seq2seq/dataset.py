from datasets import load_dataset, Features, Value

def get_dataset(
    file,
    streaming=True,
):
    dataset = load_dataset(
        "json",
        data_files=[
            file,
        ],
        features=Features(
            {
                "S": Value("string"), # Sequence
                "L": Value("int32"), # Length
                "T": Value("string"), # Target
            }
        ),
        split="train",
        streaming=streaming,
    )
    return dataset