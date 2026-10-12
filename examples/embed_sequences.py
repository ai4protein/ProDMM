"""Print embedding shapes for small protein and DNA inputs."""

import argparse

import torch

from prodmm.encoder.modeling_prolm import ProLMForMaskedLM
from prodmm.encoder_tokenizer.unitokenizer import UniTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="AI4Protein/prodmm_encoder")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    tokenizer = UniTokenizer()
    model = ProLMForMaskedLM.from_pretrained(args.model).to(args.device).eval()
    inputs = tokenizer(
        [
            "<aas>MQKHVDTAV</aas>",
            "<cds>ATGCAAAAGCACGTGGACACCGCCGTT</cds>",
            "<ncds>TGCAATGCGCCGTT</ncds><aas>MQKHVDTAV</aas>",
        ]
    )
    with torch.inference_mode():
        outputs = model(
            input_ids=inputs["input_ids"].to(args.device),
            attention_mask=inputs["attention_mask"].to(args.device),
        )
    print("Embedding shape:", tuple(outputs.hidden_states.shape))
    print("Unpadded token counts:", inputs["length"].tolist())


if __name__ == "__main__":
    main()
