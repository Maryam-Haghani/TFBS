import os
import sys

hyena_dna_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../hyena-dna"))
sys.path.insert(0, hyena_dna_dir)

from standalone_hyenadna import CharacterTokenizer
from huggingface import HyenaDNAPreTrainedModel

class HyenaDNAModel:
    def __init__(self, logger, pretrained_model_name, device, use_head):
        self.logger = logger
        self.pretrained_model_name = pretrained_model_name
        self.use_head = use_head
        self.device = device

    @classmethod
    def get_tokenizer(self, max_length):
        return CharacterTokenizer(
            characters=['A', 'C', 'G', 'T', 'N'],
            model_max_length=max_length,
            add_special_tokens=False,
            padding_side='left',
        )

    @classmethod
    def load_pretrained_model(self):
        self.logger.log_message(f"Getting pretrained model '{self.pretrained_model_name}' on device '{self.device}'...")
        model = HyenaDNAPreTrainedModel.from_pretrained(
            "../models/checkpoints",
            self.pretrained_model_name,
            download=True,
            device=self.device,
            use_head=self.use_head,  # set to False to output embeddings, not classification
            n_classes=2
        )
        return model