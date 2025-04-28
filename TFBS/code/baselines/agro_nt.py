from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model

class AgroNTModel:
    def __init__(self, logger, pretrained_model_name, device):
        self.logger = logger
        self.pretrained_model_name = pretrained_model_name
        self.device = device

    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.pretrained_model_name)

    def load_pretrained_model(self, finetune_type):
        self.logger.log_message(f"Getting pretrained model '{self.pretrained_model_name}' on device '{self.device}'...")

        model = (AutoModelForSequenceClassification
                 .from_pretrained(self.pretrained_model_name, num_labels=2))
        model = model.to(self.device)
        return self._add_LoRA_params(model) if finetune_type == "LoRA" else model

    def _add_LoRA_params(self, model):
        return model
        # from peft import LoraConfig, TaskType, get_peft_model
        #
        # self.logger.log_message(f"Setting up LoRA config...")
        #
        # peft_config = LoraConfig(
        #     task_type=TaskType.SEQ_CLS, inference_mode=False, r=1, lora_alpha=32, lora_dropout=0.1,
        #     target_modules=["query", "value"],
        #     # modules_to_save=["intermediate"] # modules that are not frozen and updated during the training
        # )
        #
        # lora_classifier = get_peft_model(model, peft_config)  # transform classifier into a peft model
        # lora_classifier.print_trainable_parameters()
        # lora_classifier.to(self.device)