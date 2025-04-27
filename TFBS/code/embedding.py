import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch

class Embedding:
    def __init__(self, is_finetuned, embedding_dir, device):
        self.model = model
        self. is_finetuned = is_finetuned
        self.embedding_dir = embedding_dir
        self.embedding_dir = os.path.join(self.embedding_dir, 'FineTuned' if is_finetuned else 'PreTrained')
        os.makedirs(self.embedding_dir, exist_ok=True)

        self.device = device

        hyena_model = HyenaDNAModel(pretrained_model_name=checkpoint_path, checkpoint_path=checkpoint_path,
                                    device=self.device, use_head=False)
        
        # Register a forward hook on the last LayerNorm (ln_f) to save the output of the LayerNorm layer as hidden_embeddinfs
        if self.is_finetuned:
            self.model = hyena_model.load_saved_model(model_path)
            self.embeddings_hook = self.model.backbone.ln_f.register_forward_hook(self._save_layernorm_output)
            
        else:
            self.model = hyena_model.load_pretrained_model()
            self.embeddings_hook = None

    def _save_layernorm_output(self, module, input, output):
        # hook captures the output of the last LayerNorm (before the classification head)
        x = output.detach().cpu().numpy()
        self.hidden_embeddings.append(output.detach().cpu().numpy())  # Shape [seq_len, hidden_dim]

    def _save(self):
        # Save in a single .npz file with named keys
        hidden_embedding_path = os.path.join(self.embedding_dir_type, f'hidden.npz')
        np.savez(hidden_embedding_path, self.hidden_embeddings, labels=self.labels)
        print(f"Hidden {self.type} embeddings saved at {hidden_embedding_path}")

        if self.is_finetuned:
            logits_embedding_path = os.path.join(self.embedding_dir_type, f'logits.npz')
            np.savez(logits_embedding_path, self.logit_embeddings, labels=self.labels)
            print(f"Logits {self.type} embeddings saved at {logits_embedding_path}")

    def _scatter_plot(self, emb, plot_name):
        labels_index = {0: "negative", 1: "positive"}
        plt.figure(figsize=(10, 7))

        for label in np.unique(self.labels):
            indices = self.labels == label
            plt.scatter(emb[indices, 0], emb[indices, 1], label=f"Class {labels_index[label]}", alpha=0.6)

        plt.title(f"Scatter plot of HyenaDNA {self.type} Embeddings: Positive vs Negative")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.embedding_dir_type, plot_name), dpi=300, bbox_inches="tight")

    def _tsne_plot(self, tsne, emb, plot_name):
        labels_index = {0: "negative", 1: "positive"}
        embeddings_2d = tsne.fit_transform(emb)

        plt.figure(figsize=(10, 7))
        for label in np.unique(self.labels):
            indices = self.labels == label
            plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], label=f"Class {labels_index[label]}",
                        alpha=0.6)

        plt.title(f"Average t-SNE of HyenaDNA {self.type} Embeddings: Positive vs Negative")
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.embedding_dir_type, plot_name), dpi=300, bbox_inches="tight")

    def generate(self, dataset, type):
        self.embedding_dir_type = os.path.join(self.embedding_dir, type)
        os.makedirs(self.embedding_dir_type, exist_ok=True)
        
        self.hidden_embeddings = []
        self.logit_embeddings = []
        self.labels = []

        self.type = type

        self.model.eval()
    
        with torch.no_grad():
            # for _, row in df.iterrows():
            for input_ids, label in dataset:

                self.labels.append(label.item())

                input_ids = input_ids.unsqueeze(0).to(self.device)  # add batch dimension and move to device
    
                # Get embeddings from the final layer output of the model
                output = self.model(input_ids)


                if self.is_finetuned: # extract the output from the head layer as logit_embeddings
                    logit_embeddings = final_embeddings = output.squeeze(0).cpu().numpy()  # Shape [2]
                    self.logit_embeddings.append(logit_embeddings)

                else:
                    # use the final hidden state output directly as hidden_embeddings
                    hidden_embeddings = output.squeeze(0).cpu().numpy() #  Shape [seq_len, hidden_dim] for pretrained model
                    self.hidden_embeddings.append(hidden_embeddings)

        self.hidden_embeddings = np.array(self.hidden_embeddings)
        print(f"Hidden {self.type} embeddings shape:", self.hidden_embeddings.shape)

        # Remove the hook after collecting embeddings (only if fine-tuned model)
        if self.is_finetuned:
            # self.embeddings_hook.remove()
            self.hidden_embeddings = self.hidden_embeddings.squeeze(1)

            self.logit_embeddings = np.array(self.logit_embeddings)

        print(f"Hidden {self.type} embeddings shape:", self.hidden_embeddings.shape)

        self.labels = np.array(self.labels)
        print(f"{self.type} labels shape:", self.labels.shape)

        self._save()


    def plot_tsne(self):
        tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)

        # average across sequence length - Resulting shape: (data_length, residue_dim)
        embeddings_avg = self.hidden_embeddings.mean(axis=1)
        print(f"Shape of averaged {self.type} embeddings: {embeddings_avg.shape}")
        self._tsne_plot(tsne, embeddings_avg, f"average-tsne.png")

        # # flatten embeddings - Resulting shape: (data length, seq_length*residue_dim)
        # embeddings_flat = self.hidden_embeddings.reshape(self.hidden_embeddings.shape[0], -1)
        # print(f"Shape of flattened embeddings: {embeddings_flat.shape}")
        # self._tsne_plot(tsne, embeddings_flat, f"flattened-tsne-finetune_{self.is_finetuned}.png")

        if self.is_finetuned:
            self._tsne_plot(tsne, self.logit_embeddings, f"logits-tsne.png")
            self._scatter_plot(self.logit_embeddings, f"logits-scatter.png")
