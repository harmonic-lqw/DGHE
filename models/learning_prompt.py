import torch
import clip

class LP(torch.nn.Module):
    def __init__(self, device, clip_model='ViT-B/32', init_template="", class_str=""):
        super(LP, self).__init__()

        self.device = device
        self.model, _ = clip.load(clip_model, device=self.device)

        template_text = init_template.format(class_str)
        len_learning_embedding = len(init_template.split(" ")) - 1

        self.tokens = clip.tokenize(template_text).to(self.device)  # (1, 77)

        with torch.no_grad():
            self.embedding = self.model.token_embedding(self.tokens).type(self.model.dtype) # (1, 77, 512)

        self.prefix = self.embedding[:, 0: 1, :]

        self.midfix = self.embedding[:, 1: 1 + len_learning_embedding, :]
        self.learning_vectors = torch.nn.Parameter(self.midfix)

        self.suffix = self.embedding[:, 1 + len_learning_embedding : , :]


    def forward(self, norm: bool = True):
        prompts = torch.cat(
            [
                self.prefix,
                self.learning_vectors,
                self.suffix
            ],
            dim=1,
        )

        prompts = prompts + self.model.positional_embedding.type(self.model.dtype)
        prompts = prompts.permute(1, 0, 2)
        prompts = self.model.transformer(prompts)
        prompts = prompts.permute(1, 0, 2)
        prompts = self.model.ln_final(prompts).type(self.model.dtype)
        prompts = prompts[torch.arange(prompts.shape[0]), self.tokens.argmax(dim=-1)] @ self.model.text_projection

        if norm:
            prompts = prompts / prompts.norm(dim=-1, keepdim=True)  # (79, 512)
        
        return prompts


        
    

