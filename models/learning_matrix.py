import torch
import clip

class LM(torch.nn.Module):
    def __init__(self, device, clip_model='ViT-B/32', init_template="", class_str="", beta=0.8):
        super(LM, self).__init__()

        self.device = device
        self.beta = beta
        self.model, _ = clip.load(clip_model, device=self.device)

        self.template_text = init_template.format(class_str)
        tokens = clip.tokenize(self.template_text).to(self.device)
    
        self.text_features = self.model.encode_text(tokens).detach()
        with torch.no_grad():
            self.learning_matrix = torch.nn.Parameter(torch.zeros_like(self.text_features))


    def forward(self, norm: bool = True):
        features = self.text_features + self.beta * self.learning_matrix
        if norm:
            features = features / features.norm(dim=-1, keepdim=True)  # (79, 512)
        return features


        
    

