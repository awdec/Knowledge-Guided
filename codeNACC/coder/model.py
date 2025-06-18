import torch
import torch.nn as nn
import numpy as np

class RotatEModel(nn.Module):
    def __init__(self, num_entities, num_relations, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.ent_re = nn.Embedding(num_entities, embed_dim)
        self.ent_im = nn.Embedding(num_entities, embed_dim)
        self.rel_phase = nn.Embedding(num_relations, embed_dim)

        # 正确注册 pi_const 为 buffer（不作为参数保存，但能自动搬到 GPU/CPU）
        self.register_buffer("pi_const", torch.tensor(np.pi))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.ent_re.weight.data)
        nn.init.xavier_uniform_(self.ent_im.weight.data)
        nn.init.xavier_uniform_(self.rel_phase.weight.data)

    def get_entity_embedding(self, entity_idx):
        re = self.ent_re(entity_idx)
        im = self.ent_im(entity_idx)
        return torch.cat([re, im], dim=-1)

    def get_relation_embedding(self, relation_idx):
        return self.rel_phase(relation_idx)

# 保存权重
rotate_model = RotatEModel(num_entities=1698, num_relations=4, embed_dim=200)
torch.save(rotate_model.state_dict(), "rotate.ckpt")
print("✅ 已正确保存 rotate.ckpt")
