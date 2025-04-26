import torch
import torch.nn as nn
import dataclasses


@dataclasses.dataclass
class NeuraNILArgs:
    k: int = 5


class NeuraNIL(nn.Module):
    def __init__(self, learner, classifier, k, inner_lr=0.001):
        super(NeuraNIL, self).__init__()
        self.learner = learner
        self.classifier = classifier
        self.k = k
        self.loss_fn = nn.CrossEntropyLoss()

        self.inner_opt = torch.optim.Adam(
            list(self.classifier.parameters()), # Only update the inner loop weights
            lr=inner_lr,
        )


    def train_inner(self, support_x, support_y, support_lengths=None):
        # Train the learner on the support set for k steps
        for _ in range(self.k):
            self.inner_opt.zero_grad()
            support_pred = self.classifier(self.learner(support_x, support_lengths), support_lengths)
            support_loss = self.loss_fn(support_pred, support_y)
            support_loss.backward(retain_graph=True)
            self.inner_opt.step()


    def forward(self, support_x, support_y, query_x, support_lengths=None, query_lengths=None):
        # Forward pass the support set and update the classifier for k steps
        self.classifier.train()
        self.train_inner(support_x, support_y, support_lengths)
        # Forward pass the query set
        query_pred = self.classifier(self.learner(query_x, query_lengths), query_lengths)
        return query_pred