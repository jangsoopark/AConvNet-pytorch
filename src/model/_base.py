import torch

import model.network


class Model(object):
    def __init__(self, **params):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = model.network.Network(
            classes=params.get('classes', 10),
            channels=params.get('channels', 1),
            dropout_rate=params.get('dropout_rate', 0.5)
        )
        self.net.to(self.device)

        self.lr = params.get('lr', 1e-3)
        self.lr_step = params.get('lr_step', [50])
        self.lr_decay = params.get('lr_decay', 0.1)

        self.lr_scheduler = None

        self.momentum = params.get('momentum', 0.9)
        self.weight_decay = params.get('weight_decay', 4e-3)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.net.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )

        if self.lr_decay:
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer=self.optimizer,
                milestones=self.lr_step,
                gamma=self.lr_decay
            )

    def optimize(self, x, y):
        p = self.net(x.to(self.device))
        loss = self.criterion(p, y.to(self.device))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def inference(self, x):
        return self.net(x.to(self.device))

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path))
        self.net.eval()
