import torch
import torch.nn as nn
import torchvision.models as tv_models
import numpy as np
from torch.utils.data import DataLoader, TensorDataset



# ─────────────────────────────────────────────────────────────────────────────
# ClassDenseNet — DenseNet-121 classifier for 96x96 histopathology patches
# ─────────────────────────────────────────────────────────────────────────────


class ClassResNet50(nn.Module):
    """
    ResNet-50 classifier following the DomainBed featurizer + classifier layout
    (Gulrajani & Lopez-Paz, 2021): ImageNet-pretrained backbone, BatchNorm
    frozen in eval mode (and BN parameters with requires_grad=False), final FC
    replaced with a fresh linear head of size num_classes.

    Interface mirrors ClassDenseNet:
        forward(x, extract_features=False) -> logits  (or (logits, features))
        predict(x), predict_proba(x), get_embeddings(x)
        z_dim = 2048
    """
    def __init__(self, num_classes, device,
                 use_dropout=False, pretrained=True, infer_batch_size=128):
        super().__init__()
        self.device = device
        self.use_dropout = use_dropout
        self.z_dim = 2048
        self.infer_batch_size = infer_batch_size

        base = tv_models.resnet50(pretrained=pretrained)
        self.features = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
            base.avgpool,
        )
        self.dropout = nn.Dropout(p=0.5) if use_dropout else nn.Identity()
        self.classifier = nn.Linear(self.z_dim, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

        self.to(device)
        self._freeze_bn()

    def _freeze_bn(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        # Always keep BN in eval mode (DomainBed convention).
        self._freeze_bn()
        return self

    def _embed(self, x):
        z = self.features(x)
        return z.view(z.size(0), -1)

    def forward(self, x, extract_features=False):
        z = self._embed(x)
        z_drop = self.dropout(z)
        logits = self.classifier(z_drop)
        if extract_features:
            return logits, z
        return logits

    def _to_tensor(self, inputs):
        if isinstance(inputs, np.ndarray):
            return torch.from_numpy(inputs).float()
        return inputs.float()

    def _batched(self, inputs, fn):
        self.eval()
        t = self._to_tensor(inputs)
        out = []
        with torch.no_grad():
            for s in range(0, len(t), self.infer_batch_size):
                b = t[s : s + self.infer_batch_size].to(self.device)
                out.append(fn(b))
        return np.concatenate(out, axis=0)

    def predict_proba(self, inputs):
        return self._batched(
            inputs,
            lambda b: torch.softmax(self.forward(b), dim=1).cpu().numpy(),
        )

    def predict(self, inputs):
        return self._batched(
            inputs,
            lambda b: self.forward(b).argmax(dim=1).cpu().numpy(),
        )

    def get_embeddings(self, inputs):
        def emb_fn(b):
            _, z = self.forward(b, extract_features=True)
            return z.cpu().numpy()
        return self._batched(inputs, emb_fn)



class _CifarBasicBlock(nn.Module):
    """Standard ResNet BasicBlock used in CIFAR ResNets."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return torch.relu(out)


class ClassResNet18Cifar(nn.Module):
    """
    ResNet-18 adapted for CIFAR-10 (32x32) following Hendrycks & Dietterich,
    "Benchmarking Neural Network Robustness to Common Corruptions and
    Perturbations" (ICLR 2019). Differences from torchvision's ImageNet
    ResNet-18: 3x3 stride-1 stem, no initial maxpool.

    Interface mirrors ClassResNet50:
        forward(x, extract_features=False) -> logits  (or (logits, features))
        predict(x), predict_proba(x), get_embeddings(x)
        z_dim = 512
    """
    def __init__(self, num_classes, device, use_dropout=False, infer_batch_size=256):
        super().__init__()
        self.device           = device
        self.use_dropout      = use_dropout
        self.z_dim            = 512
        self.infer_batch_size = infer_batch_size
        self.in_planes        = 64

        self.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1     = nn.BatchNorm2d(64)
        self.layer1  = self._make_layer(64,  2, stride=1)
        self.layer2  = self._make_layer(128, 2, stride=2)
        self.layer3  = self._make_layer(256, 2, stride=2)
        self.layer4  = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.5) if use_dropout else nn.Identity()
        self.classifier = nn.Linear(self.z_dim, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

        self.to(device)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(_CifarBasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * _CifarBasicBlock.expansion
        return nn.Sequential(*layers)

    def _embed(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        return out.view(out.size(0), -1)

    def forward(self, x, extract_features=False):
        z      = self._embed(x)
        z_drop = self.dropout(z)
        logits = self.classifier(z_drop)
        if extract_features:
            return logits, z
        return logits

    def _to_tensor(self, inputs):
        if isinstance(inputs, np.ndarray):
            return torch.from_numpy(inputs).float()
        return inputs.float()

    def _batched(self, inputs, fn):
        self.eval()
        t = self._to_tensor(inputs)
        out = []
        with torch.no_grad():
            for s in range(0, len(t), self.infer_batch_size):
                b = t[s : s + self.infer_batch_size].to(self.device)
                out.append(fn(b))
        return np.concatenate(out, axis=0)

    def predict_proba(self, inputs):
        return self._batched(inputs,
            lambda b: torch.softmax(self.forward(b), dim=1).cpu().numpy())

    def predict(self, inputs):
        return self._batched(inputs,
            lambda b: self.forward(b).argmax(dim=1).cpu().numpy())

    def get_embeddings(self, inputs):
        def emb_fn(b):
            _, z = self.forward(b, extract_features=True)
            return z.cpu().numpy()
        return self._batched(inputs, emb_fn)



class ClassDenseNet(nn.Module):
    """
    DenseNet-121 classifier for 96x96 histopathology patches (Camelyon17).

    Interface identical to ClassNNet / ClassCNN:
        forward(x, extract_features=False)
        predict_proba(inputs)            -> (N, K)     numpy float32
        predict(inputs)                  -> (N,)        numpy int64
        get_embeddings(inputs)           -> (N, 1024)   numpy float32
        z_dim = 1024
    """

    def __init__(self, num_classes, device,
                 use_dropout=False, pretrained=False, infer_batch_size=256):
        super().__init__()
        self.device           = device
        self.use_dropout      = use_dropout
        self.z_dim            = 1024
        self.infer_batch_size = infer_batch_size

        base           = tv_models.densenet121(pretrained=pretrained)
        self.features  = base.features
        self.relu      = nn.ReLU(inplace=True)
        self.avg_pool  = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout   = nn.Dropout(p=0.5) if use_dropout else nn.Identity()

        self.classifier = nn.Linear(self.z_dim, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

        self.to(device)

    def _embed(self, x):
        z = self.features(x)
        z = self.relu(z)
        z = self.avg_pool(z)
        return z.view(z.size(0), -1)

    def forward(self, x, extract_features=False):
        z      = self._embed(x)
        z_drop = self.dropout(z)
        logits = self.classifier(z_drop)
        if extract_features:
            return logits, z
        return logits

    def _to_tensor(self, inputs):
        if isinstance(inputs, np.ndarray):
            return torch.from_numpy(inputs).float()
        return inputs.float()

    def predict_proba(self, inputs):
        self.eval()
        t, results = self._to_tensor(inputs), []
        with torch.no_grad():
            for s in range(0, len(t), self.infer_batch_size):
                b = t[s : s + self.infer_batch_size].to(self.device)
                p = torch.softmax(self.forward(b), dim=1).cpu().numpy()
                results.append(p)
        return np.concatenate(results, axis=0)

    def predict(self, inputs):
        self.eval()
        t, preds = self._to_tensor(inputs), []
        with torch.no_grad():
            for s in range(0, len(t), self.infer_batch_size):
                b = t[s : s + self.infer_batch_size].to(self.device)
                preds.append(self.forward(b).argmax(dim=1).cpu().numpy())
        return np.concatenate(preds, axis=0)

    def get_embeddings(self, inputs):
        self.eval()
        t, embs = self._to_tensor(inputs), []
        with torch.no_grad():
            for s in range(0, len(t), self.infer_batch_size):
                b = t[s : s + self.infer_batch_size].to(self.device)
                _, z = self.forward(b, extract_features=True)
                embs.append(z.cpu().numpy())
        return np.concatenate(embs, axis=0)

    def predict_uncertainty_mcd(self, inputs, n_forward_passes=10):
        assert self.use_dropout, "predict_uncertainty_mcd requires use_dropout=True"
        self.train()
        t, runs = self._to_tensor(inputs).to(self.device), []
        with torch.no_grad():
            for _ in range(n_forward_passes):
                pass_p = []
                for s in range(0, len(t), self.infer_batch_size):
                    p = torch.softmax(self.forward(t[s : s + self.infer_batch_size]), dim=1)
                    pass_p.append(p.cpu().numpy())
                runs.append(np.concatenate(pass_p, axis=0))
        self.eval()
        mc       = np.stack(runs)
        mean_p   = mc.mean(axis=0)
        entropy  = -(mean_p * np.log(mean_p + 1e-10)).sum(axis=1)
        variance = mc.var(axis=0).sum(axis=1)
        return entropy, variance



class ClassNNet(nn.Module):
    def __init__(self, num_features, num_classes, device, use_dropout=False):
        super(ClassNNet, self).__init__()

        self.device = device

        self.use_dropout = use_dropout

        self.layer_1 = nn.Linear(num_features, 256)
        self.layer_2 = nn.Linear(256, 256)
        self.layer_3 = nn.Linear(256, 128)
        self.layer_4 = nn.Linear(128, 64)
        self.layer_5 = nn.Linear(64, num_classes)

        self.z_dim = 256 + 128

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(256)
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.batchnorm3 = nn.BatchNorm1d(128)
        self.batchnorm4 = nn.BatchNorm1d(64)

    def forward(self, x, extract_features=False):
        x = self.layer_1(x)
        x = self.relu(x)
        if self.use_dropout:
          x = self.dropout(x)
          x = self.batchnorm1(x)

        z2 = self.layer_2(x)
        x = self.relu(z2)
        if self.use_dropout:
          x = self.dropout(x)
          x = self.batchnorm2(x)

        z3 = self.layer_3(x)
        x = self.relu(z3)
        if self.use_dropout:
          x = self.dropout(x)
          x = self.batchnorm3(x)
        x = self.layer_4(x)
        x = self.relu(x)

        if self.use_dropout:
            x = self.dropout(x)
            x = self.batchnorm4(x)

        x = self.layer_5(x)

        if extract_features:
          return x, torch.cat([z2,z3],1)
        else:
          return x

    def predict_proba(self, inputs):
        """
        Predict probabilities given any input data
        """
        self.eval()
        get_prob = nn.Softmax(dim = 1)
        with torch.no_grad():
            # Convert to tensor if numpy array
            if isinstance(inputs, np.ndarray):
                inputs = torch.from_numpy(inputs).float()
            inputs = inputs.to(self.device)
            outputs = self(inputs)
            prob = get_prob(outputs).cpu().numpy()
        return prob

    def predict(self, inputs):
        """
        Predict class labels given input data

        Args:
            inputs: Input tensor or numpy array

        Returns:
            numpy array: Predicted class labels
        """
        self.eval()
        with torch.no_grad():
            # Convert to tensor if numpy array
            if isinstance(inputs, np.ndarray):
                inputs = torch.from_numpy(inputs).float()

            inputs = inputs.to(self.device)
            outputs = self(inputs)
            predictions = torch.argmax(outputs, dim=1)

        return predictions.cpu().numpy()

    def get_embeddings(self, inputs):
        """Extract penultimate layer representations"""
        self.eval()
        with torch.no_grad():
            if isinstance(inputs, np.ndarray):
                inputs = torch.from_numpy(inputs).float()
            inputs = inputs.to(self.device)
            # Call forward with feature extraction flag
            _, embeddings = self(inputs, extract_features=True)
        return embeddings.cpu().numpy()

    def predict_uncertainty_mcd(self, inputs, n_forward_passes=10):
        """
        Monte Carlo Dropout to estimate uncertainty.
        Returns: (mean_entropy, prediction_variance)
        """
        # Set to train mode to enable Dropout during inference
        self.train()

        if isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs).float()
        inputs = inputs.to(self.device)

        # Stack predictions from multiple passes
        outputs_list = []
        with torch.no_grad():
            for _ in range(n_forward_passes):
                outputs = self(inputs)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                outputs_list.append(probs.cpu().numpy())

        # Restore eval mode
        self.eval()

        # Shape: (n_passes, n_samples, n_classes)
        mc_probs = np.stack(outputs_list)

        # 1. Entropy of the mean probability (Predictive Entropy)
        mean_probs = np.mean(mc_probs, axis=0)
        mean_entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10), axis=1)

        # 2. Variance of the probabilities (Epistemic Uncertainty)
        # We take the sum of variances across classes or the max variance
        pred_variance = np.var(mc_probs, axis=0).sum(axis=1)

        return mean_entropy, pred_variance

class BinaryClassification(nn.Module):
    def __init__(self, num_features, device, use_dropout=False):
        super(BinaryClassification, self).__init__()

        self.device = device
        self.use_dropout = use_dropout

        self.layer_1 = nn.Linear(num_features, 256)
        self.layer_2 = nn.Linear(256, 256)
        self.layer_3 = nn.Linear(256, 128)
        self.layer_4 = nn.Linear(128, 64)
        self.layer_5 = nn.Linear(64, 1)

        self.z_dim = 256 + 128

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(256)
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.batchnorm3 = nn.BatchNorm1d(128)
        self.batchnorm4 = nn.BatchNorm1d(64)

    def forward(self, x, extract_features=False):
        x = self.layer_1(x)
        x = self.relu(x)
        if self.use_dropout:
            x = self.dropout(x)
            x = self.batchnorm1(x)

        z2 = self.layer_2(x)
        x = self.relu(z2)
        if self.use_dropout:
            x = self.dropout(x)
            x = self.batchnorm2(x)

        z3 = self.layer_3(x)
        x = self.relu(z3)
        if self.use_dropout:
            x = self.dropout(x)
            x = self.batchnorm3(x)

        x = self.layer_4(x)
        x = self.relu(x)
        if self.use_dropout:
            x = self.dropout(x)
            x = self.batchnorm4(x)

        x = self.layer_5(x)
        x = x.squeeze(1)  # Shape: [batch_size, 1] -> [batch_size]

        if extract_features:
            return x, torch.cat([z2, z3], 1)
        else:
            return x

    def predict(self, X):
        self.eval()  # Changed from self.net.eval()
        X_tensor = torch.from_numpy(X).float().to(self.device)

        with torch.no_grad():
            outputs = self(X_tensor)  # Changed from self.net(X_tensor)
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities > 0.5).long()

        return predictions.cpu().numpy()

    def predict_proba(self, X):
        self.eval()
        X_tensor = torch.from_numpy(X).float().to(self.device)

        with torch.no_grad():
            outputs = self(X_tensor)
            prob_class_1 = torch.sigmoid(outputs)
            prob_class_0 = 1 - prob_class_1
            probabilities = torch.stack([prob_class_0, prob_class_1], dim=1)

        return probabilities.cpu().numpy()

    def get_embeddings(self, inputs):
        """Extract penultimate layer representations"""
        self.eval()
        with torch.no_grad():
            if isinstance(inputs, np.ndarray):
                inputs = torch.from_numpy(inputs).float()
            inputs = inputs.to(self.device)
            _, embeddings = self(inputs, extract_features=True)
        return embeddings.cpu().numpy()

    def predict_uncertainty_mcd(self, inputs, n_forward_passes=10):
        """
        Monte Carlo Dropout to estimate uncertainty.
        Returns: (mean_entropy, prediction_variance)
        """
        self.train()

        if isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs).float()
        inputs = inputs.to(self.device)

        outputs_list = []
        with torch.no_grad():
            for _ in range(n_forward_passes):
                outputs = self(inputs)
                probs = torch.sigmoid(outputs)
                probs = torch.stack([1 - probs, probs], dim=1)
                outputs_list.append(probs.cpu().numpy())

        self.eval()

        # Shape: (n_passes, n_samples, 2)
        mc_probs = np.stack(outputs_list)

        mean_probs = np.mean(mc_probs, axis=0)
        mean_entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10), axis=1)

        pred_variance = np.var(mc_probs, axis=0).sum(axis=1)

        return mean_entropy, pred_variance


class Blackbox:
    """
    A training wrapper for PyTorch models with support for training, validation, and metric tracking.
    Accepts numpy arrays as input and handles data loading internally.
    Data can be provided either at initialization or during fit().
    """

    def __init__(self, net, device, X_train=None, y_train=None, X_val=None, y_val=None,
                 batch_size=32, max_epoch=10, learning_rate=0.001, criterion=None,
                 optimizer=None, num_classes=None, compute_accuracy=False,
                 verbose=True, shuffle_train=True):
        """
        Initialize the Blackbox trainer.

        Args:
            net: PyTorch model to train
            device: Device to train on (cuda/cpu)
            X_train: Training features as numpy array (optional, can be provided in fit())
            y_train: Training labels as numpy array (optional, can be provided in fit())
            X_val: Validation features as numpy array (optional, can be provided in fit())
            y_val: Validation labels as numpy array (optional, can be provided in fit())
            batch_size: Batch size for training
            max_epoch: Maximum number of epochs
            learning_rate: Learning rate for optimizer
            criterion: Loss function
            optimizer: Optimizer instance
            num_classes: Number of classes for classification (required if compute_accuracy=True)
            compute_accuracy: Whether to compute accuracy metrics
            verbose: Whether to print training progress
            shuffle_train: Whether to shuffle training data
        """
        self.net = net.to(device)
        self.device = device
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.criterion = criterion
        self.optimizer = optimizer
        self.compute_accuracy = compute_accuracy
        self.shuffle_train = shuffle_train
        self.ID = np.random.randint(0, high=2**31)

        # Store initial data if provided
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        # Initialize data loaders as None
        self.train_loader = None
        self.val_loader = None
        self.n_minibatches = None

        # Store num_classes for later use
        self._num_classes = num_classes
        self.num_classes = None
        self.task = None

        # If data is provided at initialization, set up data loaders
        if X_train is not None and y_train is not None:
            self._setup_data(X_train, y_train, X_val, y_val)

    def _setup_data(self, X_train, y_train, X_val=None, y_val=None):
        """
        Internal method to set up data loaders and task configuration.

        Args:
            X_train: Training features as numpy array
            y_train: Training labels as numpy array
            X_val: Validation features as numpy array (optional)
            y_val: Validation labels as numpy array (optional)
        """
        # Determine task type and number of classes
        if self.compute_accuracy:
            if self._num_classes is None:
                # Auto-detect number of classes from training labels
                self.num_classes = len(np.unique(y_train))
            else:
                self.num_classes = self._num_classes

            # Determine task type
            if self.num_classes == 2:
                self.task = 'binary'
            else:
                self.task = 'multiclass'
        else:
            self.num_classes = None
            self.task = None

        # Convert numpy arrays to PyTorch tensors
        X_train_tensor = torch.from_numpy(X_train).float()
        y_train_tensor = torch.from_numpy(y_train).long()

        # Create training dataset and loader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train
        )

        # Create validation dataset and loader if validation data is provided
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.from_numpy(X_val).float()
            y_val_tensor = torch.from_numpy(y_val).long()
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False
            )
        else:
            self.val_loader = None

        self.n_minibatches = len(self.train_loader)

        if self.verbose:
            print("===== HYPERPARAMETERS =====")
            print(f"batch_size={self.batch_size}")
            print(f"n_epochs={self.max_epoch}")
            print(f"learning_rate={self.learning_rate}")
            print(f"train_samples={len(X_train)}")
            if X_val is not None:
                print(f"val_samples={len(X_val)}")
            print("=" * 30)

    def train_single_epoch(self):
        """
        Train the model for a single epoch.

        Returns:
            tuple or float: (train_loss, train_acc) if compute_accuracy else train_loss
        """
        if self.train_loader is None:
            raise ValueError("No training data available. Provide data at initialization or in fit().")

        self.net.train()
        epoch_loss = 0
        epoch_acc = 0 if self.compute_accuracy else None

        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.net(inputs)

            # Handle both standard losses (outputs, targets) and custom losses (outputs, inputs, targets)
            try:
                # Use float targets for BCE-type losses, long targets for CrossEntropyLoss
                if isinstance(self.criterion, (nn.BCELoss, nn.BCEWithLogitsLoss)):
                    loss = self.criterion(outputs, targets.float())
                else:
                    loss = self.criterion(outputs, targets)
            except TypeError:
                # Handle custom losses with 3 arguments
                loss = self.criterion(outputs, inputs, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)

            self.optimizer.step()

            epoch_loss += loss.item()

            if self.compute_accuracy:
                from torchmetrics.functional import accuracy
                epoch_acc += float(accuracy(outputs, targets, task=self.task, num_classes=self.num_classes).cpu().numpy())

        epoch_loss /= len(self.train_loader)

        if self.compute_accuracy:
            epoch_acc /= len(self.train_loader)
            return epoch_loss, epoch_acc

        return epoch_loss

    def validate(self):
        """
        Run validation on the validation set.

        Returns:
            tuple or float: (val_loss, val_acc) if compute_accuracy else val_loss
        """
        if self.val_loader is None:
            return None

        self.net.eval()
        val_loss = 0
        val_acc = 0 if self.compute_accuracy else None

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.net(inputs)

                # Handle both standard losses (outputs, targets) and custom losses (outputs, inputs, targets)
                try:
                    # Use float targets for BCE-type losses, long targets for CrossEntropyLoss
                    if isinstance(self.criterion, (nn.BCELoss, nn.BCEWithLogitsLoss)):
                        loss = self.criterion(outputs, targets.float())
                    else:
                        loss = self.criterion(outputs, targets)
                except TypeError:
                    # Handle custom losses with 3 arguments
                    loss = self.criterion(outputs, inputs, targets)

                val_loss += loss.item()

                if self.compute_accuracy:
                    from torchmetrics.functional import accuracy
                    val_acc += float(accuracy(outputs, targets, task=self.task, num_classes=self.num_classes).cpu().numpy())

        val_loss /= len(self.val_loader)

        if self.compute_accuracy:
            val_acc /= len(self.val_loader)
            return val_loss, val_acc

        return val_loss

    def fit(self, X_train=None, y_train=None, X_val=None, y_val=None, save_dir='./models/model.pt'):
        """
        Train the model for the full number of epochs with validation.

        Args:
            X_train: Training features as numpy array (optional if provided at initialization)
            y_train: Training labels as numpy array (optional if provided at initialization)
            X_val: Validation features as numpy array (optional)
            y_val: Validation labels as numpy array (optional)
            save_dir: Path to save the trained model (set to None to skip saving)

        Returns:
            dict: Dictionary containing training statistics
        """
        from tqdm.autonotebook import tqdm
        import pathlib

        # Use provided data or fall back to initialization data
        if X_train is not None and y_train is not None:
            self._setup_data(X_train, y_train, X_val, y_val)
        elif self.train_loader is None:
            # No data at initialization and no data provided in fit()
            raise ValueError("No training data available. Provide X_train and y_train either at initialization or in fit().")

        # Create save directory if saving is enabled
        if save_dir is not None:
            pathlib.Path(save_dir).parent.mkdir(parents=True, exist_ok=True)

        # Initialize statistics dictionary
        stats = {
            'epoch': [],
            'train_loss': [],
        }

        if self.val_loader is not None:
            stats['val_loss'] = []

        if self.compute_accuracy:
            stats['train_acc'] = []
            if self.val_loader is not None:
                stats['val_acc'] = []

        print("Begin training.")

        for epoch in tqdm(range(1, self.max_epoch + 1)):
            # Train
            train_result = self.train_single_epoch()
            if self.compute_accuracy:
                train_loss, train_acc = train_result
            else:
                train_loss = train_result

            # Validate
            if self.val_loader is not None:
                val_result = self.validate()
                if self.compute_accuracy:
                    val_loss, val_acc = val_result
                else:
                    val_loss = val_result
            else:
                val_loss = None
                val_acc = None

            # Update statistics
            stats['epoch'].append(epoch)
            stats['train_loss'].append(train_loss)
            if val_loss is not None:
                stats['val_loss'].append(val_loss)
            if self.compute_accuracy:
                stats['train_acc'].append(train_acc)
                if val_acc is not None:
                    stats['val_acc'].append(val_acc)

            # Print progress
            if self.verbose:
                print(f'Epoch {epoch:03d}: train_loss: {train_loss:.3f}', end='')
                if val_loss is not None:
                    print(f' | val_loss: {val_loss:.3f}', end='')
                if self.compute_accuracy:
                    print(f' | train_acc: {train_acc:.3f}', end='')
                    if val_acc is not None:
                        print(f' | val_acc: {val_acc:.3f}', end='')
                print(flush=True)

        # Save final model and statistics if save_dir is provided
        if save_dir is not None:
            saved_state = {
                'stats': stats,
                'model_state': self.net.state_dict()
            }
            torch.save(saved_state, save_dir)
            print(f"\nTraining complete. Model saved to {save_dir}")
        else:
            print("\nTraining complete. Model not saved.")

        return stats

    def predict(self, X):
        """
        Predict the most likely class label for each sample.

        Args:
            X: Input features as numpy array (shape: [n_samples, num_features])

        Returns:
            numpy array: Predicted class labels (shape: [n_samples])
                        Values are integers from 0 to num_classes-1
        """
        self.net.eval()
        X_tensor = torch.from_numpy(X).float().to(self.device)

        with torch.no_grad():
            outputs = self.net(X_tensor)

            # Handle binary classification with single output
            if outputs.dim() == 1 or (outputs.dim() == 2 and outputs.shape[1] == 1):
                if outputs.dim() == 2:
                    outputs = outputs.squeeze(1)
                predictions = (torch.sigmoid(outputs) > 0.5).long()
            else:
                predictions = torch.argmax(outputs, dim=1)

        return predictions.cpu().numpy()

    def predict_proba(self, X):
        """
        Predict class probabilities for each sample.

        Args:
            X: Input features as numpy array (shape: [n_samples, num_features])

        Returns:
            numpy array: Predicted probabilities (shape: [n_samples, num_classes])
                        Each row sums to 1.0
        """
        self.net.eval()
        X_tensor = torch.from_numpy(X).float().to(self.device)

        with torch.no_grad():
            outputs = self.net(X_tensor)

            # Handle binary classification with single output
            if outputs.dim() == 1 or (outputs.dim() == 2 and outputs.shape[1] == 1):
                if outputs.dim() == 2:
                    outputs = outputs.squeeze(1)
                prob_class_1 = torch.sigmoid(outputs)
                prob_class_0 = 1 - prob_class_1
                probabilities = torch.stack([prob_class_0, prob_class_1], dim=1)
            else:
                # For multi-class classification
                softmax = torch.nn.Softmax(dim=1)
                probabilities = softmax(outputs)

        return probabilities.cpu().numpy()

    def get_accuracy(self, X, y):
        """
        Compute accuracy for given inputs and targets.

        Args:
            X: Input features as numpy array
            y: Target labels as numpy array

        Returns:
            float: Accuracy value
        """
        from torchmetrics.functional import accuracy
        self.net.eval()
        X_tensor = torch.from_numpy(X).float().to(self.device)
        y_tensor = torch.from_numpy(y).long().to(self.device)

        with torch.no_grad():
            outputs = self.net(X_tensor)
            acc = float(accuracy(outputs, y_tensor, task=self.task, num_classes=self.num_classes, top_k=1).cpu().numpy())

        return acc
