---
title: "Energy-Based Models & Structured Prediction"
excerpt_separator: "Energy-Based Models (EBMs) assign a scalar energy to configurations of variables and perform inference by minimizing energy"
tags:
  - Energy-Based Models
  - Dynamic Programming
  - Viterbi
  - PyTorch
categories:
  - Computer Science
---

Energy-Based Models (EBMs) assign a scalar **energy** to configurations of variables and perform inference by **minimizing energy**.  <!--more-->

<h3>Intro</h3>

We tackle **structured prediction** for text recognition: transcribing a word image into characters of **variable length**.  We (1) build a synthetic dataset, (2) pretrain a sliding-window CNN on single characters, (3) align windowed predictions to labels with a **Viterbi** dynamic program, (4) train an EBM using **cross-entropy along the best path**, and (5) compare to a Connectionist Temporal Classification CTC (Graph Transducer Networks GTN) approach.

Complete Notebook

[EBMs + Structured Prediction Jupyter Notebook HTML](/assets/html/ebm_structured_prediction.html)

[Kaggle Source from sumanyumuku98](https://www.kaggle.com/code/mizomatic/a-tutorial-on-energy-based-models-ebms/notebook)

This used to be a homework in NYU Deep Learning class taught by Alfredo Canziani and Yann LeCun, and someone reshared it in Kaggle. I had the luck to take their class, and I am grateful someone had kept a better record of this than I did.

**Highlights**
- Sliding-window CNN outputs `K×27` energies (26 letters + blank).
- **Viterbi** finds the minimum-energy alignment between windows and targets.
- EBM training: **sum of cross-entropies along the Viterbi path**.
- **GTN/CTC** alternative: graph-based, batched training without manual DP.
- Works on synthetic and “handwritten-style” fonts; simple collapse decoding recovers text.

<h3>Example 1 — Dataset, Model, Single-Character Pretraining</h3>

```python
# --- Imports & setup ---
from PIL import ImageDraw, ImageFont, Image
import string, random, time, copy
import torch
from torch import nn
from torch.optim import Adam
import torch.optim as optim
from collections import Counter
from tqdm.notebook import tqdm
import torchvision
from torchvision import transforms
from matplotlib import pyplot as plt

torch.manual_seed(0)

# --- Constants ---
ALPHABET_SIZE = 27  # 26 letters + 1 blank/divider
BETWEEN = 26

# --- Basic transforms ---
simple_transforms = transforms.Compose([transforms.ToTensor()])

# --- Synthetic dataset of word images ---
class SimpleWordsDataset(torch.utils.data.IterableDataset):
    def __init__(self, max_length, len=100, jitter=False, noise=False):
        self.max_length = max_length
        self.transforms = transforms.ToTensor()
        self.len = len
        self.jitter = jitter
        self.noise = noise
  
    def __len__(self):
        return self.len

    def __iter__(self):
        for _ in range(self.len):
            text = ''.join([random.choice(string.ascii_lowercase) for _ in range(self.max_length)])
            img = self.draw_text(text, jitter=self.jitter, noise=self.noise)
            yield img, text
  
    def draw_text(self, text, length=None, jitter=False, noise=False):
        if length is None:
            length = 18 * len(text)
        img = Image.new('L', (length, 32))
        fnt = ImageFont.truetype("fonts/Anonymous.ttf", 20)

        d = ImageDraw.Draw(img)
        pos = (random.randint(0, 7), 5) if jitter else (0, 5)
        d.text(pos, text, fill=1, font=fnt)

        img = self.transforms(img)
        img[img > 0] = 1 
        if noise:
            img += torch.bernoulli(torch.ones_like(img) * 0.1)
            img = img.clamp(0, 1)
        return img[0]

# --- Sliding window CNN (character-sized kernel) ---
class SimpleNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 512, kernel_size=(32, 18), stride=(1, 4), padding="valid")
        self.linear = nn.Linear(512, ALPHABET_SIZE)
    def forward(self, x):
        # Input: (B, 1, 32, W) -> Conv over width -> squeeze height -> (B,K,512) -> Linear -> (B,K,27)
        return self.linear(self.conv(x).squeeze(dim=-2).permute(0, 2, 1))

# --- Helpers for plotting (optional) ---
def plot_energies(ce):
    fig = plt.figure(dpi=200)
    ax = plt.axes()
    im = ax.imshow(ce.cpu().T)
    ax.set_xlabel('window locations →'); ax.xaxis.set_label_position('top')
    ax.set_ylabel('← classes'); ax.set_xticks([]); ax.set_yticks([])
    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    plt.colorbar(im, cax=cax)

# --- One-character pretraining ---
def cross_entropy(energies, *args, **kwargs):
    # Energies -> minimize => use log-soft-argmin (negate energies)
    return nn.functional.cross_entropy(-1 * energies, *args, **kwargs)

def simple_collate_fn(samples):
    images, annotations = zip(*samples)
    images = list(images)
    annotations = list(annotations)
    annotations = list(map(lambda c: torch.tensor(ord(c) - ord('a')), annotations))
    m_width = max(18, max([i.shape[1] for i in images]))
    for i in range(len(images)):
        images[i] = torch.nn.functional.pad(images[i], (0, m_width - images[i].shape[-1]))
    if len(images) == 1:
        return images[0].unsqueeze(0), torch.stack(annotations)
    else:
        return torch.stack(images), torch.stack(annotations)

def train_model(model, epochs, dataloader, criterion, optimizer):
    model.train()
    pbar = tqdm(range(epochs))
    for _ in pbar:
        train_loss = 0.0
        for images, target in dataloader:
            images = images.unsqueeze(1)  # (B,1,32,W)
            optimizer.zero_grad()
            out = model(images)          # (B,K,27), for 1 char K==1
            loss = criterion(out.squeeze(), target=target)
            loss.backward(); optimizer.step()
            train_loss += loss.item()
        train_loss /= len(dataloader)
        pbar.set_postfix({'Train Loss': train_loss})

# Usage:
# sds_1 = SimpleWordsDataset(1, len=1000, jitter=True, noise=False)
# loader_1 = torch.utils.data.DataLoader(sds_1, batch_size=16, num_workers=0, collate_fn=simple_collate_fn)
# model = SimpleNet()
# optimizer = Adam(model.parameters(), lr=1e-2)
# train_model(model, 15, loader_1, cross_entropy, optimizer)

# Accuracy check on single-char:
def get_accuracy(model, dataset):
    cnt = 0
    for img, label in dataset:
        energies = model(img.unsqueeze(0).unsqueeze(0))[0, 0]  # (27,)
        pred = energies.argmin(dim=-1)
        cnt += int(pred == (ord(label[0]) - ord('a')))
    return cnt / len(dataset)

# tds = SimpleWordsDataset(1, len=100)
# assert get_accuracy(model, tds) == 1.0
```

**Highlights**
- One-character dataset & padding collate.
- `SimpleNet` ensures `(32×18)` receptive field → per-window energies over 27 classes.

<h3>Example 2 — Alignment Utilities & Viterbi (Dynamic Programming)</h3>

```python
# --- Path/CE matrices (vectorized) ---
def build_path_matrix(energies, targets):
    """
    energies: (B, L, 27)
    targets:  (B, T) integer indices in [0..26]
    returns:  (B, L, T) where out[b,i,k] = energies[b,i,targets[b,k]]
    """
    L = energies.shape[1]
    targets_exp = targets.unsqueeze(1).repeat(1, L, 1)      # (B,L,T)
    return torch.gather(energies, 2, targets_exp)           # (B,L,T)

def build_ce_matrix(energies, targets):
    """
    ce[b,i,k] = CE(energies[b,i], targets[b,k])
    returns: (B, L, T)
    """
    L, T = energies.shape[1], targets.shape[-1]
    energies4 = energies.permute(0, 2, 1).unsqueeze(-1).repeat(1,1,1,T)  # (B,27,L,T)
    targets4  = targets.unsqueeze(1).repeat(1, L, 1)                      # (B,L,T)
    return cross_entropy(energies4, targets4, reduction='none')

# --- Label transform: interleave blanks: 'cat' -> c _ a _ t _ ---
def transform_word(s):
    encoded = []
    for c in s:
        encoded.append(ord(c) - ord('a'))
        encoded.append(BETWEEN)
    return torch.tensor(encoded)  # len = 2*len(s)

# --- Path validity & energy ---
def checkValidMapping(path, T):
    for i in range(1, len(path)):
        if path[i] < path[i-1]:
            return False
    return True

def path_energy(pm, path):
    """
    pm: (L,T) energies for (window, target-pos)
    path: list of length L with target indices
    """
    T = pm.shape[1]
    if not checkValidMapping(path, T):
        return torch.tensor(2**30)
    energy = 0.0
    for i, c in enumerate(path):
        energy += pm[i, c]
    return energy

# --- Viterbi (DP) to find best path ---
def find_path(pm):
    """
    pm: (L,T) energy matrix
    returns: (free_energy, path_points, dp)
      - free_energy: sum on best path
      - path_points: list of (i,j) along best path
      - dp: DP table (L,T)
    """
    L, T = pm.shape
    dp = torch.zeros((L, T), device=pm.device)
    parent = [[None]*T for _ in range(L)]
    dp[0,0] = pm[0,0]; parent[0][0] = (0,0)
    for j in range(1, T):
        dp[0,j] = 2**30
        parent[0][j] = (0, j)
    for i in range(1, L):
        dp[i,0] = dp[i-1,0] + pm[i,0]
        parent[i][0] = (i-1,0)
    for i in range(1, L):
        for j in range(1, T):
            a, b = dp[i-1,j], dp[i-1,j-1]
            if a < b:
                dp[i,j] = a + pm[i,j]; parent[i][j] = (i-1, j)
            else:
                dp[i,j] = b + pm[i,j]; parent[i][j] = (i-1, j-1)
    # Backtrack: pick best j in last row
    j = torch.argmin(dp[L-1]).item()
    path = []
    for i in range(L-1, -1, -1):
        path.append(j)
        _, j = parent[i][j]
    path.reverse()
    points = list(zip(range(L), path))
    return (path_energy(pm, path), points, dp)

# --- Example usage (alphabet image) ---
# alphabet = SimpleWordsDataset(1).draw_text(string.ascii_lowercase, 340)
# energies = model(alphabet.view(1,1,*alphabet.shape))   # (1,L,27)
# targets  = transform_word(string.ascii_lowercase).unsqueeze(0)  # (1,T)
# pm = build_path_matrix(energies, targets)              # (1,L,T)
# free_energy, path, dp = find_path(pm[0])
```

**Highlights**
- `build_path_matrix` gathers per-window energy for each label position.
- `find_path` implements the **minimum-energy monotone alignment**.
- Diagonal-ish best paths appear as the model improves.

<h3>Example 3 — Train EBM with Viterbi Alignments</h3>

```python
def collate_fn(samples):
    """Collate for multi-char: pad images to same width; interleave blanks in labels and pad with BETWEEN."""
    images, annotations = zip(*samples)
    images, annotations = list(images), list(annotations)
    annotations = list(map(transform_word, annotations))
    m_width = max(18, max([i.shape[1] for i in images]))
    m_len   = max(3, max([s.shape[0] for s in annotations]))
    for i in range(len(images)):
        images[i] = torch.nn.functional.pad(images[i], (0, m_width - images[i].shape[-1]))
        annotations[i] = torch.nn.functional.pad(annotations[i], (0, m_len - annotations[i].shape[0]), value=BETWEEN)
    if len(images) == 1:
        return images[0].unsqueeze(0), torch.stack(annotations)
    else:
        return torch.stack(images), torch.stack(annotations)

def train_ebm_model(model, num_epochs, train_loader, criterion, optimizer):
    """Train EBM using best-path (Viterbi) alignments."""
    pbar = tqdm(range(num_epochs))
    model.train()
    for _ in pbar:
        total = 0.0
        for samples, targets in train_loader:
            optimizer.zero_grad()
            energies = model(samples.unsqueeze(1))           # (B,L,27)
            pm = build_path_matrix(energies, targets)        # (B,L,T)
            batch_losses = []
            for b in range(pm.shape[0]):
                free_energy, best_path, _ = find_path(pm[b])  # best_path: list[(i,j)]
                j_indices = [ij[1] for ij in best_path]       # target indices along path
                # Sum CE along best path:
                batch_losses.append(criterion(energies[b], targets[b, j_indices]))
            loss = sum(batch_losses)
            total += loss.item()
            loss.backward(); optimizer.step()
        pbar.set_postfix({'train_loss': total / len(train_loader.dataset)})

# Usage:
# sds2 = SimpleWordsDataset(2, 2500)
# loader2 = torch.utils.data.DataLoader(sds2, batch_size=32, num_workers=0, collate_fn=collate_fn)
# ebm_model = copy.deepcopy(model)
# optimizer = Adam(ebm_model.parameters(), lr=1e-3)
# train_ebm_model(ebm_model, 15, loader2, cross_entropy, optimizer)
```

**Highlights**
- Loss is the **sum of cross-entropies along the Viterbi path**.
- Works but **slow** due to per-sample DP; suitable for teaching/demo.

<h3>Example 4 — GTN / CTC: Graph-Based Training & Viterbi</h3>

```python
# --- GTN-based CTC loss and Viterbi (adapted) ---
import torch.nn.functional as F
import gtn

class CTCLossFunction(torch.autograd.Function):
    @staticmethod
    def create_ctc_graph(target, blank_idx):
        g_criterion = gtn.Graph(False)
        L = len(target); S = 2 * L + 1
        for l in range(S):
            idx = (l - 1) // 2
            g_criterion.add_node(l == 0, l == S - 1 or l == S - 2)
            label = target[idx] if l % 2 else blank_idx
            g_criterion.add_arc(l, l, label)
            if l > 0:
                g_criterion.add_arc(l - 1, l, label)
            if l % 2 and l > 1 and label != target[idx - 1]:
                g_criterion.add_arc(l - 2, l, label)
        g_criterion.arc_sort(False)
        return g_criterion

    @staticmethod
    def forward(ctx, log_probs, targets, blank_idx=0, reduction="none"):
        B, T, C = log_probs.shape
        losses, scales, emissions_graphs = [None]*B, [None]*B, [None]*B

        def process(b):
            g_emissions = gtn.linear_graph(T, C, log_probs.requires_grad)
            cpu_data = log_probs[b].cpu().contiguous()
            g_emissions.set_weights(cpu_data.data_ptr())
            g_criterion = CTCLossFunction.create_ctc_graph(targets[b], blank_idx)
            g_loss = gtn.negate(gtn.forward_score(gtn.intersect(g_emissions, g_criterion)))
            scale = 1.0
            if reduction == "mean":
                L = len(targets[b]); scale = 1.0 / L if L > 0 else scale
            elif reduction != "none":
                raise ValueError("invalid reduction")
            losses[b], scales[b], emissions_graphs[b] = g_loss, scale, g_emissions

        gtn.parallel_for(process, range(B))
        ctx.auxiliary_data = (losses, scales, emissions_graphs, log_probs.shape)
        loss = torch.tensor([losses[b].item() * scales[b] for b in range(B)])
        return torch.mean(loss.cuda() if log_probs.is_cuda else loss)

    @staticmethod
    def backward(ctx, grad_output):
        losses, scales, emissions_graphs, in_shape = ctx.auxiliary_data
        B, T, C = in_shape
        input_grad = torch.empty((B, T, C))
        def process(b):
            gtn.backward(losses[b], False)
            emissions = emissions_graphs[b]
            grad = emissions.grad().weights_to_numpy()
            input_grad[b] = torch.from_numpy(grad).view(1, T, C) * scales[b]
        gtn.parallel_for(process, range(B))
        if grad_output.is_cuda:
            input_grad = input_grad.cuda()
        input_grad *= grad_output / B
        return (input_grad, None, None, None)

CTCLoss = CTCLossFunction.apply

def viterbi(energies, targets, blank_idx=0):
    outputs = -1 * energies
    B, T, C = outputs.shape
    paths, scores, emissions_graphs = [None]*B, [None]*B, [None]*B
    def process(b):
        L = len(targets[b])
        g_emissions = gtn.linear_graph(T, C, outputs.requires_grad)
        cpu_data = outputs[b].cpu().contiguous()
        g_emissions.set_weights(cpu_data.data_ptr())
        g_criterion = CTCLossFunction.create_ctc_graph(targets[b], blank_idx)
        g_inter = gtn.intersect(g_emissions, g_criterion)
        g_score = gtn.viterbi_score(g_inter)
        g_path = gtn.viterbi_path(g_inter)
        l = 0; mapped = []
        for p in g_path.labels_to_list():
            if 2*p < L:
                l = p; mapped.append(2*p)
            else:
                mapped.append(2*l + 1)
        paths[b] = mapped
        scores[b] = -1 * g_score.item()
        emissions_graphs[b] = g_emissions
    gtn.parallel_for(process, range(B))
    return (scores, paths)

def train_gtn_model(model, num_epochs, train_loader, criterion, optimizer):
    pbar = tqdm(range(num_epochs))
    model.train()
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    for _ in pbar:
        total = 0.0
        for samples, targets in train_loader:
            samples, targets = samples.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(samples.unsqueeze(1))          # (B,L,27)
            log_probs = F.log_softmax(-1.0 * outputs, dim=-1)
            loss = criterion(log_probs, targets)           # CTC
            total += loss.item()
            loss.backward(); optimizer.step()
        pbar.set_postfix({'train_loss': total / len(train_loader.dataset)})

# Usage:
# sds3 = SimpleWordsDataset(3, 2500)
# loader3 = torch.utils.data.DataLoader(sds3, batch_size=32, num_workers=0, collate_fn=collate_fn)
# gtn_model = copy.deepcopy(model)
# optimizer = Adam(gtn_model.parameters(), lr=1e-3)
# train_gtn_model(gtn_model, 15, loader3, CTCLoss, optimizer)
```

**Highlights**
- CTC = alignment graph `A_y` ∘ emissions graph `E`.
- Training uses **log-softmax** and **CTCLoss**.
- `viterbi` via GTN yields best path and score without manual DP loops.

<h3>Example 5 — From Scratch (No Pretraining) & Handwritten-Style Font</h3>

```python
# --- No-pretraining: GTN/CTC directly on multi-character data ---
# sds_np = SimpleWordsDataset(3, 2500)
# loader_np = torch.utils.data.DataLoader(sds_np, batch_size=32, num_workers=0, collate_fn=collate_fn)
# gtn_no_pretrained = SimpleNet()
# optimizer = Adam(gtn_no_pretrained.parameters(), lr=1e-3)
# train_gtn_model(gtn_no_pretrained, 20, loader_np, CTCLoss, optimizer)

# --- Custom "handwritten-style" font dataset ---
class CustomWordsDataset(torch.utils.data.IterableDataset):
    def __init__(self, max_length, len=100, jitter=False, noise=False, custom_fonts_path=None):
        self.max_length = max_length
        self.transforms = transforms.ToTensor()
        self.len = len
        self.jitter = jitter
        self.noise = noise
        self.custom_fonts_path = custom_fonts_path
  
    def __len__(self):
        return self.len

    def __iter__(self):
        for _ in range(self.len):
            text = ''.join([random.choice(string.ascii_lowercase) for _ in range(self.max_length)])
            img = self.draw_text(text, jitter=self.jitter, noise=self.noise)
            yield img, text
  
    def draw_text(self, text, length=None, jitter=False, noise=False):
        if length is None:
            length = 18 * len(text)
        img = Image.new('L', (length, 32))
        fnt = ImageFont.truetype("fonts/Anonymous.ttf" if not self.custom_fonts_path else self.custom_fonts_path, 20)
        d = ImageDraw.Draw(img)
        pos = (random.randint(0, 7), 5) if jitter else (0, 5)
        d.text(pos, text, fill=1, font=fnt)
        img = self.transforms(img)
        img[img > 0] = 1 
        if noise:
            img += torch.bernoulli(torch.ones_like(img) * 0.1)
            img = img.clamp(0, 1)
        return img[0]

# Usage (after downloading a font to ./fonts/3dumb/2Dumb.ttf):
# sds_hw = CustomWordsDataset(3, 2500, custom_fonts_path="./fonts/3dumb/2Dumb.ttf")
# loader_hw = torch.utils.data.DataLoader(sds_hw, batch_size=32, num_workers=0, collate_fn=collate_fn)
# gtn_hw = SimpleNet()
# optimizer = Adam(gtn_hw.parameters(), lr=1e-3)
# train_gtn_model(gtn_hw, 20, loader_hw, CTCLoss, optimizer)
```

**Highlights**
- Training **from scratch** with GTN/CTC works, though convergence can be slower.
- Domain shift (e.g., handwritten font) lowers scores; still decodes plausible strings.

<h3>Example 6 — Decoding (Collapse Repeats, Drop Blanks)</h3>

```python
def indices_to_str(indices):
    """
    Collapse heuristic:
      1) Map 0..25 -> letters; 26 -> '_'
      2) Split by '_' segments; take the most frequent char per segment
      3) Remove '_' between segments -> final string with '_' as segment delimiter (optional)
    """
    out = []
    for ind in indices:
        out.append('_' if ind == BETWEEN else chr(ind + ord('a')))
    segments = "".join(out).split('_')
    collapsed = [Counter(seg).most_common(1)[0][0] for seg in segments if seg]
    return "_".join(collapsed)  # or "".join(collapsed) to remove underscores entirely

# Example:
# img = SimpleWordsDataset(5, len=1).draw_text('hello')
# energies = ebm_model(img.unsqueeze(0).unsqueeze(0))
# min_indices = energies[0].argmin(dim=-1)
# print(indices_to_str(min_indices))
```
