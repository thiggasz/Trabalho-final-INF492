import pandas as pd
from sklearn.utils import resample
import h5py
import io
import numpy as np
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
import torch
from collections import Counter
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
import torchvision
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
import copy
import torch.nn as nn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, auc, roc_curve
import torch.nn.functional as F
import matplotlib.pyplot as plt

dataset_path = 'datasets/isic-2024-challenge/train-image.hdf5'
dataset_metadata = 'datasets/isic-2024-challenge/train-metadata.csv'

logs_path = 'logs/'
models_path = 'models/'

"""# Dataloader"""

def my_tensor_image_show ( image , label=None ):
    image = image.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)
    plt.imshow(image)
    plt.axis('off')
    if label is None :
        plt.title('Image in tensor format.')
    else :
        plt.title(f'Image in tensor format | Class: {label}')
    plt.show()


def my_imshow(img, numImages=10):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
    img = std * img + mean
    img = np.clip(img, 0, 1)

    img = torchvision.utils.make_grid(img[:numImages],nrow=numImages//2)

    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))

    plt.axis('off')
    plt.imshow(npimg)
    plt.show()

def show_images(data_loader, numImages=10) :
    print(f"Train samples, {data_loader['train']['length']}")
    # get some random training images
    dataiter = iter(data_loader['train']['data'])
    images = next(dataiter)[0]
    my_imshow(images, numImages)

    print(f"Val samples, {data_loader['val']['length']}")
    # get some random val images
    dataiter = iter(data_loader['val']['data'])
    images = next(dataiter)[0]
    my_imshow(images, numImages)

    print(f"Test samples, {data_loader['test']['length']}")
    # get some random training images
    dataiter = iter(data_loader['test']['data'])
    images = next(dataiter)[0]
    my_imshow(images, numImages)

class ISICDataset(torch.utils.data.Dataset):
    def __init__(self, df, hdf5_path, transform=None):
        self.df = df
        self.hdf5_path = hdf5_path
        self.transform = transform
        self.file = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.hdf5_path, 'r')

        isic_id = self.df.iloc[idx]['isic_id']
        label = self.df.iloc[idx]['target']

        img_bytes = self.file[isic_id][()]
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, int(label)

def isic_dataset(hdf5_path, test_transform, architecture, DA=True, n_samples=40000):
    # Defines the dataset based on the sample size
    df = pd.read_csv(dataset_metadata)

    patient_col = 'patient_id'

    # Define the train set
    train_test = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, rest_idx = next(train_test.split(df, groups=df[patient_col]))

    df_train_full = df.iloc[train_idx].reset_index(drop=True)
    df_rest = df.iloc[rest_idx].reset_index(drop=True)

    # Definies the validation and test sets
    val_test = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_idx, test_idx = next(val_test.split(df_rest, groups=df_rest[patient_col]))

    df_val  = df_rest.iloc[val_idx].reset_index(drop=True)
    df_test = df_rest.iloc[test_idx].reset_index(drop=True)

    # Subsamples the train set
    df_train_malignant = df_train_full[df_train_full['target'] == 1]
    df_train_benign = df_train_full[df_train_full['target'] == 0]

    samples = min(n_samples, len(df_train_benign))
    df_train_benign_sub = resample(df_train_benign, replace=False, n_samples=samples, random_state=42)

    df_train = pd.concat([df_train_malignant, df_train_benign_sub]).sample(frac=1, random_state=42).reset_index(drop=True)

    # Applies data augmentation on the train set
    if architecture == 'resnet50':
        img_size = 224
        resize_size = 256
    elif architecture == 'efficientnet_b1':
        img_size = 240
        resize_size = 270

    train_transform = test_transform
    if DA:
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(resize_size),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.RandomVerticalFlip(p=0.5),
            torchvision.transforms.RandomRotation(90),
            torchvision.transforms.ColorJitter(
                brightness=0.2, contrast=0.2,
                saturation=0.1, hue=0.0),
            torchvision.transforms.RandomAffine(
                degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            torchvision.transforms.RandomCrop(img_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485,0.456,0.406],
                                            std=[0.229,0.224,0.225]),
            # torchvision.transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
        ])

    # Creates the final dataset
    dataset = {
        'train': ISICDataset(df=df_train, hdf5_path=hdf5_path, transform=train_transform),
        'val'  : ISICDataset(df=df_val, hdf5_path=hdf5_path, transform=test_transform),
        'test' : ISICDataset(df=df_test, hdf5_path=hdf5_path, transform=test_transform),
        'labels': ['Benign', 'Malignant']
    }

    return dataset

def create_dataloader(dataset, batch_size, show_image=True):
    # Creates a more balanced sampler
    train_df = dataset['train'].df
    targets = train_df['target'].values
    class_counts = np.bincount(targets)

    target_proportions = np.array([0.70, 0.30])
    class_weights = target_proportions / class_counts

    sample_weights = np.array([class_weights[t] for t in targets])

    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # Defines the dataloader
    dataloader = {
        #'train' : {'data' : torch.utils.data.DataLoader(dataset['train'], batch_size=batch_size, sampler=sampler, num_workers=8, pin_memory=True) , 'length' : len(dataset['train']) },
        'train' : {'data' : torch.utils.data.DataLoader(dataset['train'], batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True), 'length' : len(dataset['train']) },
        'val'   : {'data' : torch.utils.data.DataLoader(dataset['val']  , batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True), 'length' : len(dataset['val'])   },
        'test'  : {'data' : torch.utils.data.DataLoader(dataset['test'] , batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True), 'length' : len(dataset['test'])  },
        'labels' : dataset['labels']
    }

    if show_image:
        show_images(dataloader, 10)

    return dataloader

def dataloader_stats(dataloader, dataset_dict):
    train_counts = Counter(dataset_dict['train'].df['target'].values)
    val_counts   = Counter(dataset_dict['val'].df['target'].values)
    test_counts  = Counter(dataset_dict['test'].df['target'].values)

    class_width = max(len(label) for label in dataloader['labels'])
    length_star = 15*3+class_width+5+2

    print("-" * length_star)
    print(f"|{'Class':^{class_width+2}}|{'Train (Real)':^15}|{'Val':^15}|{'Test':^15}|")
    print("-" * length_star)

    for i in range(len(dataloader['labels'])):
        train_samples = train_counts[i]
        val_samples   = val_counts[i]
        test_samples  = test_counts[i]

        train_perc = train_samples / dataloader['train']['length'] * 100
        val_perc   = val_samples / dataloader['val']['length'] * 100
        test_perc  = test_samples / dataloader['test']['length'] * 100

        print(
            f"| {dataloader['labels'][i]:<{class_width}} "
            f"|{train_perc:5.2f}% ({train_samples:>5})"
            f"|{val_perc:5.2f}% ({val_samples:>5})"
            f"|{test_perc:5.2f}% ({test_samples:>5})|"
        )

    print("-" * length_star)
    print(
        f"| {'Total':<{class_width}} " + "|"
        + f"({dataloader['train']['length']})".center(15) + "|"
        + f"({dataloader['val']['length']})".center(15) + "|"
        + f"({dataloader['test']['length']})".center(15) + "|"
    )
    print("-" * length_star)

"""# Training"""

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)

        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets.view(-1))
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def pauc_score(labels, probs, min_tpr=0.80):
    labels = np.asarray(labels)
    probs = np.asarray(probs)

    v_gt = np.abs(labels - 1)
    v_pred = 1.0 - probs

    max_fpr = abs(1 - min_tpr)

    fpr, tpr, _ = roc_curve(v_gt, v_pred)

    if max_fpr == 0 or max_fpr == 1:
        return auc(fpr, tpr)

    stop = np.searchsorted(fpr, max_fpr, "right")
    if stop >= len(fpr):
        return auc(fpr, tpr)
    
    x_interp = [fpr[stop - 1], fpr[stop]]
    y_interp = [tpr[stop - 1], tpr[stop]]
    tpr = np.append(tpr[:stop], np.interp(max_fpr, x_interp, y_interp))
    fpr = np.append(fpr[:stop], max_fpr)

    return auc(fpr, tpr)

def train(data_loader, net, epochs=100, lr=1e-4, prefix='', upper_bound=0.18, device='cpu',
          save=False, debug=False, plot_histograms=False, lambda_reg=0, models_path='models/', tensorboard_path='runs/', alpha_weights=None):

    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=lambda_reg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    loss = FocalLoss(alpha=alpha_weights, gamma=2.0)

    now = datetime.now()
    suffix = now.strftime("%Y%m%d_%H%M%S")
    prefix = prefix + '-' + suffix if prefix != '' else suffix

    log_file = open(f'training_log_{prefix}.txt', 'a', buffering=1)

    def log(msg):
        tqdm.write(msg)
        log_file.write(msg + '\n')

    writer = SummaryWriter(log_dir=tensorboard_path + prefix)
    writer.add_graph(net, next(iter(data_loader['train']['data']))[0].to(device))

    val_metrics_history = []
    train_metrics_history = []
    max_pauc  = -1.0
    best_model = copy.deepcopy(net)
    num_batches = len(data_loader['train']['data'])

    for epoch in tqdm(range(epochs), desc='Training epochs...'):
        net.train()

        epoch_train_labels = []
        epoch_train_probs = []
        epoch_loss = 0.0

        for idx, (train_x, train_label) in enumerate(data_loader['train']['data']):
            train_x = train_x.to(device)
            train_label = train_label.to(device)

            optimizer.zero_grad()
            predict_y = net(train_x)

            error = loss(predict_y, train_label.long())
            error.backward()
            optimizer.step()

            probs = torch.softmax(predict_y, dim=1)[:, 1]
            epoch_train_labels.extend(train_label.cpu().numpy())
            epoch_train_probs.extend(probs.detach().cpu().numpy())

            epoch_loss += error.item()

            if debug and idx % 10 == 0:
                log(f'Batch {idx}/{num_batches} - Loss: {error.item():.4f}')

        train_pauc = pauc_score(epoch_train_labels, epoch_train_probs)
        avg_train_loss = epoch_loss / num_batches
        train_metrics_history.append(train_pauc)

        if plot_histograms:
            plot_histograms_tensorboard(writer, net, epoch)

        val_pauc, val_loss = validate(net, data_loader['val'], device=device, criterion=loss)
        val_metrics_history.append(val_pauc)

        writer.add_scalars('Loss', {'Treino': avg_train_loss, 'Validacao': val_loss}, epoch)
        writer.add_scalars('Metrica_pAUC', {'Treino': train_pauc, 'Validacao': val_pauc}, epoch)

        scheduler.step()

        if val_pauc > max_pauc:
            best_model = copy.deepcopy(net)
            max_pauc = val_pauc
            log(f'Novo melhor modelo! pAUC Validação: {max_pauc:3.4f} (Treino: {train_pauc:3.4f})')

        log(f'Epoch: {epoch+1:3d} | Train pAUC: {train_pauc:3.4f} | Val pAUC: {val_pauc:3.4f}')

        if val_pauc > upper_bound:
            log("Meta de pAUC atingida. Parando antecipadamente.")
            break

    if save:
        path = f'{models_path}{prefix}-AUC{max_pauc:.3f}.pkl'
        torch.save(best_model.state_dict(), path)
        log(f'Modelo salvo em: {path}')

    plt.plot(train_metrics_history, label='Treino pAUC')
    plt.plot(val_metrics_history, label='Validação pAUC')
    plt.title('pAUC  ao longo das Épocas')
    plt.legend()

    plot_path = f'pAUC_{prefix}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')

    log(f'Gráfico salvo em: {plot_path}')

    plt.show()

    writer.flush()
    writer.close()
    log_file.close()

    return best_model

def is_valid_tensor(tensor):
    if tensor is None:
        return False
    if not torch.isfinite(tensor).all():
        return False
    if tensor.numel() == 0:
        return False
    if torch.isnan(tensor).any():
        return False
    return True

def plot_histograms_tensorboard ( writer, net, epoch ) :
    layers = list(net.modules())

    layer_id = 1
    linear_id = 1
    for layer in layers:
        if isinstance(layer, nn.Conv2d) :

            if is_valid_tensor(layer.bias) :
                writer.add_histogram(f'Bias/conv{layer_id}', layer.bias, epoch )

            if is_valid_tensor(layer.weight):
                writer.add_histogram(f'Weight/conv{layer_id}', layer.weight, epoch )

            if is_valid_tensor(layer.weight.grad):
                writer.add_histogram(f'Grad/conv{layer_id}', layer.weight.grad, epoch )

            layer_id += 1

        if isinstance(layer, nn.Linear) :

            if is_valid_tensor(layer.bias) :
                writer.add_histogram(f'Bias/linear{linear_id}', layer.bias, epoch )

            if is_valid_tensor(layer.weight):
                writer.add_histogram(f'Weight/linear{linear_id}', layer.weight, epoch )

            if is_valid_tensor(layer.weight.grad):
                writer.add_histogram(f'Grad/linear{linear_id}', layer.weight.grad, epoch )

            linear_id += 1

"""# Validation"""

def validate(model, data, device='cpu', criterion=None, confusion_matrix_labels=None):
    model.eval()
    error = 0.0

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for test_x, test_label in data['data']:
            test_x = test_x.to(device)
            test_label = test_label.to(device)

            predict_y = model(test_x)

            predict_ys = torch.max(predict_y, axis=1)[1]

            probs = torch.softmax(predict_y, dim=1)[:, 1]

            if criterion is not None:
                error += criterion(predict_y, test_label.long()).item()

            all_labels.extend(test_label.cpu().numpy())
            all_preds.extend(predict_ys.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    label_np = np.array(all_labels)
    pred_np = np.array(all_preds)
    prob_np = np.array(all_probs)

    p_auc = pauc_score(label_np, prob_np)

    error = error / len(data['data'])

    if confusion_matrix_labels is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        cm_norm = confusion_matrix(label_np, pred_np, normalize='true')
        disp_norm = ConfusionMatrixDisplay(np.round(cm_norm * 100, 1), display_labels=confusion_matrix_labels)
        disp_norm.plot(xticks_rotation=90, cmap='Blues', values_format='.1f')
        plt.title('Normalized Confusion Matrix (%)')
        plt.savefig(f'confusion_matrix_normalized_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()

        cm_abs = confusion_matrix(label_np, pred_np)
        disp_abs = ConfusionMatrixDisplay(cm_abs, display_labels=confusion_matrix_labels)
        disp_abs.plot(xticks_rotation=90, cmap='Blues', values_format='.0f')
        plt.title('Confusion Matrix')
        plt.savefig(f'confusion_matrix_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()

    if criterion is None:
        return p_auc
    else:
        return p_auc, error

"""# Dataloader and model definition"""

batch_size = 128

#architecture = 'resnet50'
architecture = 'efficientnet_b1'

if architecture == 'resnet50':
  transform = ResNet50_Weights.IMAGENET1K_V1.transforms()

  dataset = isic_dataset(dataset_path, transform, architecture)
  dataloader = create_dataloader(dataset, batch_size)
else:
  transform = EfficientNet_B1_Weights.IMAGENET1K_V1.transforms()

  dataset = isic_dataset(dataset_path, transform, architecture)
  dataloader = create_dataloader(dataset, batch_size)

dataloader_stats(dataloader, dataset)

"""# Fine tunning"""

if architecture == 'resnet50':
  model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
  for param in model.parameters():
    param.requires_grad = False

  model.fc = torch.nn.Sequential(
        torch.nn.Dropout(p=0.3),
        torch.nn.Linear(2048, len(dataloader['labels']))
   )

  for param in model.layer4.parameters():
    param.requires_grad = True
else:
  model = efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V1)
  for param in model.parameters():
    param.requires_grad = False

  num_features = model.classifier[1].in_features
  model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(num_features, len(dataloader['labels']))
  )

  for param in model.features[7].parameters():
    param.requires_grad = True
  for param in model.features[8].parameters():
    param.requires_grad = True

if torch.cuda.is_available():
    my_device = torch.device("cuda")
else:
    my_device = torch.device("cpu")

print(f"Running on {my_device.type}")

pesos_alpha = torch.tensor([0.05, 0.95]).to(my_device)
#pesos_alpha = None

model = model.to(my_device)

epochs = 30
lr = 1e-4
lambda_reg = 1e-4
prefix = '{}-TL-e-{}-lr-{}'.format(architecture, epochs, lr)

net = train(dataloader, model,
            epochs=epochs, device=my_device, save=True,
            prefix=prefix, lr=lr, plot_histograms=True, lambda_reg = lambda_reg, models_path=models_path, tensorboard_path=logs_path, alpha_weights=pesos_alpha)

test_metric = validate ( net ,
                          dataloader['test'] ,
                          device=my_device,
                          confusion_matrix_labels=dataloader['labels']
                         )

if isinstance(test_metric, tuple):
    print(f"pAUC in test: {test_metric[0]:.4f}")
else:
    print(f"pAUC in test: {test_metric:.4f}")