import os
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_auc_score, precision_score, recall_score, f1_score
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
from combined_loss import combined_loss

THRESHOLD = 0.4

def save_checkpoint(state, filename):
    torch.save(state, filename)

def train(model, optimizer, train_loader, test_loader, external_loader, epochs : int):
    model.train()

    train_epoch_losses, train_epoch_auc = [], []
    
    for epoch in range(epochs):
        model.train()
        losses = []
        total_samples = 0
        
        total_positives, true_positives = 0, 0
        total_negatives, true_negatives = 0, 0
        false_positives, false_negatives = 0, 0
        all_logits, all_labels, all_probs, all_preds = [], [], [], []
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{epochs}")

        for batch_index, batch in progress_bar:
            optimizer.zero_grad()
            (t0_first_images, t1_first_images, t2_first_images, t3_first_images), non_mri, labels, pids = batch

            t0_first_images = t0_first_images.cuda()
            t3_first_images = t3_first_images.cuda()

            labels = labels.cuda().float()
            if len(labels) == 1:
                continue
            
            logits = model(t0_first_images, t3_first_images, non_mri).squeeze()

            all_logits.extend(logits.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

            loss = combined_loss(logits, labels)
            losses.append(loss.item())
            probs = torch.sigmoid(logits)
            preds = (probs > THRESHOLD).int()

            all_probs.extend(probs.detach().cpu().numpy())
            all_preds.extend(preds.detach().cpu().numpy())
            total_samples += labels.size(0)

            for gt, pred in zip(labels, preds):
                if gt:
                    total_positives += 1
                    if pred:
                        true_positives += 1
                    else:
                        false_negatives += 1
                else:
                    total_negatives += 1
                    if pred:
                        false_positives += 1
                    else:
                        true_negatives += 1
                
            loss.backward()
            optimizer.step()

        epoch_loss = np.mean(losses)
        epoch_accuracy = (true_positives + true_negatives) / total_samples

        all_logits = np.array(all_logits)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        all_preds = np.array(all_preds)

        epoch_auc = roc_auc_score(all_labels, all_probs)
        epoch_sensitivity = true_positives / total_positives if total_positives != 0 else 0
        epoch_specificity = true_negatives / total_negatives if total_negatives != 0 else 0

        epoch_precision = precision_score(all_labels, all_preds, zero_division=0)
        epoch_recall = recall_score(all_labels, all_preds, zero_division=0)

        epoch_f1 = f1_score(all_labels, all_preds, zero_division=0)

        train_epoch_losses.append(epoch_loss)
        train_epoch_auc.append(epoch_auc)

        print("LOSS: {:.3f}".format(epoch_loss), 
        " ACCURACY: {:.3f}".format(epoch_accuracy), 
        " AUC: {:.3f}".format(epoch_auc),
        " F1: {:.3f}".format(epoch_f1),
        " PRECISION: {:.3f}".format(epoch_precision), 
        " RECALL/SENSITIVITY: {:.3f}".format(epoch_recall), 
        " SPECIFICITY: {:.3f}".format(epoch_specificity))
   
def test(model, test_loader):
    model.eval()
    
    losses = []
    total_samples = 0
    
    total_positives, true_positives = 0, 0
    total_negatives, true_negatives = 0, 0
    false_positives, false_negatives = 0, 0

    all_logits, all_labels, all_probs, all_preds = [], [], [], []
    with torch.no_grad():
        for batch_index, batch in enumerate(test_loader):
            (t0_first_images, t1_first_images, t2_first_images, t3_first_images), non_mri, labels, pids = batch

            t0_first_images = t0_first_images.cuda()
            t3_first_images = t3_first_images.cuda()

            labels = labels.cuda().float()
            
            logits = model(t0_first_images, t3_first_images, non_mri).squeeze()

            all_logits.extend(logits.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

            loss = combined_loss(logits, labels)
            losses.append(loss.item())
            probs = torch.sigmoid(logits)
            preds = (probs > THRESHOLD).int()

            all_probs.extend(probs.detach().cpu().numpy())
            all_preds.extend(preds.detach().cpu().numpy())
            total_samples += labels.size(0)

            # check positives: gt 1, pred 1
            for gt, pred in zip(labels, preds):
                if gt:
                    total_positives += 1
                    if pred:
                        true_positives += 1
                    else:
                        false_negatives += 1
                else:
                    total_negatives += 1
                    if pred:
                        false_positives += 1
                    else:
                        true_negatives += 1

    epoch_loss = np.mean(losses)
    epoch_accuracy = (true_positives + true_negatives) / total_samples

    all_logits = np.array(all_logits)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)

    # Calculate AUC, sensitivity/specificity
    epoch_auc = roc_auc_score(all_labels, all_probs)
    epoch_sensitivity = true_positives / total_positives if total_positives != 0 else 0
    epoch_specificity = true_negatives / total_negatives if total_negatives != 0 else 0

    epoch_precision = precision_score(all_labels, all_preds, zero_division=0)
    epoch_recall = recall_score(all_labels, all_preds, zero_division=0)

    epoch_f1 = f1_score(all_labels, all_preds, zero_division=0)

    print("LOSS: {:.3f}".format(epoch_loss), 
    " ACCURACY: {:.3f}".format(epoch_accuracy), 
    " AUC: {:.3f}".format(epoch_auc),
    " F1: {:.3f}".format(epoch_f1),
    " PRECISION: {:.3f}".format(epoch_precision), 
    " RECALL/SENSITIVITY: {:.3f}".format(epoch_recall), 
    " SPECIFICITY: {:.3f}".format(epoch_specificity))