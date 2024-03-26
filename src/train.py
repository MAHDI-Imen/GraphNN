from tqdm import tqdm
from copy import deepcopy
import numpy as np

import torch
from sklearn.metrics import f1_score, accuracy_score


#####################################################
############### TEST FUNCTION #######################
#####################################################
def evaluate_ppi(model, device, dataloader):
    score_list_batch = []

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch = batch.to(device)
            output = model(batch.x, batch.edge_index)
            predict = np.where(output.detach().cpu().numpy() >= 0, 1, 0)
            score = f1_score(batch.y.cpu().numpy(), predict, average="micro")
            score_list_batch.append(score)

            del batch
            del output
            torch.cuda.empty_cache()

    return np.array(score_list_batch).mean()

def evaluate_graph_classification(model, device, dataloader, num_classes):
    score_list_batch = []

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch = batch.to(device)
            output = model(batch.x, batch.edge_index, batch.batch)
            if num_classes>=2:
                pred = output.argmax(dim=1)
            else:
                pred = (output > 0.5).int()
            score = f1_score(batch.y.cpu().numpy(), pred.cpu().numpy(), average="micro")
            score_list_batch.append(score)

            del batch
            del output
            torch.cuda.empty_cache()

    return np.array(score_list_batch).mean()


#####################################################
############### TRAIN FUNCTION #######################
#####################################################
def train_ppi(
    model,
    loss_fcn,
    device,
    optimizer,
    max_epochs,
    train_dataloader,
    val_dataloader,
    save_best=False,
):

    best_model_state_dict = None
    best_score = 0

    epoch_list = []
    scores_list = []

    # loop over epochs
    with tqdm(range(max_epochs), unit="epoch") as t:
        for epoch in range(max_epochs):
            model.train()
            losses = []
            # loop over batches
            for i, train_batch in enumerate(train_dataloader):
                optimizer.zero_grad()
                train_batch_device = train_batch.to(device)
                # logits is the output of the model
                logits = model(train_batch_device.x, train_batch_device.edge_index)
                # compute the loss
                loss = loss_fcn(logits, train_batch_device.y)
                # optimizer step
                loss.backward()
                optimizer.step()
                losses.append(loss.cpu().detach().item())

            loss_data = np.array(losses).mean()

            # evaluate the model on the validation set
            score = evaluate_ppi(model, device, val_dataloader)
            if score > best_score:
                best_score = score
                if save_best:
                    best_model_state_dict = deepcopy(model.state_dict())

            t.set_postfix(loss=loss_data, f1=score)
            t.update(1)

            scores_list.append(score)
            epoch_list.append(epoch)

            del train_batch_device
            del logits
            del loss
            torch.cuda.empty_cache()

    # Load the best model state
    if save_best and best_model_state_dict is not None:
        model.load_state_dict(best_model_state_dict)
        print("Best model loaded")

    return epoch_list, scores_list


def train_graph_classification(
    model,
    loss_fcn,
    device,
    optimizer,
    max_epochs,
    train_dataloader,
    val_dataloader,
    num_classes,
    save_best=False,
):

    best_model_state_dict = None
    best_score = 0

    epoch_list = []
    scores_list = []

    # loop over epochs
    with tqdm(range(max_epochs), unit="epoch") as t:
        for epoch in range(max_epochs):
            model.train()
            losses = []
            # loop over batches
            for i, train_batch in enumerate(train_dataloader):
                optimizer.zero_grad()
                train_batch_device = train_batch.to(device)
                # logits is the output of the model
                logits = model(train_batch_device.x, train_batch_device.edge_index, train_batch_device.batch)
                if num_classes>=2:
                    # one hot encoding of y
                    one_hot_y = torch.nn.functional.one_hot(train_batch_device.y, num_classes=num_classes).float()
                    # compute the loss
                    loss = loss_fcn(logits, one_hot_y)
                else:
                    loss = loss_fcn(logits, train_batch_device.y.unsqueeze(-1).float())

                # optimizer step
                loss.backward()
                optimizer.step()
                losses.append(loss.cpu().detach().item())

            loss_data = np.array(losses).mean()

            # evaluate the model on the validation set
            score = evaluate_graph_classification(model, device, val_dataloader, num_classes)
            if score > best_score:
                best_score = score
                if save_best:
                    best_model_state_dict = deepcopy(model.state_dict())

            t.set_postfix(loss=loss_data, f1=score)
            t.update(1)

            scores_list.append(score)
            epoch_list.append(epoch)

            del train_batch_device
            del logits
            del loss
            torch.cuda.empty_cache()

    # Load the best model state
    if save_best and best_model_state_dict is not None:
        model.load_state_dict(best_model_state_dict)
        print("Best model loaded")

    return epoch_list, scores_list