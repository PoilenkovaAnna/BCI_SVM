import torch
import torch.nn as nn
from metrics import calculate_f1score, calculate_accuracy


def train_model(model , train_loader, device, optimizer, criterion, epochs, scheduler = None):

    for epoch in range(epochs):
        correct, total, ep_loss = 0, 0, 0
        model.train()
        for iter,  batch in enumerate(train_loader):

            morlets, labels, indexs =  batch
            morlets = morlets.float()

            logits = model(morlets.to(device))

            if logits.shape[-1] == 1: # бинарная
                logits = torch.sigmoid(logits.flatten())
                labels = labels.float()
                loss = criterion(logits, labels.to(device))
            else:
                loss = criterion(logits, labels.to(device)) # Многоклассовая

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

def train_model_plot( pp, model , train_loader, test_loader, device, optimizer, criterion, epochs, scheduler = None):
    for epoch in range(epochs):

        total_correct, total_instances, ep_loss = 0, 0, 0
        model.train()
        for iter,  batch in enumerate(train_loader):

            morlets, labels, indexs =  batch
            morlets = morlets.float()
            morlets, labels = morlets.to(device), labels.to(device)

            optimizer.zero_grad()

            logits = model(morlets)
            loss = criterion(logits, labels) # Многоклассовая

            loss.backward()
            optimizer.step()

            ep_loss += loss.item()

            predicted = torch.argmax(logits, dim=1)

            total_correct += sum(predicted == labels).item()
            total_instances+=len(labels)

        if scheduler is not None:
            scheduler.step()

        # Logging
        pp.add_scalar('loss_train', ep_loss/ total_instances)
        pp.add_scalar('accuracy_train', total_correct/ total_instances)

        accuracy, loss_val = calculate_accuracy(model, test_loader, criterion, device)
        pp.add_scalar('loss_val',loss_val)
        pp.add_scalar('accuracy_val',accuracy)

        pp.display([['loss_train','loss_val'],['accuracy_train','accuracy_val']])
    return pp

def train_model_plot_modern(pp, model , train_loader, batch_sampler,  test_loader, device, optimizer, criterion, epochs, scheduler = None):

    criterion_none = nn.CrossEntropyLoss(reduction = 'none')

    threshold_epoch = 7

    for epoch in range(epochs):

        removed_indexs_list = []

        total_correct, total_instances, ep_loss = 0, 0, 0
        model.train()
        for iter,  batch in enumerate(train_loader):

            morlets, labels, indexs =  batch
            morlets = morlets.float()
            morlets, labels = morlets.to(device), labels.to(device)

            optimizer.zero_grad()

            logits = model(morlets)
            loss = criterion(logits, labels) # Многоклассовая
            loss_none = criterion_none(logits, labels)

            # начиная с threshold_epoch убираем обьекты с большим loss
            if threshold_epoch <= epoch:
               # print(f'indexs {indexs.tolist()}')
               # print(f'loss_none {loss_none.tolist()}')
               # print("====================")
                zipped = list(zip(loss_none.tolist(), indexs.tolist()))

              #  print("zipped")
               # print(zipped)
                removed_indexs_list+=zipped # add morlets, labels
               # print("removed_indexs_list")
               # print(removed_indexs_list)

            loss.backward()
            optimizer.step()

            ep_loss += loss.item()

            predicted = torch.argmax(logits, dim=1)

            total_correct += sum(predicted == labels).item()
            total_instances+=len(labels)

        # delete datapoint with hight loss
        if threshold_epoch <= epoch:
            sort = sorted(zipped, key=lambda x: x[0])
            deleted_indexes = [indexs for loss_none, indexs in sort ]
            print([loss_none for loss_none, indexs in sort ])
            print(f'before len batch_sampler {len(batch_sampler)}')
            batch_sampler.delete(deleted_indexes[len(deleted_indexes)-2:])
            print(f'after len batch_sampler {len(batch_sampler)}')



        if scheduler is not None:
            scheduler.step()

        # Logging
        pp.add_scalar('loss_train', ep_loss/ total_instances)
        pp.add_scalar('accuracy_train', total_correct/ total_instances)

        accuracy, loss_val = calculate_accuracy(model, test_loader, criterion, device)
        pp.add_scalar('loss_val',loss_val)
        pp.add_scalar('accuracy_val',accuracy)

        pp.display([['loss_train','loss_val'],['accuracy_train','accuracy_val']])
    return pp