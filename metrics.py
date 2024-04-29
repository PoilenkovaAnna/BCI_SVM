from sklearn.metrics import f1_score
import torch

def calculate_f1score(model, test_dataset):
    f1 = 0

    with torch.no_grad():
        model.eval()
        for morlet, label in test_dataset:
            output = model(morlet[...].float().to(device))

            predicted = torch.argmax(output, 1)
            f1 += f1_score(label, predicted.cpu().detach().numpy(), average='weighted')

    return f1 / len(test_dataset)


def calculate_accuracy(model, test_dataset, criterion, device):

    total_correct, total_instances, ep_loss = 0, 0, 0

    with torch.no_grad():

        model.eval()
        for morlets, labels, indexs in test_dataset:
            morlets = morlets.float()

            morlets, labels = morlets.to(device), labels.to(device)

            logits = model(morlets)
            loss = criterion(logits, labels)

            ep_loss += loss.item()

            predicted = torch.argmax(logits, dim=1)

            total_correct += sum(predicted == labels).item()
            total_instances+=len(labels)

    return total_correct/ total_instances,  ep_loss/ total_instances