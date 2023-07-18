import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import models
from data import load_data

best_test_acc = 0.0


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=20):
    global best_test_acc

    def train():
        model.train(True)
        train_correct = 0
        train_total = 0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs = Variable(inputs.to(device))
            labels = Variable(labels.to(device))
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_total += labels.size(0)
            _, train_predictions = torch.max(outputs.data, 1)
            train_correct += (train_predictions == labels.data).sum()
        train_acc_0 = 100 * train_correct.double() / train_total
        train_acc = train_acc_0.cpu().numpy()
        print('train_acc =', ('%.3f' % train_acc))
        return train_acc

    def validate():
        model.train(False)
        val_correct = 0
        val_total = 0
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            val_total += labels.size(0)
            _, val_predictions = torch.max(outputs.data, 1)
            val_correct += (val_predictions == labels.data).sum()
        val_acc_0 = 100 * val_correct.double() / val_total
        val_acc = val_acc_0.cpu().numpy()
        print('val_acc =', ('%.3f' % val_acc))
        return val_acc

    def test():
        global best_test_acc
        model.train(False)
        test_correct = 0
        test_total = 0
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            test_total += labels.size(0)
            _, test_predictions = torch.max(outputs.data, 1)
            test_correct += (test_predictions == labels.data).sum()
        test_acc_0 = 100 * test_correct.double() / test_total
        if test_acc_0 > best_test_acc:
            best_test_acc = test_acc_0
        test_acc = test_acc_0.cpu().numpy()
        print('test_acc =', ('%.3f' % test_acc))
        return test_acc

    train_acc = np.zeros(num_epochs)
    val_acc = np.zeros(num_epochs)
    test_acc = np.zeros(num_epochs)

    for epoch in range(num_epochs):
        print('The', epoch + 1, 'time of training and testing')
        train_acc[epoch] = train()
        val_acc[epoch] = validate()
        scheduler.step(val_acc[epoch])
        test_acc[epoch] = test()
        if test_acc[epoch] > best_test_acc:
            best_model = model
            torch.save(best_model, 'best_model.pt')

    plot_curves(num_epochs, train_acc, val_acc, test_acc)
    print(f'best_acc{best_test_acc}')


def plot_curves(epoch, train_acc, val_acc, test_acc):
    epochs = range(1, epoch + 1)
    plt.plot(epochs, train_acc, ls='-', lw=2, label='train accuracy', color='b')
    plt.plot(epochs, val_acc, ls='-', lw=2, label='val accuracy', color='g')
    plt.plot(epochs, test_acc, ls='-', lw=2, label='test accuracy', color='r')
    plt.legend()
    train_max_indx = np.argmax(train_acc)
    val_max_indx = np.argmax(val_acc)
    test_max_indx = np.argmax(test_acc)
    plt.plot(train_max_indx + 1, train_acc[train_max_indx], 'ks')
    plt.plot(val_max_indx + 1, val_acc[val_max_indx], 'gs')
    plt.plot(test_max_indx + 1, test_acc[test_max_indx], 'rs')
    show_max_1 = '[' + str('Best Training accuracy') + ' ' + ('%.4f' % train_acc[train_max_indx]) + ']'
    show_max_2 = '[' + str('Best Validation accuracy') + ' ' + ('%.4f' % val_acc[val_max_indx]) + ']'
    show_max_3 = '[' + str('Best Test accuracy') + ' ' + ('%.4f' % test_acc[test_max_indx]) + ']'
    plt.annotate(show_max_1, xy=(train_max_indx + 1, train_acc[train_max_indx]), xytext=(25, 75))
    plt.annotate(show_max_2, xy=(val_max_indx + 1, val_acc[val_max_indx]), xytext=(25, 70))
    plt.annotate(show_max_3, xy=(test_max_indx + 1, test_acc[test_max_indx]), xytext=(25, 65))
    plt.title('Training, Validation, and Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.savefig('./acc.png')
    plt.show()


if __name__ == '__main__':
    num_classes = 2
    data_dir = "../data/"
    input_size = 224
    batch_size = 64
    train_val_split = 0.8

    num_epochs = 150
    lr = 0.001

    model = models.model_creator(num_classes=num_classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_loader, val_loader, test_loader = load_data(data_dir=data_dir, input_size=input_size,
                                                      batch_size=batch_size, train_val_split=train_val_split)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)

    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=num_epochs)