import json
import matplotlib.pyplot as plt
import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torchvision.models as models
import torch.optim as optim


train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408),
    (0.2675, 0.2565, 0.2761)),])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408),
    (0.2675, 0.2565, 0.2761)),])


# Download the CIFAR-100 dataset and apply the transformations
trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
num_data = len(trainset)
testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)
# Create train and test loaders to feed data in batches to the model
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,shuffle=False, num_workers=2)


if torch.cuda.is_available():
  device = "cuda"
else:
  device = "cpu"
print(f"Using {device} device")


# Teacher Model
class TeacherNet(nn.Module):
    def __init__(self, num_classes=100):
        super(TeacherNet, self).__init__()
        self.model = models.resnet50(pretrained=False, num_classes=100)

    def forward(self, x):
        x = self.model(x)
        return x
    

#student model
class StudentNet(nn.Module):
    def __init__(self, num_classes=100):
        super(StudentNet, self).__init__()
        self.model = models.mobilenet_v2(pretrained=False, num_classes=100)

    def forward(self, x):
        x = self.model(x)
        return x



num_epochs = 240
num_classes = 100
teacher_model = TeacherNet(num_classes=100).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(teacher_model.parameters(), lr=0.05,
                      momentum=0.9, weight_decay=5e-4)
#lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 180, 210], gamma=0.1)


#train teacher
test_accs = []
for epoch in range(num_epochs):
    train_loss = 0
    train_correct = 0
    total = 0

    # Set the model to training mode
    teacher_model.train()

    for i, (inputs, labels) in enumerate(trainloader):
        # Move the inputs and labels to the device (GPU) if available
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = teacher_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Compute the training accuracy
        _, predicted = torch.max(outputs.data, 1)
        train_correct += (predicted == labels).sum().item()
        total += labels.size(0)
        train_loss += loss.item() * labels.size(0)

    # Compute the average training loss and accuracy

    train_loss /= total
    train_acc = 100 * train_correct / total
    print(
        f"Epoch [{epoch+1}/{num_epochs}] - - Train Loss: {train_loss: .4f}, Train Acc: {train_acc: .2f}")

    test_loss = 0
    test_correct = 0
    total = 0
    teacher_model.eval()
    with torch.no_grad():

        for inputs, labels in testloader:
            # Move the inputs and labels to the device (GPU) if available
            inputs, labels = inputs.to(device), labels.to(device)
            # Compute the model's predictions
            outputs = teacher_model(inputs)

            # Compute the testing loss
            loss = criterion(outputs, labels)
            test_loss += loss.item() * labels.size(0)

            # Compute the testing accuracy
            _, predicted = torch.max(outputs.data, 1)
            test_correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # Compute the average testing loss and accuracy
        test_loss /= total
        test_acc = 100 * test_correct / total
        test_accs.append(test_acc)

        # Print the epoch number, training loss and accuracy, and testing loss and accuracy
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    lr_scheduler.step()

mean_test_acc = sum(test_accs) / len(test_accs)
print('Mean Test Accuracy: {:.2f}%'.format(mean_test_acc))


print('Finished Training')
print('----------------------------------------------------------------')
#torch.save(teacher_model.state_dict(), "teacherResNet50.pth")
path = "saved_outputs"
os.makedirs(path, exist_ok=True)

# Save the output tensor to a file
torch.save(teacher_model.state_dict(),
           os.path.join(path, "teacherResNet50.pth"))




#train student
num_epochs = 240
num_classes = 100
student_model = StudentNet(num_classes=100).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(student_model.parameters(), lr=0.01,
                      momentum=0.9, weight_decay=5e-4)
# lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 180, 210], gamma=0.1)

std_test_accs = []
for epoch in range(num_epochs):
    train_loss = 0
    train_correct = 0
    total = 0

    # Set the model to training mode
    student_model.train()

    for i, (inputs, labels) in enumerate(trainloader):
        # Move the inputs and labels to the device (GPU) if available
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = student_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Compute the training accuracy
        _, predicted = torch.max(outputs.data, 1)
        train_correct += (predicted == labels).sum().item()
        total += labels.size(0)
        train_loss += loss.item() * labels.size(0)

    # Compute the average training loss and accuracy
    train_loss /= total
    train_acc = 100 * train_correct / total
    print(
        f"Epoch [{epoch+1}/{num_epochs}] - - Train Loss: {train_loss: .4f}, Train Acc: {train_acc: .2f}")

    test_loss = 0
    test_correct = 0
    total = 0
    student_model.eval()
    with torch.no_grad():

        for inputs, labels in testloader:
            # Move the inputs and labels to the device (GPU) if available
            inputs, labels = inputs.to(device), labels.to(device)
            # Compute the model's predictions
            outputs = student_model(inputs)

            # Compute the testing loss
            loss = criterion(outputs, labels)
            test_loss += loss.item() * labels.size(0)

            # Compute the testing accuracy
            _, predicted = torch.max(outputs.data, 1)
            test_correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # Compute the average testing loss and accuracy
        test_loss /= total
        test_acc = 100 * test_correct / total
        std_test_accs.append(test_acc)

        # Print the epoch number, training loss and accuracy, and testing loss and accuracy
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    lr_scheduler.step()

mean_test_acc = sum(std_test_accs) / len(std_test_accs)
print('Mean Test Accuracy: {:.2f}%'.format(mean_test_acc))

print('Finished Standard Student Training')
print('----------------------------------------------------------------')


#classical KD with student training
kd_student_model = StudentNet(num_classes=100).to(device)
teacher_model = TeacherNet(num_classes=100).to(device)
teacher_model.load_state_dict(torch.load(os.path.join(path, "teacherResNet50.pth")))


def kd_loss(logits_student, logits_teacher, temperature):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher,
                       reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(kd_student_model.parameters(), lr=0.01, momentum=0.9)
# lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 180, 210], gamma=0.1)

T = 4.0  # set temperature to 1.0 ~ 20.0
kd_test_accs = []
for epoch in range(num_epochs):
    train_loss = 0
    train_correct = 0
    total = 0

    # Set the model to training mode
    kd_student_model.train()

    for i, (inputs, labels) in enumerate(trainloader):
        # Move the inputs and labels to the device (GPU) if available
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        teacher_logits = teacher_model(inputs).detach()
        outputs = kd_student_model(inputs)
        loss = kd_loss(outputs, teacher_logits, T)
        loss.backward()
        optimizer.step()

        # Compute the training accuracy
        _, predicted = torch.max(outputs.data, 1)
        train_correct += (predicted == labels).sum().item()
        total += labels.size(0)
        train_loss += loss.item() * labels.size(0)

    # Compute the average training loss and accuracy
    train_loss /= total
    train_acc = 100 * train_correct / total
    print(
        f"Epoch [{epoch+1}/{num_epochs}] - - Train Loss: {train_loss: .4f}, Train Acc: {train_acc: .2f}")

    test_loss = 0
    test_correct = 0
    total = 0
    kd_student_model.eval()
    with torch.no_grad():

        for inputs, labels in testloader:
            # Move the inputs and labels to the device (GPU) if available
            inputs, labels = inputs.to(device), labels.to(device)
            # Compute the model's predictions
            teacher_logits = teacher_model(inputs).detach()
            outputs = kd_student_model(inputs)

            # Compute the testing loss
            loss = kd_loss(outputs, teacher_logits, T)
            test_loss += loss.item() * labels.size(0)

            # Compute the testing accuracy
            _, predicted = torch.max(outputs.data, 1)
            test_correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # Compute the average testing loss and accuracy
        test_loss /= total
        test_acc = 100 * test_correct / total
        kd_test_accs.append(test_acc)

        # Print the epoch number, training loss and accuracy, and testing loss and accuracy
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    lr_scheduler.step()

mean_test_acc = sum(kd_test_accs) / len(kd_test_accs)
print('Mean Test Accuracy: {:.2f}%'.format(mean_test_acc))

print('Finished KD Student Training')
print('----------------------------------------------------------------')


num_epochs = 240
num_classes = 100
dkd_student_model = StudentNet(num_classes=100).to(device)
teacher_model = TeacherNet(num_classes=100).to(device)
teacher_model.load_state_dict(torch.load(os.path.join(path, "teacherResNet50.pth")))


def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
        F.kl_div(log_pred_student_part2,
                 pred_teacher_part2, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    # print(f"TCKD_lOSS:{tckd_loss}, NCKD_LOSS: {nckd_loss}")
    return alpha * tckd_loss + beta * nckd_loss


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(dkd_student_model.parameters(), lr=0.01, momentum=0.9)
# lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 180, 210], gamma=0.1)


alpha = 1.0
beta = 2.0
T = 4.0

dkd_test_accs = []
for epoch in range(num_epochs):
    train_loss = 0
    train_correct = 0
    total = 0

    # Set the model to training mode
    dkd_student_model.train()

    for i, (inputs, labels) in enumerate(trainloader):
        # Move the inputs and labels to the device (GPU) if available
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        teacher_logits = teacher_model(inputs).detach()
        outputs = dkd_student_model(inputs)
        loss = dkd_loss(outputs, teacher_logits, labels, alpha, beta, T)
        loss.backward()
        optimizer.step()

        # Compute the training accuracy
        _, predicted = torch.max(outputs.data, 1)
        train_correct += (predicted == labels).sum().item()
        total += labels.size(0)
        train_loss += loss.item() * labels.size(0)

    # Compute the average training loss and accuracy
    train_loss /= total
    train_acc = 100 * train_correct / total
    print(
        f"Epoch [{epoch+1}/{num_epochs}] - - Train Loss: {train_loss: .4f}, Train Acc: {train_acc: .2f}")

    test_loss = 0
    test_correct = 0
    total = 0
    dkd_student_model.eval()
    with torch.no_grad():

        for inputs, labels in testloader:
            # Move the inputs and labels to the device (GPU) if available
            inputs, labels = inputs.to(device), labels.to(device)
            # Compute the model's predictions
            teacher_logits = teacher_model(inputs).detach()
            outputs = dkd_student_model(inputs)

            # Compute the testing loss
            loss = dkd_loss(outputs, teacher_logits, labels, alpha, beta, T)
            test_loss += loss.item() * labels.size(0)

            # Compute the testing accuracy
            _, predicted = torch.max(outputs.data, 1)
            test_correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # Compute the average testing loss and accuracy
        test_loss /= total
        test_acc = 100 * test_correct / total
        dkd_test_accs.append(test_acc)

        # Print the epoch number, training loss and accuracy, and testing loss and accuracy
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    lr_scheduler.step()

mean_test_acc = sum(dkd_test_accs) / len(dkd_test_accs)
print('Mean Test Accuracy: {:.2f}%'.format(mean_test_acc))

print('Finished DKD Student Training')
print('----------------------------------------------------------------')


# """

#   Plot 3 graphs, one for KD, one for DKD, and one without
#   any knowledge distillation on student model for validation data.           
#   Set x-axis to number of epochs and y-axis to accuracy. Set legend
#   equal to 'DKD-std-acc', 'KD-std-acc', 'std-acc'.

# """
# legend = ['DKD-std-acc', 'KD-std-acc', 'std-acc']
# #############################################################################
# # PLACE YOUR CODE HERE                                                      #
# #############################################################################
# epochs = 240
# plt.figure(figsize=(10, 10))
# plt.subplot(2, 1, 1)
# plt.plot(range(1, epochs+1), dkd_test_accs, 'r', label='DKD-std-acc')
# plt.plot(range(1, epochs+1), kd_test_accs, 'b', label='KD-std-acc')
# plt.plot(range(1, epochs+1), std_test_accs, 'g', label='std-acc')
# plt.title('Validation Accuracy: KD vs DKD vs No-KD')
# plt.xlabel('epochs')
# plt.ylabel('accuracy')

# #############################################################################
# plt.legend(legend, loc='upper left')
# plt.show()


with open('test_acc_outputs.json', 'w') as f:
    json.dump([dkd_test_accs, kd_test_accs, std_test_accs], f)
