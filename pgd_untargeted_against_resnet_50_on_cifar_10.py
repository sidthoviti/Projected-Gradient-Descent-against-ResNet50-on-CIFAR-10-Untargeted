import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.autograd import Variable
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
import random


def load_dataset(batch_size):
    # Set dataset path
    dataset_path = './data/cifar10'

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Load CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(root=dataset_path, train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=dataset_path, train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                             shuffle=False, num_workers=2)

    # Class names for CIFAR-10 dataset
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    return trainset, trainloader, testset, testloader, classes


def train(model, trainloader, criterion, optimizer, device):
    train_loss = 0.0
    train_total = 0
    train_correct = 0

    # Switch to train mode
    model.train()

    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Update training loss
        train_loss += loss.item() * inputs.size(0)

        # Compute training accuracy
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    # Compute average training loss and accuracy
    train_loss = train_loss / len(trainloader.dataset)
    train_accuracy = 100.0 * train_correct / train_total

    return model, train_loss, train_accuracy


def test(model, testloader, criterion, device):
    test_loss = 0.0
    test_total = 0
    test_correct = 0

    # Switch to evaluation mode
    model.eval()

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Update test loss
            test_loss += loss.item() * inputs.size(0)

            # Compute test accuracy
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    # Compute average test loss and accuracy
    test_loss = test_loss / len(testloader.dataset)
    test_accuracy = 100.0 * test_correct / test_total

    return test_loss, test_accuracy


def train_epochs(model, trainloader, testloader, criterion, optimizer, device, num_epochs, save_interval=5):
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        model, train_loss, train_accuracy = train(model, trainloader, criterion, optimizer, device)
        test_loss, test_accuracy = test(model, testloader, criterion, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        print(f'Train Loss: {train_loss:.4f} - Train Accuracy: {train_accuracy:.2f}%')
        print(f'Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy:.2f}%')
        print()

        # Save the model if the current test accuracy is higher than the best accuracy
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            checkpoint = {
                'epoch' : epoch,
                'model_state_dict' : model.state_dict(),
                #'optimizer_state_dict': optimizer.state_dict(),
                'test_accuracy' : test_accuracy
            }
            torch.save(checkpoint, 'best_model.pth')

    return model, train_losses, train_accuracies, test_losses, test_accuracies

def fgsm_attack(model, criterion, images, labels, device, epsilon):
    original_images = images.clone().detach().to(device)

    images.requires_grad_(True)
    outputs = model(images)
    loss = criterion(outputs, labels).to(device)
    model.zero_grad()
    loss.backward()
    gradient = images.grad.data

    perturbations = epsilon * torch.sign(gradient)
    with torch.no_grad():
        adversarial_images = images + perturbations
        perturbations = torch.clamp(adversarial_images - original_images, min=-epsilon, max=epsilon)
        adversarial_images = torch.clamp(original_images + perturbations, 0, 1)

    return adversarial_images, perturbations



def pgd_attack(model, criterion, images, labels, device, epsilon, alpha, num_iters):
    original_images = images.clone().detach().to(device)
    adversarial_images = images.clone().detach().to(device)

    for _ in range(num_iters):
        # Compute gradients within the loop
        adversarial_images.requires_grad_(True)
        outputs = model(adversarial_images)
        loss = criterion(outputs, labels)

        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            gradient = adversarial_images.grad.data
            adversarial_images = adversarial_images + alpha * gradient.sign()
            #perturbations = alpha * torch.sign(gradient)

            perturbations = torch.clamp(adversarial_images - original_images, min=-epsilon, max=epsilon)
            adversarial_images = torch.clamp(original_images + perturbations, 0, 1)

        adversarial_images = adversarial_images.detach()
    return adversarial_images, perturbations

def test_adversarial(model, testloader, criterion, device, epsilon, alpha, num_iters, attack_type='PGD'):
    adversarial_correct = 0
    attack_success = 0
    total = 0

    model.eval()

    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)

        if attack_type == 'PGD':
            adversarial_images, _ = pgd_attack(model, criterion, images, labels, device, epsilon, alpha, num_iters)
        elif attack_type == 'FGSM':
            adversarial_images, _ = fgsm_attack(model, criterion, images, labels, device, epsilon)
        else:
            raise ValueError("Unknown attack type. Supported types: 'PGD', 'FGSM'")

        adversarial_outputs = model(adversarial_images)
        _, adversarial_predicted = torch.max(adversarial_outputs.data, 1)

        adversarial_correct += (adversarial_predicted == labels).sum().item()
        attack_success += (adversarial_predicted != labels).sum().item()
        total += labels.size(0)

    adversarial_accuracy = 100.0 * adversarial_correct / total
    attack_success_rate = 100.0 * attack_success / total
    print(f'{attack_type} Attack - Epsilon = {epsilon}:')
    print(f'Adversarial Accuracy: {adversarial_accuracy:.2f}%')
    print(f'Attack Success Rate: {attack_success_rate:.2f}%')
    print('------------------------------------------------------')
    return adversarial_accuracy, attack_success_rate


def plot_loss(train_losses, test_losses):
    plt.figure()
    plt.plot(range(len(train_losses)), train_losses, label='Training Loss')
    plt.plot(range(len(test_losses)), test_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.show()


def plot_accuracy(train_accuracies, test_accuracies):
    plt.figure()
    plt.plot(range(len(train_accuracies)), train_accuracies, label='Training Accuracy')
    plt.plot(range(len(test_accuracies)), test_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('accuracy_plot.png')
    plt.show()


def plot_image(dataset, model, classes, device):
    idx = random.randint(0, len(dataset))
    label = dataset[idx][1]
    img = dataset[idx][0].unsqueeze(0).to(device)  # Move the input image tensor to the GPU
    model.eval()
    output = model(img)
    _, predicted = torch.max(output.data, 1)
    # Convert the image and show it
    img = img.squeeze().permute(1, 2, 0).cpu()  # Move the image tensor back to the CPU and adjust dimensions
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Predicted: {classes[predicted]}, True: {classes[label]}')
    plt.savefig('predicted_image.png')
    plt.show()
    print("Predicted label: ", classes[predicted[0].item()])
    print("Actual label: ", classes[label])


def plot_adv_images(testset, model, criterion, classes, device, epsilon_list, alpha=0.01, num_iters=10):
    model.eval()
    # Use first image from the testset for visualization
    dataiter = iter(testset)
    images, labels = dataiter.next()
    images, labels = images.to(device), labels.to(device)

    # If epsilon_list is not a list, convert it to a list with a single element
    if not isinstance(epsilon_list, list):
        epsilon_list = [epsilon_list]

    fig, axes = plt.subplots(len(epsilon_list) + 1, 1, figsize=(15, 10))

    # Plot original images
    img_grid = make_grid(images.cpu().data, normalize=True)
    axes[0].imshow(img_grid.permute(1, 2, 0))
    axes[0].set_title("Original Images")
    axes[0].axis('off')

    for idx, epsilon in enumerate(epsilon_list):
        adv_images = generate_adversarial_examples(images, labels, model, criterion, epsilon, alpha, num_iters, device)
        img_grid_adv = make_grid(adv_images.cpu().data, normalize=True)
        axes[idx + 1].imshow(img_grid_adv.permute(1, 2, 0))
        axes[idx + 1].set_title(f"Adversarial Images (Epsilon = {epsilon})")
        axes[idx + 1].axis('off')

    plt.tight_layout()
    plt.show()

def generate_adversarial_examples(images, labels, model, criterion, epsilon, alpha, num_iters, device):
    images_adv = images.clone().detach().requires_grad_(True).to(device)
    labels = labels.to(device)

    for i in range(num_iters):
        outputs = model(images_adv)
        model.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        perturbation = alpha * images_adv.grad.sign()
        images_adv = images_adv + perturbation
        perturbation = torch.clamp(images_adv - images, min=-epsilon, max=epsilon)
        images_adv = torch.clamp(images + perturbation, min=0, max=1).detach_()
        images_adv.requires_grad = True

    return images_adv


def epsilon_compare_multi(epsilon_values, adv_acc_pgd, att_succ_pgd, adv_acc_fgsm, att_succ_fgsm):
    if len(epsilon_values) != len(adv_acc_pgd) or len(epsilon_values) != len(att_succ_pgd) or len(epsilon_values) != len(adv_acc_fgsm) or len(epsilon_values) != len(att_succ_fgsm):
        print("Error: Input lists have different lengths.")
        return

    plt.figure(figsize=(10, 6))

    plt.plot(epsilon_values, adv_acc_pgd, 'o-', label='PGD Adversarial Accuracy')
    plt.plot(epsilon_values, att_succ_pgd, 'o-', label='PGD Attack Success Rate')
    plt.plot(epsilon_values, adv_acc_fgsm, 'o-', label='FGSM Adversarial Accuracy')
    plt.plot(epsilon_values, att_succ_fgsm, 'o-', label='FGSM Attack Success Rate')

    for i in range(len(epsilon_values)):
        plt.text(epsilon_values[i], adv_acc_pgd[i], f"{adv_acc_pgd[i]:.2f}", ha='center', va='bottom')
        plt.text(epsilon_values[i], att_succ_pgd[i], f"{att_succ_pgd[i]:.2f}", ha='center', va='bottom')
        plt.text(epsilon_values[i], adv_acc_fgsm[i], f"{adv_acc_fgsm[i]:.2f}", ha='center', va='bottom')
        plt.text(epsilon_values[i], att_succ_fgsm[i], f"{att_succ_fgsm[i]:.2f}", ha='center', va='bottom')

    plt.xlabel('Epsilon')
    plt.ylabel('Percentage')
    plt.title('Comparison of Adversarial Accuracies and Attack Success Rates between PGD and FGSM')
    plt.legend()
    plt.tight_layout()
    plt.savefig('epsilon_comparison.png')
    plt.show()



def main(train_model, epsilon_list, alpha=0.01, num_iters=10):
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Load the dataset
    num_classes = 10
    batch_size = 64
    trainset, trainloader, testset, testloader, classes = load_dataset(batch_size)

    # Load the pre-trained model
    model = models.resnet50(pretrained=True)
    # Modify conv1 to suit CIFAR-10
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # Modify the final fully connected layer according to the number of classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    # Move the model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Set hyperparameters
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    num_epochs = 60
    # Default single epsilon value if no list is provided
    epsilon_values = epsilon_list if epsilon_list else [0.3]  # Default to 0.3 if no epsilon_list is provided

    if train_model:
        print("Training the model...")
        # Train the model
        model, train_losses, train_accuracies, test_losses, test_accuracies = train_epochs(
            model, trainloader, testloader, criterion, optimizer, device, num_epochs)

        # Plot the loss and accuracy curves
        plot_loss(train_losses, test_losses)
        plot_accuracy(train_accuracies, test_accuracies)
        # Plot and save an example image
        plot_image(testset, model, classes, device)
        # Visualize some adversarial examples
        print("Generating Visualization Plot")
        plot_adv_images(testset, model, criterion, classes, device, epsilon_values[0], alpha, num_iters)
    else:
        # Load the best model
        best_model = models.resnet50(pretrained=True)
        # Modify conv1 to suit CIFAR-10
        best_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        best_model.fc = nn.Linear(num_features, num_classes)
        # Load checkpoints
        checkpoint = torch.load('/kaggle/input/best-model-cifar-10-resnet50/best_model.pth')
        best_model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        test_accuracy = checkpoint['test_accuracy']
        best_model = best_model.to(device)
        print("Best Trained Model Loaded!")
        print(f"Checkpoint at Epoch {epoch+1} with accuracy of {test_accuracy}%")

        # Test the best model on adversarial examples
        if epsilon_list:
            adversarial_accuracies_pgd = []
            attack_success_rates_pgd = []
            adversarial_accuracies_fgsm = []
            attack_success_rates_fgsm = []
            print("Testing with clean data again to compare with checkpoint accuracy...")
            _, clean_test_accuracy = test(best_model, testloader, criterion, device)
            print(f"Clean Test Accuracy: {clean_test_accuracy:.2f}%\n")
            for epsilon in epsilon_values:
                adv_acc_pgd, att_succ_pgd = test_adversarial(best_model, testloader, criterion, device, epsilon, alpha, num_iters, attack_type='PGD')
                adversarial_accuracies_pgd.append(adv_acc_pgd)
                attack_success_rates_pgd.append(att_succ_pgd)

                adv_acc_fgsm, att_succ_fgsm = test_adversarial(best_model, testloader, criterion, device, epsilon, alpha, num_iters, attack_type='FGSM')
                adversarial_accuracies_fgsm.append(adv_acc_fgsm)
                attack_success_rates_fgsm.append(att_succ_fgsm)

                # Visualize adversarial examples for the current epsilon
                print(f"Generating Visualization Plot for epsilon = {epsilon}")
                plot_adv_images(testset, best_model, criterion, classes, device, epsilon, alpha, num_iters)

            epsilon_compare_multi(epsilon_values, adversarial_accuracies_pgd, attack_success_rates_pgd, adversarial_accuracies_fgsm, attack_success_rates_fgsm)
        else:
            adv_acc_pgd, att_succ_pgd = test_adversarial(best_model, testloader, criterion, device, epsilon_values[0], alpha, num_iters, attack_type='PGD')
            adv_acc_fgsm, att_succ_fgsm = test_adversarial(best_model, testloader, criterion, device, epsilon_values[0], alpha, num_iters, attack_type='FGSM')
            print(f"PGD - Adversarial Accuracy: {adv_acc_pgd}, Attack Success Rate: {att_succ_pgd}")
            print(f"FGSM - Adversarial Accuracy: {adv_acc_fgsm}, Attack Success Rate: {att_succ_fgsm}")
            print("Generating Visualization Plot")
            plot_adv_images(testset, best_model, criterion, classes, device, epsilon_values[0], alpha, num_iters)

if __name__ == '__main__':
    main(train_model=True, epsilon_list=None)
    main(train_model=False, epsilon_list=[0.01, 0.03, 0.07, 0.1, 0.3, 0.5], alpha=0.5, num_iters=20)

