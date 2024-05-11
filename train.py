import click
import torch
import torchvision
import torchvision.transforms as transforms
import wandb
from model import ViT, ViTConfig

def tokenize(images_BCHW, patch_size=16):
    B, C, _, _ = images_BCHW.shape
    # unfold the tensor along H and W dims into patch_size-size slices starting every patch_size indices
    patches = images_BCHW.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    # T (dim 2) is the number of patches == sequence length
    patches_BCTPP = patches.contiguous().view(B, C, -1, patch_size, patch_size)
    patches_BTCPP = patches_BCTPP.permute(0, 2, 1, 3, 4)
    T = patches_BTCPP.shape[1]
    patches_BTI = patches_BTCPP.contiguous().view(B, T, C * patch_size * patch_size)
    return patches_BTI

def compute_acc(net, patch_size, loader):
    net.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for x_BCHW, y_B in loader:
            x_BNI = tokenize(x_BCHW, patch_size)
            logits_BL, _ = net(x_BNI, y_B)
            _, predicted = torch.max(logits_BL, 1)
            total += y_B.numel()
            correct += (predicted == y_B).sum().item()
    net.train()
    return 100 * correct / total

@click.command()
@click.option("--lr", type=float, default=3e-4)
@click.option("--batch_size", type=int, default=128)
@click.option("--epochs", type=int, default=20)
@click.option("--use_wandb", type=bool, default=True)
@click.option("--log_freq", type=int, default=200)
def main(lr, batch_size, epochs, use_wandb, log_freq):
    if use_wandb:
        wandb.init(project="minViT")

    # image pre-processing
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # load data
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    config = ViTConfig()
    net = ViT(config)
    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=0.01)
    print(net)

    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, (x_BCHW, y_B) in enumerate(trainloader, 0):
            x_BNI = tokenize(x_BCHW, config.patch_size)

            optimizer.zero_grad()
            _, loss  = net(x_BNI, y_B)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if (i + 1) % log_freq == 0:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / log_freq:.3f}')
                if use_wandb:
                    wandb.log({"train/loss": running_loss / log_freq})
                running_loss = 0.0

        # compute train and test accs
        train_acc = compute_acc(net, config.patch_size, trainloader)
        test_acc = compute_acc(net, config.patch_size, testloader)
        print(f'train acc: {train_acc:.3f}, test acc: {test_acc:.3f}')
        if use_wandb:
            wandb.log({"train/acc": train_acc, "test/acc": test_acc})


if __name__ == "__main__":
    main()

