import click
import torch
from torch.cuda.amp import GradScaler, autocast
import torchvision
import torchvision.transforms as transforms
import wandb
from model import ViT, ViTConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            x_BNI = tokenize(x_BCHW, patch_size).to(device, non_blocking=True)
            y_B = y_B.to(device, non_blocking=True)
            logits_BL, _ = net(x_BNI, y_B)
            _, predicted = torch.max(logits_BL, 1)
            total += y_B.numel()
            correct += (predicted == y_B).sum().item()
    net.train()
    return 100 * correct / total

@click.command()
@click.option("--lr", type=float, default=2e-3)
@click.option("--batch_size", type=int, default=1024)
@click.option("--epochs", type=int, default=20)
@click.option("--use_wandb", type=bool, default=True)
@click.option("--log_freq", type=int, default=25)
@click.option("--eval_freq", type=int, default=4)
@click.option("--n_head", type=int, default=12)
@click.option("--model_dim", type=int, default=192)
@click.option("--dropout", type=float, default=0.1)
@click.option("--weight_decay", type=float, default=0.01)
@click.option("--patch_size", type=int, default=16)
@click.option("--compile", type=bool, default=True)
@click.option("--run_name", type=str, default="minViT")
def main(lr, batch_size, epochs, use_wandb, log_freq, eval_freq, n_head, model_dim, dropout, weight_decay, patch_size, compile, run_name):
    config = {arg: value for arg, value in locals().items() if arg != 'args'}
    # image pre-processing
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # load data
    num_workers = 4
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=num_workers, pin_memory=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=num_workers, pin_memory=True)

    vit_config = ViTConfig(n_head=n_head, model_dim=model_dim, dropout=dropout, patch_size=patch_size)
    net = ViT(vit_config).to(device)
    if compile:
        net = torch.compile(net)
    scaler = GradScaler()
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    dtype = torch.float32
    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=weight_decay)
    
    if use_wandb:
        wandb.init(project="minViT", name=run_name, config=config)
    print(net)
    print("CUDA:", torch.cuda.is_available())
    print("bfloat16:", dtype == torch.bfloat16)

    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, (x_BCHW, y_B) in enumerate(trainloader, 0):
            x_BNI = tokenize(x_BCHW,  vit_config.patch_size).to(device, non_blocking=True)
            y_B = y_B.to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast(dtype=dtype):
                _, loss  = net(x_BNI, y_B)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # print statistics
            running_loss += loss.item()
            if (i + 1) % log_freq == 0:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / log_freq:.3f}')
                if use_wandb:
                    wandb.log({"train/loss": running_loss / log_freq})
                running_loss = 0.0

        # compute train and test accs
        if epoch % eval_freq == 0:
            train_acc = compute_acc(net, vit_config.patch_size, trainloader)
            test_acc = compute_acc(net, vit_config.patch_size, testloader)
            print(f'train acc: {train_acc:.3f}, test acc: {test_acc:.3f}')
            if use_wandb:
                wandb.log({"train/acc": train_acc, "test/acc": test_acc})


if __name__ == "__main__":
    main()

