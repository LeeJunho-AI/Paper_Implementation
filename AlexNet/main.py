import torch
from alexnet import AlexNet
from train import train_model
from evaluate import evaluate_model
from dataset import get_dataset, get_transforms, get_dataloader

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AlexNet().to(device)

    train_transforms = get_transforms(train=True)
    test_transforms = get_transforms(train=False)

    train_dataset = get_dataset(transform=train_transforms, train=True)
    test_dataset = get_dataset(transform=test_transforms, train=False)
    
    train_loader = get_dataloader(train_dataset, shuffle=True)
    test_loader = get_dataloader(test_dataset, shuffle=False)

    learning_rate = 0.001
    training_epochs = 1
    for mode in ["train", "evaluate"]:
        if mode == "train":
            print("Train Started...")
            train_model(model, train_loader, device, learning_rate=learning_rate, training_epochs=training_epochs)
    
        elif mode == "evaluate":
            print("Evaluation Started...")
    
            # 모델 평가
            evaluate_model(model, test_loader, device)

if __name__ == "__main__":
    main()

