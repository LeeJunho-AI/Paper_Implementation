import torch
from alexnet import AlexNet
from train import train_model
from evaluate import evaluate_model
from dataset import get_dataset, get_transforms, get_dataloader

def main():
    # 디바이스 설정 (GPU 또는 CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 초기화
    model = AlexNet().to(device)

    # 데이터 전처리 및 로더 설정
    train_transforms = get_transforms(train=True)
    test_transforms = get_transforms(train=False)

    train_dataset = get_dataset(transform=train_transforms, train=True)
    test_dataset = get_dataset(transform=test_transforms, train=False)
    
    train_loader = get_dataloader(train_dataset, shuffle=True)
    test_loader = get_dataloader(test_dataset, shuffle=False)

    learning_rate = 0.001
    training_epochs = 1
    for mode in ["train", "evaluate"]:
        # 실행 모드에 따른 동작
        if mode == "train":
            print("학습을 시작합니다...")
            # 모델 학습
            train_model(model, train_loader, device, learning_rate=learning_rate, training_epochs=training_epochs)
    
        elif mode == "evaluate":
            print("평가를 시작합니다...")
    
            # 모델 평가
            evaluate_model(model, test_loader, device)

if __name__ == "__main__":
    main()

