from utils import read_data, load_bw
from train import train

def start():
    print("Calling train function..")
    return train(cv=False)


if __name__ == "__main__":
    start()

