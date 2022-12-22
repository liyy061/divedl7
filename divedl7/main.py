from MyDataloader import load_data_fashion_mnist, try_gpu
from MyModel1 import LeNet_5
from MyModel2 import LeNet_9
from MyModel3 import LeNetChan_30
from MyTrain import train_ch6
from MyPlotlib import use_svg_display
from matplotlib import pyplot as plt

def main():
    use_svg_display()
    batch_size = 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size)
    net = LeNet_5().net
    # net = LeNet_9().net #Model2
    # net = LeNetChan_30() #Model3
    lr, num_epochs = 0.9, 10
    train_ch6(net, train_iter, test_iter, num_epochs, lr, try_gpu())
    plt.show()

main()