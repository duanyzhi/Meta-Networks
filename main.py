from network.net import meta_networks as mn
import argparse

def run(pattern, data):
    net = mn(pattern, data)
    net.build_network()
    # if pattern == "train":
    #     net.bp()
    net.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--pattern', type=str, default='test',  # pattern前面加两个--表示pattern是可选参数
                        required=True, help='Choice train or test model')
    parser.add_argument('--data', type=str, default='omn',
                        required=False, help='Choice which dataset')

    args = parser.parse_args()

    # assert args.data == 'omn' or args.data == 'oxford' or args.data == 'campus'

    run(args.pattern, args.data) # train or test

# python main.py --pattern train --data omn
# python main.py --pattern test --data omn
