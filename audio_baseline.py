if __name__ == '__main__':
    # -*- coding: utf-8 -*-
    import numpy as np
    import os
    import argparse
    import time
    import torch
    import torch.nn as nn
    import torch.backends.cudnn as cudnn
    import torchaudio
    import torchaudio.transforms as T
    import torch.nn as nn
    from torchvision.transforms import Resize
    import torchvision.transforms as trn
    import torchvision.datasets as dset
    import torch.nn.functional as F
    from tqdm import tqdm
#    from models.allconv import AllConvNet
    from models.wrn import WideResNet
#    from models.melspecnet import MelSpecNet
    from torch.utils.data import Dataset, DataLoader, Subset, random_split
    from pathlib import Path
#    import matplotlib.pyplot as plt
    from urbansounddataset import UrbanSoundDataset
    from sklearn.model_selection import train_test_split

    # go through rigamaroo to do ...utils.display_results import show_performance
#    if __package__ is None:
#        import sys
#        from os import path
#        sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
#        from utils.validation_dataset import validation_split

    parser = argparse.ArgumentParser(description='Trains a CIFAR Classifier',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataset', type=str, choices=['cifar10', 'cifar100', 'custom'],
                        help="Specify 'custom'")
    parser.add_argument('--model', '-m', type=str, default='allconv',
                        choices=['allconv', 'wrn', 'pretrained_resnet'], help='Choose architecture.')
    parser.add_argument('--calibration', '-c', action='store_true',
                        help='Train a model to be used for calibration. This holds out some data for validation.')
    # Optimization options
    parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='The initial learning rate.')
    parser.add_argument('--batch_size', '-b', type=int, default=64, help='Batch size.')
    parser.add_argument('--test_bs', type=int, default=200)
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
    # WRN Architecture
    parser.add_argument('--layers', default=40, type=int, help='total number of layers')
    parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
    parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
    # Checkpoints
    parser.add_argument('--save', '-s', type=str, default='./snapshots/baseline', help='Folder to save checkpoints.')
    parser.add_argument('--load', '-l', type=str, default='', help='Checkpoint path to resume / test.')
    parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
    # Acceleration
    parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
    parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')
    args = parser.parse_args()

    state = {k: v for k, v in args._get_kwargs()}
    print(state)

    torch.manual_seed(1)
    np.random.seed(1)

    # Data, Config
    data_path = Path('/home/wim17006/UrbanSoundOOD/UrbanSound8K/audio')
    annotations_path = Path('/home/wim17006/UrbanSoundOOD/UrbanSound8K/metadata/UrbanSound8K.csv')
    sample_rate = 22050
    num_samples = 88200
    seed = 43

    if args.ngpu > 0:
        device = torch.device('cuda')
    if args.ngpu == 0:
        device = 'cpu'

    label_map = {
        'air_conditioner': 0, 'car_horn': 1, 'children_playing': 2, 'dog_bark': 3,
        'drilling': 4, 'engine_idling': 5, 'gun_shot': 6, 'jackhammer': 7, 'siren': 8,
        'street_music': 9
    }


    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    usd = UrbanSoundDataset(annotations_path, data_path, mel_spectrogram, sample_rate,
                            num_samples, device=device)

    # split data
    # https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets
    train_size = int(len(usd) * 0.8)
    test_size = len(usd) - train_size
    train_split, test_split = random_split(usd, [train_size, test_size])

    train_loader = DataLoader(train_split, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_split, batch_size=args.batch_size)

    calib_indicator = ''
    if args.calibration:
        train_data, val_data = validation_split(train_split, val_share=0.1)
        calib_indicator = '_calib'

    # View some images from the loaders
    # images, labels = next(iter(train_loader))
    # fig = plt.figure(figsize=(25,4))
    #
    # for idx in np.arange(10):
    #     ax = fig.add_subplot(2, 5, idx+1, xticks=[], yticks=[])
    #     print(images[idx][0].shape)
    #     plt.imshow(images[idx][0])
    #     ax.set_xlabel(labels[idx])
    # plt.show()

    #Create model
    num_classes = 10
    if args.model == 'allconv':
        net = AllConvNet(num_classes)
    elif args.model == 'melspecnet':
        net = MelSpecNet(num_classes)
    elif args.model == 'pretrained_resnet':
        net = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    else:
        net = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate)


    start_epoch = 0
    # Restore model if desired
    if args.load != '':
        for i in range(1000 - 1, -1, -1):
            model_name = os.path.join(args.load, args.dataset + calib_indicator + '_' + args.model +
                                      '_baseline_epoch_' + str(i) + '.pt')
            if os.path.isfile(model_name):
                net.load_state_dict(torch.load(model_name))
                print('Model restored! Epoch:', i)
                start_epoch = i + 1
                break
        if start_epoch == 0:
            assert False, "could not resume"

    # if args.ngpu > 1:
    #     net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

    # if args.ngpu > 0:
    #     net.cuda()
    #     torch.cuda.manual_seed(1)

    net.to(device)
    print(net)

    # cudnn.benchmark = True  # fire on all cylinders

    optimizer = torch.optim.SGD(
        net.parameters(), state['learning_rate'], momentum=state['momentum'],
        weight_decay=state['decay'], nesterov=True)


    def cosine_annealing(step, total_steps, lr_max, lr_min):
        return lr_min + (lr_max - lr_min) * 0.5 * (
                1 + np.cos(step / total_steps * np.pi))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(
            step,
            args.epochs * len(train_loader),
            1,  # since lr_lambda computes multiplicative factor
            1e-6 / args.learning_rate))


    # /////////////// Training ///////////////

    def train():
        net.train()  # enter train mode
        loss_avg = 0.0
        for step, (data, target) in enumerate(train_loader):
            # print(f'step: {step}')
            data, target = data.to(device), target.to(device)

            # forward
            x = net(data)

            # backward
            optimizer.zero_grad()
            loss = F.cross_entropy(x, target)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # exponential moving average
            loss_avg = loss_avg * 0.8 + float(loss) * 0.2

        state['train_loss'] = loss_avg


    # test function
    def test():
        net.eval()
        loss_avg = 0.0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)

                # forward
                output = net(data)
                loss = F.cross_entropy(output, target)

                # accuracy
                pred = output.data.max(1)[1]
                correct += pred.eq(target.data).sum().item()

                # test loss average
                loss_avg += float(loss.data)

        state['test_loss'] = loss_avg / len(test_loader)
        state['test_accuracy'] = correct / len(test_loader.dataset)


    if args.test:
        test()
        print(state)
        exit()

    # Make save directory
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    if not os.path.isdir(args.save):
        raise Exception('%s is not a dir' % args.save)

    with open(os.path.join(args.save, args.dataset + calib_indicator + '_' + args.model +
                                      '_baseline_training_results.csv'), 'w') as f:
        f.write('epoch,time(s),train_loss,test_loss,test_error(%)\n')

    print('Beginning Training\n')

    # Main loop
    for epoch in range(start_epoch, args.epochs):
        # print(f'epoch: {epoch}')

        state['epoch'] = epoch
        begin_epoch = time.time()

        train()
        test()

        # Save model
        torch.save(net.state_dict(),
                   os.path.join(args.save, args.dataset + calib_indicator + '_' + args.model +
                                '_baseline_epoch_' + str(epoch) + '.pt'))
        # Let us not waste space and delete the previous model
        prev_path = os.path.join(args.save, args.dataset + calib_indicator + '_' + args.model +
                                 '_baseline_epoch_' + str(epoch - 1) + '.pt')
        if os.path.exists(prev_path): os.remove(prev_path)

        # Show results
        with open(os.path.join(args.save, args.dataset + calib_indicator + '_' + args.model +
                                          '_baseline_training_results.csv'), 'a') as f:
            f.write('%03d,%05d,%0.6f,%0.5f,%0.2f\n' % (
                (epoch + 1),
                time.time() - begin_epoch,
                state['train_loss'],
                state['test_loss'],
                100 - 100. * state['test_accuracy'],
            ))

        # # print state with rounded decimals
        # print({k: round(v, 4) if isinstance(v, float) else v for k, v in state.items()})

        print('Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | Test Loss {3:.3f} | Test Error {4:.2f}'.format(
            (epoch + 1),
            int(time.time() - begin_epoch),
            state['train_loss'],
            state['test_loss'],
            100 - 100. * state['test_accuracy'])
        )
