import argparse
import os


def parse_option():
    parser = argparse.ArgumentParser('Visual Prompting for CLIP')

    parser.add_argument('--name', type=str, default='exp',
                        help='Description of the experiment')
    parser.add_argument("--device", type = str, default = 'cuda', help = "Specify device type to use (default: gpu > cpu)")
    parser.add_argument("--device_id", type = int, default = 0, help = "Specify device id if using single gpu")

    parser.add_argument('--print_freq', type=int, default=50,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=3,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epoch5s')

    # optimization
    parser.add_argument('--optim', type=str, default='sgd',
                        help='optimizer to use')
    parser.add_argument('--learning_rate', type=float, default=40,
                        help='learning rate')
    parser.add_argument("--weight_decay", type=float, default=0,
                        help="weight decay")
    parser.add_argument("--warmup", type=int, default=1000,
                        help="number of steps to warmup for")
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--patience', type=int, default=10000)

    # model
    parser.add_argument('--model', type=str, default='clip')
    parser.add_argument('--arch', type=str, default='ViT-B/32', choices=['vit_b32','ViT-B/32','ViT-B/16',"RN50", "RN101", "RN50x4"],)
    parser.add_argument('--pretrained', default=True, help =' whether to use pretrained model')
    parser.add_argument('--method', type=str, default='padding',
                        choices=['padding', 'random_patch', 'fixed_patch', 'badnoise', 'badfeature', 'malpoisnum'],
                        help='choose visual prompting method')
    parser.add_argument('--prompt_size', type=int, default=30,
                        help='size for visual prompts')



    # other
    parser.add_argument('--seed', type=int, default=0,
                        help='seed for initializing training')
    parser.add_argument('--checkpoints', type=str, default='./results',
                        help='path to save models')

    parser.add_argument('--filename', type=str, default=None,
                        help='filename to save')
    parser.add_argument('--trial', type=int, default=1,
                        help='number of trials')
    parser.add_argument('--resume', type=str, default=None,
                        help='path to resume from checkpoint')
    parser.add_argument('--evaluate', default=False,
                        action="store_true",
                        help='evaluate model test set')
    parser.add_argument('--ct', default=False,
                        action="store_true",
                        help='evaluate model test set')

    parser.add_argument('--use_wandb', default=False,
                        action="store_true",
                        help='whether to use wandb')

    # fine-tuning
    # parser.add_argument('--linear_probe_batch_size', type=str, default='128', help='dataset')


    # dataset
    # parser.add_argument('--root', type=str, default='./data',
    #                     help='dataset')
    parser.add_argument('--dataset', '-data', type=str, default='cifar10',
                        help='dataset')
    parser.add_argument('--dataset_eval', '-val', type=str, default='cifar10',
                        help='dataset')
    parser.add_argument('--image_size', type=int, default=224,
                        help='image size')

    parser.add_argument('--train_data_dir', type=str, default=None,
                        help='the train datasets')
    parser.add_argument('--eval_data_dir', type=str, default=None,
                        help='the test datasets')

    # backdoor
    # parser.add_argument('--backdoor', type=str, default=False,
    #                     help='is backdoor')
    parser.add_argument('--backdoor', action='store_true',help='is backdoor')
    parser.add_argument('--mode', type=str, default="all2one",
                        help='backdoor type')
    parser.add_argument('--ori_label', type=int, default=0,
                        help='backdoor ori label')
    parser.add_argument('--target', type=int, default=0,
                        help='backdoor target')
    parser.add_argument('--pois_ratio', type=float, default=0.1,
                        help='poison ratio')
    parser.add_argument('--eps', type=str, default="4/255",
                        help='backdoor noise')
    parser.add_argument('--attack', type=str, default="vpa",
                        help='backdoor noise')

    parser.add_argument('--detector_lr', type=float, default=0.01,
                        help='detector lr ')
    parser.add_argument('--detector_len', type=float, default=0.1,
                        help='detector lr ')

    parser.add_argument('--resume_t', type=str, default=None,
                        help='trigger resume')

    parser.add_argument('--resume_c', type=str, default=None,
                        help='trigger resume')
    parser.add_argument('--resume_detect', type=str, default=None,
                        help='trigger resume')

    parser.add_argument('--lambda1', type=float, default=0.01,
                        help='hyper-params-mse')
    parser.add_argument('--lambda2', type=float, default=0.01,
                        help='hyper-params-norm')
    parser.add_argument('--lambda3', type=float, default=0.001,
                        help='hyper-params-ce')
    parser.add_argument('--lambda4', type=float, default=0.001,
                        help='hyper-params-adv')
    parser.add_argument('--lambda5', type=float, default=0.001,
                        help='hyper-params-adv')

    parser.add_argument('--clip_model_name', type=str, default=None,
                        help='trigger resume')


    # parser.add_argument('--', type=str, default=None,
    #                     help='')
    args = parser.parse_args()



    data_map = {
        'CIFAR10': 'CIFAR10','cifar10': 'CIFAR10',

        'CIFAR100': 'CIFAR100', 'cifar100': 'CIFAR100',

        'Flowers102': 'Flowers102',
        'Food101':'Food101',
        'GTSRB': 'GTSRB',
        'FGVCAircraft': 'FGVCAircraft',
        'StanfordCars': 'stanford-cars',
        'RenderedSST2': 'RenderedSST2',
        'STL10':'STL10',
        'EuroSAT': 'EuroSAT/EuroSAT_RGB',
        'SVHN': 'SVHN',
        'OxfordIIITPet': 'OxfordIIITPet',
        'SUN397': 'SUN397',
        'UCF101': 'UCF101',
        'DTD': 'DTD',
        'RESISC': 'RESISC',
        'CLEVR': 'CLEVR',
        'Calech101': 'Calech101',
        'OxfordPets': 'OxfordPets',
        'ImageNetV2': 'ImageNetV2',
        'ImageNet-Sketch': 'ImageNet-Sketch',
        'ImageNet-A': 'ImageNet-A',
        'ImageNet-R': 'ImageNet-R',
    }


    # 获取根目录
    home_dir = os.path.expanduser("~")
    args.root = home_dir
    print(args.root)
    # 数据目录
    args.dataset_eval = args.dataset

    args.train_data_dir = os.path.join(args.root, 'datasets', data_map[args.dataset_eval])
    args.eval_data_dir = os.path.join(args.root, 'datasets', data_map[args.dataset_eval])


    # args.filename = '{}_{}_{}_{}_{}_{}_lr_{}_decay_{}_bsz_{}_warmup_{}_trial_{}'. \
    #     format(args.method, args.prompt_size, args.dataset, args.model, args.arch,
    #            args.optim, args.learning_rate, args.weight_decay, args.batch_size, args.warmup, args.trial)

    print(args.__dict__)
    return args

