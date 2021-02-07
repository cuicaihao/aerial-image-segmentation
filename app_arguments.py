import argparse


def app_argparse():

    # Creating a parser
    parser = argparse.ArgumentParser(prog='AIL: Aerial Image Labelling',
                                     usage='%(prog)s [options]',
                                     description='Aerial Image Labelling with Deep Learning.',
                                     epilog="And that's how AI can help")

    # Adding arguments
    # positional arguments:
    parser.add_argument('--input_RGB',   type=str, default="images/case_03/RGB.png",
                        help='string of RGB image file path')

    parser.add_argument('--input_GT',   type=str, default="images/case_03/GT.png",
                        help='string of Ground Truce (GT image file path')

    parser.add_argument('--output_model_path', type=str, default="weights/CapeTown.model.weights.pt",
                        help='string of Ground Truce (GT image file path')

    parser.add_argument('--output_loss_plot',  type=str, default="output/loss_plot.png",
                        help='string of Ground Truce (GT image file path')

    parser.add_argument('--output_images',  type=str, default="output/",
                        help='string of output image file path')

    # optional arguments:
    parser.add_argument('--version', '-v', action='version',
                        version='%(prog)s 1.0.0')

    # flags
    parser.add_argument('--use_gpu', default=False,
                        help='use GPU(default: True)')

    parser.add_argument('--use_pretrain',  default=False,
                        help='use pretrained model')

    # hyper-parameters

    parser.add_argument('--tile_size', nargs=2, default=(250, 250), type=int,
                        help='input tile size ')

    parser.add_argument('--epochs',  default=1, type=int,
                        help='epoch number')

    parser.add_argument('--batch_size',  default=4, type=int,
                        help='batch size (?, channel, width, height)')

    parser.add_argument('--learning_rate',  default=1e-4, type=float,
                        help='model training learning rate')

    parser.add_argument('--weight_decay',  default=5e-3, type=float,
                        help='model weight decay / l2 regularization')
    return parser


def test_app_argparse():
    parser = app_argparse()
    args = parser.parse_args()
    assert args.epochs == 1
    assert type(args.epochs) == int
    assert args.batch_size == 4
    assert args.use_gpu == False
    assert args.use_pretrain == False
    print("PASS")


if __name__ == "__main__":
    parser = app_argparse()
    args = parser.parse_args()
    # print(args)
    # print(args.version)
    test_app_argparse()
    for arg in vars(args):
        print(arg, getattr(args, arg))
