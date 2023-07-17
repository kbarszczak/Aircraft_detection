import traceback
import argparse
import time
import os


def get_parser():
    parser = argparse.ArgumentParser(description='Training FBNet model')
    parser.add_argument("-ver", "--version", required=True, type=str, help="The version of YOLO used")
    parser.add_argument("-t", "--target", required=False, type=str, default="training",
                        help="The path where data is stored during the training (such as history etc.)")
    parser.add_argument("-d", "--data", required=False, type=str,
                        default="D:\\Data\\Aircraft_Detection\\dataset",
                        help="The source path of the dataset")
    parser.add_argument("-dt", "--train", required=False, type=str, default="train.txt",
                        help="The name of file that contains training samples split")
    parser.add_argument("-de", "--evaluate", required=False, type=str, default="evaluate.txt",
                        help="The name of file that contains testing samples split")
    parser.add_argument("-dv", "--valid", required=False, type=str, default="valid.txt",
                        help="The name of file that contains validating samples split")
    parser.add_argument("-dev", "--device", required=False, type=str, default="gpu",
                        help="The device used during the training (cpu or gpu)")
    parser.add_argument("-b", "--batch_size", required=False, type=int, default=2, help="The batch size")
    parser.add_argument("-e", "--epochs", required=False, type=int, default=10, help="The epochs count")
    parser.add_argument("-iw", "--width", required=False, type=int, default=256, help="The input image width")
    parser.add_argument("-ih", "--height", required=False, type=int, default=144, help="The input image height")

    return parser.parse_args()


def run(parser):
    # verify arguments
    assert parser.version in ['v1'], f'Version {parser.version} is not currently supported'
    assert parser.target is not "", 'Target path cannot be empty'
    assert parser.data is not "", 'Data path cannot be empty'
    assert parser.train is not "", 'Train split filename path cannot be empty'
    assert parser.evaluate is not "", 'Evaluate split filename path cannot be empty'
    assert parser.valid is not "", 'Valid split filename path cannot be empty'
    assert parser.device in ['gpu', 'cpu',
                             'cuda:0'], "Device can only be set to gpu/cuda:0 (no parallelism supported) or cpu"
    assert parser.name, "Name cannot be empty"
    assert parser.batch_size > 0, "Batch size cannot be negative"
    assert parser.epochs > 0, "Epochs cannot be negative"
    assert parser.width > 0, "Width cannot be negative"
    assert parser.height > 0, "Height cannot be negative"

    # import proper model version
    if parser.version == "v1":
        import detection.providers.yolo_v1.yolo_provider as yolo_provider
    else:
        raise NotImplementedError(f"Model {parser.version} is not implemented")

    # check if dataset exists
    if not os.path.exists(parser.data):
        raise FileNotFoundError(f"Cannot find the following dir: '{parser.data}'")
    if not os.path.exists(os.path.join(parser.data, parser.train)):
        raise FileNotFoundError(f"Cannot find the following file: '{os.path.join(parser.data, parser.train)}'")
    if not os.path.exists(os.path.join(parser.data, parser.evaluate)):
        raise FileNotFoundError(f"Cannot find the following file: '{os.path.join(parser.data, parser.evaluate)}'")
    if not os.path.exists(os.path.join(parser.data, parser.valid)):
        raise FileNotFoundError(f"Cannot find the following file: '{os.path.join(parser.data, parser.valid)}'")

    # check if destination dir exists
    if not os.path.exists(parser.target):
        raise FileNotFoundError(f"Cannot find the following dir: '{parser.target}'")

    target_path = os.path.join(parser.target, f'yolo_{parser.version}')
    if not os.path.exists(target_path):
        os.mkdir(target_path)

    stamp = int(time.time())
    target_path = os.path.join(target_path, str(stamp))
    if not os.path.exists(target_path):
        os.mkdir(target_path)

    # get provider and instantiate objects needed for training
    provider = yolo_provider.YoloProvider()
    model = provider.get_model(
        device=parser.device,
        target_path=target_path
    )
    train_data, test_data, valid_data = provider.get_data(
        data_path=parser.data,
        train_file=parser.train,
        test_file=parser.evaluate,
        valid_file=parser.valid,
        batch_size=parser.batch_size,
        width=parser.width,
        height=parser.height
    )

    # launch training
    model.fit(train_data, valid_data, parser.epochs)

    # launch evaluation
    model.evaluate(test_data)


if __name__ == "__main__":
    try:
        run(get_parser())
    except Exception as e:
        print(f'The following exception occurred: {e}. Stack trace:')
        traceback.print_exc()
