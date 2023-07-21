import traceback
import argparse
import time
import os
import cv2

import numpy as np


def get_parser():
    parser = argparse.ArgumentParser(description='Detection with YOLO models')
    parser.add_argument("-ver", "--version", required=True, type=str, help="The version of YOLO used")
    parser.add_argument("-mp", "--model_path", required=True, type=str, help="The path to pretrained YOLO model")
    parser.add_argument("-s", "--source", required=True, type=str, help="The source path of the data for detection")
    parser.add_argument("-lm", "--load_mode", required=True, type=str,
                        help="The loading mode. Available modes: [dataset, video]")
    parser.add_argument("-t", "--target", required=False, type=str, default=".\\results",
                        help="The path where data are stored during the detection")
    parser.add_argument("-dev", "--device", required=False, type=str, default="cuda",
                        help="The device used during the training (cpu or cuda)")
    parser.add_argument("-b", "--batch_size", required=False, type=int, default=32,
                        help="The batch size for the YOLO model")
    parser.add_argument("-iw", "--width", required=False, type=int, default=160, help="The input image width")
    parser.add_argument("-ih", "--height", required=False, type=int, default=160, help="The input image height")
    parser.add_argument("-sf", "--split_file", required=False, type=str, default="evaluate.txt",
                        help="The name of the file containing sample splits (applicable only for the load_mode set to dataset)")

    return parser.parse_args()


def process_single_sample(image: np.ndarray, detection: np.ndarray, shape: tuple[int, int, int], target: str,
                          index: int, classes=43, cell_count=5, threshold=0.5):
    image = (np.transpose(image, (1, 2, 0)) * 255).astype('uint8')
    for row in range(len(detection)):
        for column in range(len(detection[row])):
            cell = detection[row, column]
            for box_index in range(classes, len(cell), 5):
                box = cell[box_index:box_index + 5]
                if box[0] > threshold:
                    class_index = int(np.argmax(cell[:classes]))
                    xcell, ycell, widthcell, heightcell = box[1], box[2], box[3], box[4]

                    x = (xcell + column) / cell_count
                    y = (ycell + row) / cell_count
                    width, height = widthcell / cell_count, heightcell / cell_count

                    x, y, width, height = x * shape[1], y * shape[0], width * shape[1], height * shape[0]

                    image = cv2.rectangle(image, (int(x - width / 2), int(y - height / 2)),
                                          (int(x + width / 2), int(y + height / 2)), (255, 0, 0), -1)
                    image = cv2.putText(image, str(class_index), (int(x - width / 2), int(y - height / 2 - 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, (255, 0, 0), -1, cv2.LINE_AA)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(target, f'{index}.png'), image)


def dataset_mode(model, data, target: str, shape: tuple[int, int, int]):
    batches, detections = model.predict(data)
    index = 0
    for batch, batch_detection in zip(batches, detections):
        for image, detection in zip(batch, batch_detection):
            process_single_sample(image, detection, shape, target, index)
            index += 1


def video_mode(model, source: str, target: str, batch_size: int):
    # todo: write video mode
    raise NotImplementedError("Video mode is not yet implemented")


def run(parser):
    # verify arguments
    assert parser.version in ['v1'], f'Version {parser.version} is not currently supported'
    assert parser.model_path != "", 'The path to pretrained model cannot be empty'
    assert parser.source != "", 'The source path of the data cannot be empty'
    assert parser.load_mode in ['dataset', 'video'], "The data can only be loaded from the dataset or video"
    assert parser.target != "", 'Target path cannot be empty'
    assert parser.device in ['cpu',
                             'cuda'], "Device can only be set to gpu/cuda:0 (no parallelism supported) or cpu"
    assert parser.batch_size > 0, "Batch size cannot be negative"
    assert parser.width > 0, "Width cannot be negative"
    assert parser.height > 0, "Height cannot be negative"
    assert parser.split_file != "", "The split filename cannot be empty"

    # import proper model version
    if parser.version == "v1":
        import detection.models.yolo_v1.provider as p
    else:
        raise NotImplementedError(f"Model {parser.version} is not implemented")

    # check if model, source, and target exist
    if not os.path.exists(parser.model_path):
        raise FileNotFoundError(f"Cannot find the following file: '{parser.model_path}'")
    if not os.path.exists(parser.source):
        raise FileNotFoundError(f"Cannot find the following file/dir: '{parser.source}'")
    if not os.path.exists(parser.target):
        raise FileNotFoundError(f"Cannot find the following dir: '{parser.target}'")

    # create unique dir for the result
    target_path = os.path.join(parser.target, f'yolo_{parser.version}')
    if not os.path.exists(target_path):
        os.mkdir(target_path)

    stamp = int(time.time())
    target_path = os.path.join(target_path, str(stamp))
    if not os.path.exists(target_path):
        os.mkdir(target_path)

    # get provider and instantiate objects needed for detection
    provider = p.YoloProvider()
    model = provider.get_model(
        device=parser.device,
        target_path=target_path,
        input_shape=(parser.height, parser.width, 3)
    )
    model.load(parser.model_path)

    if parser.load_mode == "dataset":
        data, _, _ = provider.get_data(
            data_path=parser.source,
            train_file=parser.split_file,
            test_file=None,
            valid_file=None,
            batch_size=parser.batch_size,
            input_shape=(parser.height, parser.width, 3)
        )
        dataset_mode(
            model=model,
            data=data,
            target=target_path,
            shape=(parser.height, parser.width, 3)
        )
    elif parser.load_mode == "video":
        video_mode(
            model=model,
            source=parser.source,
            target=target_path,
            batch_size=parser.batch_size
        )


if __name__ == "__main__":
    try:
        run(get_parser())
    except Exception as e:
        print(f'The following exception occurred: {e}. Stack trace:')
        traceback.print_exc()
