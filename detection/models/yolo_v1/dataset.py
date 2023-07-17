class YoloDataset:
    def __init__(self, path: str, split_filename: str, **kwargs):
        super(YoloDataset, self).__init__(**kwargs)

        self.path = path
        self.split_filename = split_filename
