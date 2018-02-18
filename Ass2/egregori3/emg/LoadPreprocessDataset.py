from csv import reader

class LoadPreprocessDataset:

    # Load file
    def _load_csv(self, filename):
        dataset = list()
        with open(filename, 'r') as file:
            csv_reader = reader(file)
            for row in csv_reader:
                if not row:
                    continue
                dataset.append(row)
        return dataset


    # Convert string column to float
    def _str_column_to_float(self, dataset, column):
        for row in dataset:
            row[column] = float(row[column].strip())


    # Convert string column to integer
    def _str_column_to_int(self, dataset, column):
        class_values = [row[column] for row in dataset]
        unique = set(class_values)
        lookup = dict()
        for i, value in enumerate(unique):
            lookup[value] = i
        for row in dataset:
            row[column] = lookup[row[column]]
        return lookup


    # Find the min and max values for each column
    def _dataset_minmax(self, dataset):
        minmax = list()
        stats = [[min(column), max(column)] for column in zip(*dataset)]
        return stats


    # Rescale dataset columns to the range 0-1
    def _normalize_dataset(self, dataset, minmax):
        for row in dataset:
            for i in range(len(row)-1):
                row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


    def GetDataset(self):
        return self.dataset


    def __init__(self, filename):
        self.dataset = self._load_csv(filename)
        for i in range(len(self.dataset[0])-1):
            self._str_column_to_float(self.dataset, i)

        # convert class column to integers
        self._str_column_to_int(self.dataset, len(self.dataset[0])-1)

        # normalize input variables
        minmax = self._dataset_minmax(self.dataset)
        self._normalize_dataset(self.dataset, minmax)

