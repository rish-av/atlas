


#class to load data
class DataLoader():

    def __init__(self, path):
        self.path = path

    def load(self):
        with open(self.path, 'r') as f:
            return f.read()

    def __len__(self):
        with open(self.path, 'r') as f:
            return len(f.readlines())


    # return item should contain the data for level of agent    
    def __getitem__(self, idx):
        with open(self.path, 'r') as f:
            return f.readlines()[idx]