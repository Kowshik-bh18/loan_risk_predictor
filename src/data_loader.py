import pandas as pd

class DataLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        """Load CSV data into pandas DataFrame"""
        try:
            self.data = pd.read_csv(self.file_path)
            print("Data loaded successfully.")
            return self.data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def show_basic_info(self):
        """Display basic dataset information"""
        if self.data is not None:
            print("\nDataset Info:")
            print(self.data.info())
            print("\nFirst 5 Rows:")
            print(self.data.head())
        else:
            print("Data not loaded yet.")