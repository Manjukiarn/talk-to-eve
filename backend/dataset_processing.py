import pandas as pd


# Load and process the Twitter conversation dataset
def load_data(file_path):
    # Use a different encoding if utf-8 fails
    data = pd.read_csv(file_path, encoding='ISO-8859-1')  # Change the encoding here
    return data

# Function to process the dataset
def process_data(data):
    # Create a copy of the DataFrame slice to avoid the SettingWithCopyWarning
    processed_data = data[['conversation_id', 'message', 'sentiment']].copy()
    processed_data.dropna(inplace=True)  # Now this won't raise a warning
    return processed_data


if __name__ == "__main__":
    # Load and process the dataset
    dataset = load_data(r'F:\talk-to-eve\backend\test.csv')  # Use the full file path
    processed_data = process_data(dataset)

    print(processed_data.head())  # Display the first few rows of the processed dataset

