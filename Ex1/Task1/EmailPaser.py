from pathlib import Path
from datetime import datetime

# Function to extract the date from the given mail file
def get_date(mail_file)  -> datetime.date:
    filename = Path(mail_file).name
    date_part = filename.split('.')[1] # Extract the date part from the filename
    date = datetime.strptime(date_part, "%Y-%m-%d").date() # Convert the string to a date object
    return date

# Function to extract the subject from the given mail file
def get_subject(mail_file) -> str:
    with open(mail_file, 'r', encoding='utf-8') as file:
        subject = file.readline()[9:] # Read the first line and remove the first 9 characters
        return subject

# Function to extract the body text from the given mail file
def get_bodytext(mail_file) -> str:
    with open(mail_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        body_text = ''.join(lines[1:]) # Join all lines except the first one
        return body_text

# Function to check if the mail is spam
def is_spam(mail_file) -> bool:
    filename = Path(mail_file).name
    return filename.split('.')[-2] == 'spam'


# Example usage / test
folder = Path("/Datasets/email/test/spam")

for file in folder.iterdir():
    if file.is_file():
        print("="*50)
        print(f"File: {file.name}")
        print("Date: ", get_date(file))
        print("Subject: ", get_subject(file))
        # print("Body Text: ", get_bodytext(file))
        print("Is Spam: ", is_spam(file))