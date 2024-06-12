import zipfile
import os
from bs4 import BeautifulSoup
import markdown2

class MarkdownExtractor:
    @staticmethod
    def markdown_to_text(md_content):
        """Convert Markdown content to plain text."""
        html_content = markdown2.markdown(md_content)
        soup = BeautifulSoup(html_content, 'html.parser')
        return soup.get_text()

    @classmethod
    def extract_and_convert(cls, zip_path, extract_path, output_file_path):
        """Extract Markdown files from a zip and convert to plain text."""
        # Extract all files from the zip
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

        # Create the combined plain text file
        with open(output_file_path, 'w') as output_file:
            # Traverse all extracted files
            for root, _, files in os.walk(extract_path):
                for file in files:
                    if file.endswith('.md'):
                        # Open and read each Markdown file
                        file_path = os.path.join(root, file)
                        with open(file_path, 'r') as md_file:
                            md_content = md_file.read()
                        # Convert Markdown content to text and write to output
                        output_file.write(cls.markdown_to_text(md_content) + '\n')
