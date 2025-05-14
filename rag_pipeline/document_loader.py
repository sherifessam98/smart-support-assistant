import os
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter   #Tool to split text into chunks


def load_text_file(filepath: str) -> str:
    """
    loads a plain text file from the given file path and return its content as a string.

    """

    with open (filepath,"r", encoding="utf-8") as f:
        return f.read()

def chunk_text(text:str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Splits text into overlapping chunks using Langchain's text splitter.
    :param text: the full text to split
    :param chunk_size: number of characters per chunk
    :param overlap: how many character to repeat per chunks
    :return: A list of chunks (strings)
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = overlap,
        length_function = len
    )
    return splitter.split_text(text) #split the text and return chunks as a list
