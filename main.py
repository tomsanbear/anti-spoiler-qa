import json
import os
import sqlite3

import cohere
import ebooklib
import inquirer
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from ebooklib import epub
from tqdm import tqdm
import numpy as np
from halo import Halo

# Setup API keys for cohere
load_dotenv()

co = cohere.Client(os.getenv("COHERE_API_KEY"))

# Use sqlite to store the books, chapters, and embeddings, run the "migrations" and setup the tables
connection = sqlite3.connect("default.db")
connection.execute(
    """
CREATE TABLE IF NOT EXISTS books (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    UNIQUE(title)
);
"""
)
connection.execute(
    """
CREATE TABLE IF NOT EXISTS chapters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    book_id INTEGER NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    FOREIGN KEY (book_id) REFERENCES books(id),
    UNIQUE(book_id, title)
);
"""
)
connection.execute(
    """
CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chapter_id INTEGER NOT NULL,
    content TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    embedding BLOB NOT NULL,
    FOREIGN KEY (chapter_id) REFERENCES chapters(id)
);
"""
)
connection.commit()


def store_book(path) -> str:
    """Given a path to an epub file, extracts the chapters and stores both book and chapters and returns the id, skips creation if already exists"""
    book = epub.read_epub(path)
    title = book.get_metadata("DC", "title")[0][0]
    cursor = connection.execute("SELECT id FROM books WHERE title=?", (title,))
    book_row = cursor.fetchone()
    if book_row is not None:
        return book_row[0]

    # Create the book in the database and return the id
    connection.execute("INSERT INTO books (title) VALUES (?)", (title,))
    cursor = connection.execute("SELECT id FROM books WHERE title=?", (title,))
    book_id = cursor.fetchone()[0]

    # First we load the chapter title and their content
    items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
    chapters = []
    for item in items:
        chapter_name = item.get_name()
        soup = BeautifulSoup(item.get_body_content(), "html.parser")
        text = " ".join([p.text for p in soup.find_all("p")])
        chapters.append((chapter_name, text))

    # Insert the chapters into the database, do nothing if exists
    for chapter in chapters:
        connection.execute(
            "INSERT OR IGNORE INTO chapters (book_id, title, content) VALUES (?, ?, ?)",
            (book_id, chapter[0], chapter[1]),
        )

    connection.commit()
    return book_id


def process_chapters(book_id) -> list:
    """Given a book_id, processes the chapters and stores the embeddings in the database, skips if already exists. Returns the chapters."""

    cursor = connection.execute("SELECT title FROM books WHERE id=?", (book_id,))
    book_title = cursor.fetchone()[0]

    cursor = connection.execute(
        "SELECT id, title, content FROM chapters WHERE book_id=?", (book_id,)
    )
    chapters = cursor.fetchall()

    for chapter in tqdm(chapters):
        chapter_id = chapter[0]
        chapter_title = chapter[1]
        content = chapter[2]

        # skip empty content
        if not content:
            continue

        # check if we have embeddings for this chapter already
        cursor = connection.execute(
            "SELECT COUNT(*) FROM embeddings WHERE chapter_id=?", (chapter_id,)
        )
        count = cursor.fetchone()[0]
        if count > 0:
            continue

        # split the chapter into chunks of 200 words
        words = content.split()
        chunks = [" ".join(words[i : i + 200]) for i in range(0, len(words), 200)]

        # reformat chunk with a title
        formatted_chunks = [
            f"Excerpt from {book_title}, {chapter_title}\n\n" + chunk
            for chunk in chunks
        ]

        # generate embeddings for each chunk
        embeddings = co.embed(
            texts=formatted_chunks,
            model="embed-english-light-v3.0",
            input_type="search_document",
        ).embeddings
        if not embeddings:
            continue

        # store the embeddings in the database
        for i, embedding in tqdm(enumerate(embeddings)):
            connection.execute(
                "INSERT INTO embeddings (chapter_id, content, chunk_index, embedding) VALUES (?, ?, ?, ?)",
                (chapter_id, formatted_chunks[i], i, json.dumps(embedding)),
            )
        connection.commit()
    return chapters


def get_embeddings(chapter_ids):
    """Given a list of chapter_ids, returns the embedding vector and the chunk content for those chapters"""

    cursor = connection.execute(
        "SELECT embedding, content FROM embeddings WHERE chapter_id IN ({})".format(
            ",".join(["?"] * len(chapter_ids))
        ),
        chapter_ids,
    )

    return cursor.fetchall()


def get_preceding_chapter_ids(book_id, current_chapter_title):
    cursor = connection.execute(
        "SELECT id FROM chapters WHERE book_id=? AND id <= (SELECT id FROM chapters WHERE book_id=? AND title=?)",
        (book_id, book_id, current_chapter_title),
    )
    return [row[0] for row in cursor.fetchall()]


def compute_cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    """Compute the cosine similarity between two vectors"""

    return (u @ v) / (np.linalg.norm(u) * np.linalg.norm(v))


if __name__ == "__main__":
    questions = [
        inquirer.Path(
            "path",
            message="Select an epub file",
            path_type=inquirer.Path.FILE,
            exists=True,
            default="books/romeo_and_juliet.epub",
        )
    ]
    answers = inquirer.prompt(questions)

    print("processing book...")
    book_id = store_book(answers["path"])

    print("processing chapters...")
    chapters = process_chapters(book_id)

    print("preparing for qa...")
    questions = [
        inquirer.List(
            "current_chapter",
            message="What chapter are you currently reading?",
            choices=[chapter[1] for chapter in chapters],
        ),
    ]
    current_chapter_title = inquirer.prompt(questions)["current_chapter"]
    print(current_chapter_title)

    print("finding chapters...")
    chapter_ids = get_preceding_chapter_ids(book_id, current_chapter_title)

    print("setting up vector database...")
    embeddings = get_embeddings(chapter_ids)

    # Main loop for the conversation
    chat_history = []
    while True:
        # get the next message from the user
        query = input("USER (/q to exit) > ")
        if query == "/q":
            break

        # get the embedding vector for the query
        query_embedding = co.embed(
            texts=[query],
            model="embed-english-light-v3.0",
            input_type="search_document",
        ).embeddings[0]

        # perform the cosine similarity ordering within the process since we don't have that many embeddings
        similarities = [
            compute_cosine_similarity(
                query_embedding, np.array(json.loads(embedding[0]))
            )
            for embedding in embeddings
        ]
        ordered_embeddings = [
            embedding for _, embedding in sorted(zip(similarities, embeddings))
        ]

        # rerank the embeddings and keep only the top 10
        reranked_embeddings = [
            ranking.document["text"]
            for ranking in co.rerank(
                query,
                [embedding[1] for embedding in ordered_embeddings],
                model="rerank-english-v2.0",
                top_n=10,
            ).results
        ]

        # generate an answer for the user and print it out
        spinner = Halo(text="Chatbot is thinking...", spinner="dots")
        spinner.start()
        prompt = """
You are an expert at answering questions about books. Today you are going to help answer questions from a user reading a book.

Here are some important rules to follow:
- You are not allowed to spoil the plot of the book. Even if you know the answer to a question, unless it is explicitly stated in the documents provided, you should not give it away.
- You must only use the information provided to you in the documents to answer the user's questions
- Only answer the question asked by the user, do not provide any extra information
"""
        answer = co.chat(
            query,
            chat_history=chat_history,
            preamble_override=prompt.strip(),
            model="command-nightly",
            documents=[{"snippet": embedding} for embedding in reranked_embeddings],
            prompt_truncation="AUTO",
        )
        spinner.stop()
        print(f"CHATBOT > {answer.text}")

        chat_history += [
            {"user_name": "User", "text": query},
            {"user_name": "Chatbot", "text": answer.text},
        ]
