# Anti Spoiler QA

A quick demo of a RAG pipeline that answers questions about a book, using excerpts from the book as it's only source of knowledge. Uses Cohere as the underlying LLM and Embedding Vector provider.

![](./demo.gif)

## Quickstart

I used Romeo and Juliet from Project Gutenberg as an example epub, you can find and download that here: https://www.gutenberg.org/ebooks/1513.epub.noimages

1. Install pyenv (`brew install pyenv` on OSX)
2. Run `./deps.sh`
3. Create a new file named `.env` and add a variable with your Cohere API key `COHERE_API_KEY`
4. Run `python main.py` and follow the prompts!

## Limitations

This was a pretty quick and rudimentary take at building this pipeline, I would like to add better metdata and chapter name parsing as it's really dependent on the name of the internal HTML files right now
