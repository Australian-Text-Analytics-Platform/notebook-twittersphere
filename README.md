# Australian Twittersphere (AuTS) aggregated data - Notebook

[![Binder](https://binderhub.rc.nectar.org.au/badge_logo.svg)](https://binderhub.rc.nectar.org.au/v2/gh/Australian-Text-Analytics-Platform/notebook-twittersphere/HEAD?labpath=exploration.ipynb)

In this notebook (exploration.ipynb) you will be able to find ways of getting data about the QUT Digital Observatory's Australian Twittersphere (AuTS) collection. For more information on the AuTS go here - https://www.digitalobservatory.net.au/resources/australian-twittersphere/

The purpose of this notebook is to explore the data contained in the AuTS to see if the data you are interested in (e.g. topics, words, etc) are held in the collection.

The notebook/s can access:

+ N-grams for the entire collection - 1-grams and 3-grams plus 3-grams with emojis as single words
+ Domains/URLs for the entire collection
+ Hashtags for the entire collection

&#x1F6D1; Note: Using this notebook will download data from https://data.ldaca.edu.au portal's aggregated collection

To run the notebook locally, run `uv run --with jupyter jupyter lab` in your terminal. 

## Changing the notebook

To add a dependency, run `uv add libraryname`; be sure to update `requirements.txt` by running `uv export --format requirements.txt --no-hashes > requirements.txt` (this copies the output of the export command into requirements.txt, where binder can read them)
