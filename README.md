# Mining Footprints CNN

## A note on imagery sources

This project currently uses PlanetScope imagery, which requires a license from Planet to download.
I originally wanted to use Sentinel-2 imagery for this project, but the ESA was in the middle of a infrastructure migration and didn't want to write code to download from an API that was going to be phased out.
I'd love to build Sentinel-2 into this project though, if you could help out with that reach out!

For now, to use this project you'll need to save your Planet API key to `.planet_key.txt` in this directory.

## Getting started

1. Update `config.toml`

- Set your project path!
- Limit which countries you want to search

2. Download mine footprints

```sh
python src/download_footprints.py
```

3. Extract mine footprints
```sh
python src/extract_footprints.py
```

4. Search Planet for imagery

```sh
python src/planet_search.py
```

5. Download imagery from planet

```sh
python src/planet_download.py
```

...
