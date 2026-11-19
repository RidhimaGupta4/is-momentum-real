# Third-party notices

## StatsBomb Open Data
All analysis in this repository is based on StatsBomb Open Data
(https://github.com/statsbomb/open-data, commit `4b73468fc5b0f1950f9f66fada70ad3a4f9327cb`), used under the
StatsBomb Public Data User Agreement (`LICENSE.pdf` in that repository).

- No StatsBomb data are included in this repository. `fetch_data.sh` downloads the files directly from
  StatsBomb's repository, and every file is checked against its git blob hash in `data_manifest.tsv`.
- The results in `results/`, the figures in `figures/` and the interactive page in `docs/` are derived
  analysis of StatsBomb data. They are shared for non-commercial use, as the agreement requires, and are not
  covered by this repository's MIT licence.
- The StatsBomb logo (`assets/statsbomb_logo.png`, `explorer/statsbomb_logo_small.png`, a resized copy) is
  taken unmodified from `img/SB - Icon Lockup - Colour positive.png` in the same repository and commit. It
  is shown to credit StatsBomb as the data source, as the agreement requires; it remains StatsBomb's
  trademark and is not covered by this repository's licence.
- The user agreement asks users to register at statsbomb.com/resource-centre. When this project was
  published (November 2026), that address redirected to Hudl's product page, StatsBomb's open-data README no
  longer mentioned registration, and no registration form could be found. The data are used under the
  agreement as published in the repository (`LICENSE.pdf`), with the attribution it requires.

## Barlow and Barlow Condensed fonts
Embedded in the interactive page. Copyright 2017 The Barlow Project Authors
(https://github.com/jpt/barlow), licensed under the SIL Open Font License, Version 1.1
(full text in `explorer/fonts/OFL.txt`).

## Method credit
The model is a log-linear self-exciting point process (in the family introduced by Hawkes, 1971) fitted through its
exact piecewise-exponential likelihood; goodness of fit uses the time-rescaling theorem (Brown et al., 2002).
Independent implementation; see README references.
