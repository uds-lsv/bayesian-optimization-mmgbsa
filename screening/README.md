Download the ZINC22 drug-like subset from [here](https://cartblanche22.docking.org/tranches/2d).

Make sure `rdkit` and `tqdm` are installed.

```bash
$ python similarity.py --help

usage: similarity.py [-h] [-i INPUT] -l LIGANDS [LIGANDS ...] -o OUTPUT [-t THRESHOLD] [-b BATCH_SIZE]

Screen ZINC22 based on maximum common substructure

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Input query molecules as SMILES string.
  -l LIGANDS [LIGANDS ...], --ligands LIGANDS [LIGANDS ...]
                        Gzip compressed input ligand file with SMILES strings.
  -o OUTPUT, --output OUTPUT
                        Output directory
  -t THRESHOLD, --threshold THRESHOLD
                        Similarity threshold
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Batch size
```

Once screening has finished, combine all results
```bash
cat <outputdir>/* > similarity_scores.csv
```
where outputdir is the same path as used during screening.


To also get counts of how many molecules come from which tranche, run
```bash
wc -l <outputdir>/*.smi.csv | sed '$ d; s/^ *\([0-9]*\) \(.*\)$/\1,"\2"/' > tranche_counts.csv
```


# Searching ZINC
```bash
$ python search_zinc.py

usage: search_zinc.py [-h] -i INPUTS [INPUTS ...] -l LIGANDS [LIGANDS ...] -o OUTPUT [-b BATCH_SIZE]

Search compressed ZINC22 for a ZINC ID's

options:
  -h, --help            show this help message and exit
  -i INPUTS [INPUTS ...], --inputs INPUTS [INPUTS ...]
                        ZINC ids of query molecules.
  -l LIGANDS [LIGANDS ...], --ligands LIGANDS [LIGANDS ...]
                        Gzip compressed input ligand file with SMILES strings.
  -o OUTPUT, --output OUTPUT
                        Output directory
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Batch size
```
